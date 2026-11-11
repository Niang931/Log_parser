"""Follow-up reads the LLM may ask for after reading the bundle: typed moves, not SQL. Each is
checked here (known services only, an id that really occurs in the bundle, a window it may not
stretch past ``move_max_hours``), compiled by the store adapter like every other read, and shown
for approval before it runs. What comes back goes through the same bundle (lines get refs, rates
become series), so the LLM never sees raw query output.

* ``lines``: lines matching keywords in some services, in a window it picks around the anchor;
* ``trace``: every line, in any service, that carries an id copied from the bundle (a request,
  trace or order id): follows one request across services;
* ``rate``: error/warn lines (or all lines) per service per time step: when a service started
  failing, or went quiet;
* ``neighbours``: the templates that logged errors in the same minutes in *any* service, beyond
  the group asked about: a shared dependency (DNS, the database) shows up here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, Field

from .engine import Ctx, Round, log_read, span
from .models import Bucket, LogLine, Report, Series, TemplateCount
from .query import Filters, Query


class MoveKind(StrEnum):
    LINES = "lines"
    TRACE = "trace"
    RATE = "rate"
    NEIGHBOURS = "neighbours"


class MoveRequest(BaseModel):
    """One follow-up read, as the LLM asks for it (part of the answer's structured output)."""

    kind: MoveKind = Field(
        description=(
            "lines: keyword search in some services; trace: every line in any service carrying "
            "an id; rate: error/warn lines per service over time; neighbours: errors in any "
            "service around the anchor"
        )
    )
    why: str = Field("", description="what this read would settle")
    keywords: list[str] = Field(
        default_factory=list, description="lines/rate: words the lines contain"
    )
    services: list[str] = Field(
        default_factory=list,
        description="lines/rate: services to read (from the bundle or the groups)",
    )
    value: str = Field(
        "",
        description="trace: an id copied exactly from a line (request id, trace id, order id)",
    )
    minutes_before: int = Field(
        0, description="start this many minutes before the anchor (0: the default window)"
    )
    minutes_after: int = Field(
        0, description="end this many minutes after the anchor (0: the default window)"
    )
    all_lines: bool = Field(
        False,
        description="rate: count every line, not only errors and warnings (to see a service go quiet)",
    )


# minutes before/after the anchor when the LLM leaves the window at 0; None: the wide RCA window
DEFAULT_WINDOW: dict[MoveKind, tuple[float, float] | None] = {
    MoveKind.LINES: None,
    MoveKind.TRACE: None,
    MoveKind.RATE: None,
    MoveKind.NEIGHBOURS: (5, 2),
}


@dataclass
class Move:
    """A request that passed the checks, ready to compile into a read."""

    kind: MoveKind
    filters: Filters
    start: datetime
    end: datetime
    why: str = ""
    value: str = ""
    all_lines: bool = False
    services: list[str] = field(default_factory=list)

    def describe(self) -> str:
        what = {
            MoveKind.LINES: self.filters.describe(),
            MoveKind.TRACE: f"every line carrying {self.value!r}",
            MoveKind.RATE: f"{'all' if self.all_lines else 'error/warn'} lines per minute of "
            + ", ".join(self.services),
            MoveKind.NEIGHBOURS: "errors in any service around the anchor",
        }[self.kind]
        return f"{what}, {self.start:%H:%M}-{self.end:%H:%M} UTC"


def check_moves(requests: list[MoveRequest], report: Report, ctx: Ctx) -> list[Move]:
    """The requests that hold up, as moves, deduplicated and capped; the rest are dropped without
    a word (they were never going to run)."""
    known = ctx.services or {e.service for e in report.evidence if e.service}
    text = "\n".join(e.message for e in report.evidence)
    cfg, out, seen = ctx.rca, [], set()
    longest = cfg.move_max_hours * 60
    for r in requests:
        services = [s for s in r.services if s in known][:5]
        keywords = [k.strip()[:100] for k in r.keywords if k.strip()][:3]
        value = r.value.strip()
        # an id must come from the bundle: the LLM follows evidence, it doesn't guess
        if r.kind == MoveKind.TRACE and not (
            4 <= len(value) <= 100 and " " not in value and value in text
        ):
            continue
        if r.kind == MoveKind.LINES and not (keywords or services):
            continue
        if r.kind == MoveKind.RATE and not services:
            continue
        default = DEFAULT_WINDOW[r.kind] or (cfg.wide_before_minutes, cfg.wide_after_minutes)
        before = min(max(r.minutes_before, 0) or default[0], longest)
        after = min(max(r.minutes_after, 0) or default[1], longest)
        at = report.anchor.at
        start, end = at - timedelta(minutes=before), min(at + timedelta(minutes=after), ctx.now)
        if start >= end:
            continue
        filters = {
            MoveKind.TRACE: Filters(keywords=[value]),
            MoveKind.NEIGHBOURS: Filters(levels=["error", "warn"]),
            MoveKind.RATE: Filters(
                services=services,
                keywords=keywords,
                levels=[] if r.all_lines else ["error", "warn"],
            ),
            MoveKind.LINES: Filters(services=services, keywords=keywords),
        }[r.kind]
        key = (r.kind, repr(filters), start, end)
        if key not in seen:
            seen.add(key)
            out.append(Move(r.kind, filters, start, end, r.why[:200], value, r.all_lines, services))
    return out[: cfg.moves_per_round]


def step_for(start: datetime, end: datetime) -> int:
    """At most 60 steps, each a whole number of minutes."""
    return 60 * max(1, math.ceil((end - start).total_seconds() / 60 / 60))


def move_round(n: int, moves: list[Move], ctx: Ctx, purpose: str) -> Round:
    queries: list[Query] = []
    for i, m in enumerate(moves, 1):
        args = {"filters": asdict(m.filters), "move": m.kind, **span(m.start, m.end)}
        title, why = f"Follow-up ({m.kind}): {m.describe()}", m.why or "the model asked for this"
        if m.kind in (MoveKind.LINES, MoveKind.TRACE):
            queries.append(
                log_read(
                    ctx,
                    round_no=n,
                    query_no=i,
                    tool="rows",
                    title=title,
                    why=why,
                    limit=100 if m.kind == MoveKind.TRACE else 50,
                    **args,
                )
            )
        elif m.kind == MoveKind.RATE:
            queries.append(
                log_read(
                    ctx,
                    round_no=n,
                    query_no=i,
                    tool="buckets",
                    title=title,
                    why=why,
                    step=step_for(m.start, m.end),
                    by="service",
                    **args,
                )
            )
        else:
            queries.append(
                log_read(
                    ctx,
                    round_no=n,
                    query_no=i,
                    tool="template_counts",
                    title=title,
                    why=why,
                    **args,
                )
            )
    return Round(n, purpose, queries)


def varied(lines: list[LogLine], per: int = 3) -> list[LogLine]:
    """At most ``per`` lines of each (template, service): forty copies of one error say no more than three."""
    seen: dict[tuple, int] = {}
    out = []
    for line in lines:
        key = (line.template_id, line.service)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= per:
            out.append(line)
    return out


def absorb(report: Report, done: Round, outputs: dict[str, list[dict]], ctx: Ctx) -> Report:
    """The bundle with what the approved follow-up reads found: lines, and rates as series."""
    from .rca import add_lines

    cap = len(report.evidence) + ctx.rca.follow_up_lines
    for q in done.queries:
        if q.id not in outputs:
            continue
        rows, kind = outputs[q.id], MoveKind(q.args.get("move", MoveKind.LINES))
        why = f"follow-up {kind}: {q.why}"
        if q.tool == "rows":
            report.evidence = add_lines(
                report.evidence,
                varied([LogLine.model_validate(r) for r in rows]),
                ctx,
                why=why,
                cap=cap,
            )
        elif q.tool == "template_counts":
            counts = sorted((TemplateCount.model_validate(r) for r in rows), key=lambda c: c.first)
            lines = [
                LogLine(
                    log_id=c.log_id,
                    timestamp=c.first,
                    service=c.service,
                    host=c.host,
                    level=c.level,
                    template_id=c.template_id,
                    message=c.example,
                )
                for c in counts
            ]
            report.evidence = add_lines(report.evidence, lines[:10], ctx, why=why, cap=cap)
        elif q.tool == "buckets":
            start, end = (datetime.fromisoformat(q.args[k]) for k in ("start", "end"))
            step = int(q.args["step"])
            steps = max(1, math.ceil((end - start).total_seconds() / step))
            per: dict = {}
            for b in (Bucket.model_validate(r) for r in rows):
                if 0 <= b.bucket < steps:
                    per.setdefault(b.key, [0] * steps)[b.bucket] += b.n
            for s in q.args["filters"].get("services") or []:
                # a service with no line at all is the answer, not a gap
                per.setdefault(s, [0] * steps)
            levels = q.args["filters"].get("levels")
            for service, counts in sorted(per.items(), key=lambda kv: str(kv[0])):
                report.series.append(
                    Series(
                        ref=f"S{len(report.series) + 1}",
                        service=None if service is None else str(service),
                        what="error/warn lines" if levels else "lines",
                        start=start,
                        step_seconds=step,
                        counts=counts,
                    )
                )
    return report
