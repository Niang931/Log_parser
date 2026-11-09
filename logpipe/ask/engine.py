"""From a validated QuerySpec to an answer, in rounds. Each round is a list of reads
(:class:`~logpipe.ask.query.Query`) that are shown for approval before any of them runs; what
the next round needs depends on what the last one found (a root-cause search first looks for
when things broke, then reads around that moment).

The engine itself never calls the LLM except through ``ctx.adapter`` in :mod:`synth` (the final
explanation) and never reads anything that isn't one of the approved queries.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ..server.stores import StoreReads
from ..settings import get_settings
from .query import Filters, Query
from .spec import QuerySpec, time_range

MAX_ROUNDS = 6  # discovery, evidence, one widening, up to three follow-ups


@dataclass
class Ctx:
    store: StoreReads
    templates: dict[int, str]  # id -> text, canonical ids only
    canonical: Callable[[int], int] = lambda t: t
    scorer_for: Callable[[str], Any] = lambda service: None  # service -> LogBERT scorer
    agents: Callable[[], list[dict]] = list
    alerts: Callable[[int], list[dict]] = lambda limit: []
    search_templates: Callable[[str, int], list[tuple[int, float]]] = lambda text, k: []
    neighbours: Callable[[int, int], list[tuple[int, float]]] = lambda tid, k: []  # LogBERT vectors
    shutdown: set[int] = field(default_factory=set)
    adapter: Any = None
    now: datetime = field(default_factory=lambda: datetime.now(UTC))
    window: int = 20  # LogBERT window length
    use_logbert: bool = True
    use_semantic: bool = True
    budget_tokens: int = 6000


@dataclass
class Round:
    n: int
    purpose: str
    queries: list[Query]


@dataclass
class Outcome:
    """What ``advance`` returns: the next round, or the finished result."""
    round: Round | None = None
    result: dict | None = None
    state: dict = field(default_factory=dict)


def _q(n: int, i: int, tool: str, title: str, why: str, compiled: str, engine: str, **args) -> Query:
    return Query(id=f"r{n}q{i}", tool=tool, title=title, why=why, args=args, compiled=compiled, engine=engine)


def local(n: int, i: int, tool: str, title: str, why: str, describe: str, **args) -> Query:
    """A read that touches no log data in the store (agent heartbeats, the template index)."""
    return _q(n, i, tool, title, why, describe, "local", **args)


def log_read(ctx: Ctx, n: int, i: int, tool: str, title: str, why: str, **args) -> Query:
    start, end = (datetime.fromisoformat(args[k]) for k in ("start", "end"))
    f = Filters(**args.get("filters", {})).builder()
    c = ctx.store.compiler
    if tool == "rows":
        compiled = c.rows(f, start, end, args.get("limit", 200), args.get("newest_first", False))
    elif tool == "template_counts":
        compiled = c.template_counts(f, start, end)
    elif tool == "buckets":
        compiled = c.buckets(f, start, end, args["step"], args.get("by", "template_id"))
    elif tool == "aggregate":
        spec = QuerySpec.model_validate(args["spec"])
        assert spec.aggregate is not None
        compiled = c.build({**span(start, end), "filters": Filters.of(spec).builder(), "group_by": spec.aggregate.group_by,
                            "measure": {"fn": spec.aggregate.fn, "field": spec.aggregate.field},
                            "over_time": spec.aggregate.over_time, "limit": spec.limit})
    else:
        raise ValueError(f"unknown read {tool}")
    return _q(n, i, tool, title, why, compiled, c.engine, **args)


def span(start: datetime, end: datetime) -> dict:
    return {"start": start.astimezone(UTC).isoformat(), "end": end.astimezone(UTC).isoformat()}


def execute(q: Query, ctx: Ctx) -> list[dict]:
    """Run one approved read. Log reads run the compiled text exactly as it was shown."""
    if q.status != "approved":
        raise PermissionError(f"query {q.id} is {q.status}, not approved")
    if q.engine != "local":
        cfg = get_settings().ask
        return ctx.store.records(q.compiled, cfg.row_cap if q.tool == "rows" else cfg.agg_cap)
    if q.tool == "agents":
        return [{"host": a.get("hostname"), "last_seen": a.get("last_seen"), "online": a.get("online")}
                for a in ctx.agents()]
    if q.tool == "alerts":
        start, end = q.args["start"], q.args["end"]
        return [a for a in ctx.alerts(500) if a["window_end"] >= start and a["window_start"] <= end]
    raise ValueError(f"unknown local read {q.tool}")


# --- planning ---------------------------------------------------------------------------
def describe_templates(found: list[tuple[int, float]], templates: dict[int, str]) -> str:
    return "; ".join(f"#{t} {templates.get(t, '?')[:60]} ({s:.2f})" for t, s in found)


def lookup_templates(spec: QuerySpec, ctx: Ctx) -> tuple[list[int], str]:
    """Templates an event/semantic question is about, from the template catalog (which holds
    template text, not log lines): given ids, else the closest matches to its description."""
    if spec.template_ids:
        return [ctx.canonical(t) for t in spec.template_ids], "templates named in the question"
    text = spec.semantic_query or spec.anchor_event or " ".join(spec.keywords)
    found = [(t, s) for t, s in ctx.search_templates(text, 5) if s > 0] if ctx.use_semantic else []
    return [t for t, _ in found], f"closest templates to {text!r}: {describe_templates(found, ctx.templates) or 'none'}"


def first_round(spec: QuerySpec, ctx: Ctx) -> Outcome:
    if spec.intent == "general":
        return Outcome(result={"kind": "general", "answer": spec.answer or "(no answer)",
                               "note": "From general knowledge, not from your logs."})
    if spec.intent == "rca":
        from .rca import rca_first_round

        return rca_first_round(spec, ctx)
    start, end = time_range(spec, ctx.now)
    f = asdict(Filters.of(spec))
    if spec.intent == "summary":
        from .rca import summary_round

        return summary_round(spec, ctx, start, end)
    if spec.aggregate:
        return Outcome(Round(1, "count", [log_read(ctx, 1, 1, "aggregate", "Count",
                                                    f"{spec.aggregate.fn} per {', '.join(spec.aggregate.group_by) or 'range'}",
                                                    spec=spec.model_dump(mode="json"), **span(start, end))]))
    note = ""
    if spec.intent in ("event", "semantic"):
        tids, note = lookup_templates(spec, ctx)
        if not tids and not spec.keywords:
            return Outcome(result={"kind": "clarify", "message": "No template looks like that event. Name it with "
                                   "words the lines contain (in quotes), or pick a template on the Templates page.",
                                   "note": note})
        f["template_ids"] = sorted(set(f["template_ids"]) | set(tids)) if tids else f["template_ids"]
        if tids:
            f["keywords"] = []  # the templates are the match; the words described them
    queries = [log_read(ctx, 1, 1, "rows", "Matching lines", "the newest lines that match: " + Filters(**f).describe(),
                        filters=f, limit=spec.limit, newest_first=True, **span(start, end))]
    if spec.intent in ("event", "semantic"):
        queries.append(log_read(ctx, 1, 2, "template_counts", "When each template occurred",
                                "first and last time per template, to say when the event happened",
                                filters=f, **span(start, end)))
    return Outcome(Round(1, "search", queries), state={"note": note})


def advance(spec: QuerySpec, state: dict, done: Round, outputs: dict[str, list[dict]], ctx: Ctx) -> Outcome:
    """The next round, or the result, from what the approved reads of ``done`` returned."""
    if spec.intent in ("rca", "summary"):
        from .rca import rca_advance

        return rca_advance(spec, state, done, outputs, ctx)
    q = done.queries[0]
    if q.id not in outputs:
        return Outcome(result={"kind": "clarify", "message": "The search was not approved, so there is nothing to show."})
    rows = outputs[q.id]
    if q.tool == "aggregate":
        cols = list(rows[0]) if rows else []
        return Outcome(result={"kind": "table", "columns": cols, "rows": [[r[c] for c in cols] for r in rows],
                               "compiled": q.compiled})
    result = {"kind": "lines", "lines": [_line(r, ctx) for r in rows], "note": state.get("note", "")}
    if len(done.queries) > 1 and done.queries[1].id in outputs:
        result["templates"] = [{"template_id": ctx.canonical(int(c["template_id"])), "service": c["service"],
                                "count": c["n"], "first": c["first"], "last": c["last"],
                                "text": ctx.templates.get(ctx.canonical(int(c["template_id"])), "")}
                               for c in outputs[done.queries[1].id]]
    return Outcome(result=result)


def _line(r: dict, ctx: Ctx) -> dict:
    tid = ctx.canonical(int(r["template_id"]))
    return {"log_id": r["log_id"], "timestamp": r["timestamp"], "host": r["host"], "service": r["service"],
            "level": r["level"], "template_id": tid, "message": r["message"]}


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z0-9]{2,}", text.lower())}


def overlap_search(templates: dict[int, str]) -> Callable[[str, int], list[tuple[int, float]]]:
    """Word-overlap search over template text: the fallback when no embedding index is built."""
    def search(text: str, k: int) -> list[tuple[int, float]]:
        q = words(text)
        if not q:
            return []
        scored = [(t, len(q & words(s)) / len(q)) for t, s in templates.items()]
        return sorted((x for x in scored if x[1] > 0), key=lambda x: -x[1])[:k]
    return search


def clamp_now(t: datetime, ctx: Ctx) -> datetime:
    return min(t, ctx.now + timedelta(minutes=1))


def run_unattended(spec: QuerySpec, ctx: Ctx) -> dict:
    """Every round approved as planned: for the eval and tests only (the server never does this
    unless approval is switched off). The result carries the rounds that ran."""
    out, state, rounds = first_round(spec, ctx), {}, []
    while out.result is None and len(rounds) < MAX_ROUNDS:
        state |= out.state
        r = out.round
        assert r is not None
        outputs = {}
        for q in r.queries:
            q.status = "approved"
            outputs[q.id] = execute(q, ctx)
            q.status = "done"
        rounds.append(r)
        out = advance(spec, state, r, outputs, ctx)
    result = out.result or {"kind": "clarify", "message": "too many rounds"}
    return result | {"rounds": [{"n": r.n, "purpose": r.purpose, "queries": [asdict(q) for q in r.queries]} for r in rounds]}
