"""Root-cause analysis without the LLM: find when things broke (the anchor), read around it,
and rank what changed. Every number that decides a line's place is kept next to it, so the
report can say why each line is there. The numbers themselves are ``RcaSettings``.

Anchor, in priority order (the first that holds wins; none -> ask the user, never guess):
1. the time the user gave (or the incident's start);
2. silence: a service that was logging steadily went quiet (or its agent stopped heartbeating);
3. the first line of a shutdown-type template (OOM, SIGTERM, segfault, refused, disk full, ...);
4. the first sustained LogBERT alert (two or more alert windows within 5 minutes);
5. the first ERROR/WARN template that never occurred in the 24 hours before.

Window: [anchor - 30 min, anchor + 2 min], widened once to [anchor - 2 h, anchor + 5 min] when
fewer than 3 candidates stand out.

Score per (template, service) seen in the window = anomaly x rarity x proximity:
* anomaly: the largest of the line's level (error 0.8, warn 0.5, else 0.1), 0.9 for a shutdown
  template, and the share of its lines LogBERT did not expect (unseen or mispredicted);
* rarity: 1 for a template absent from the 24 h baseline, else how far its count in the window
  exceeds the baseline median for a window that long ((ratio - 1) / 9, clamped to 0.02..1);
* proximity: exp(-|first seen - anchor| / 10 min).
Ties go to the earlier template: a cause comes before its effects. A new template first seen up to
2 minutes before the strongest one gets anomaly 0.8 (a precursor), and the earliest template
scoring at least half the best leads the answer ("earliest in chain").
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta
from enum import IntEnum
from itertools import pairwise
from statistics import median

from pydantic import BaseModel

from .engine import Ctx, Outcome, Round, local, log_read, span
from .hints import value_hints
from .models import (
    AgentSeen,
    Alert,
    Anchor,
    Bucket,
    Candidate,
    ClarifyResult,
    Evidence,
    LogLine,
    RcaState,
    Report,
    TemplateCount,
    TimeSpan,
)
from .spec import QuerySpec, time_range


class Discover(IntEnum):
    """The reads of the first round, by position."""

    PER_MINUTE = 1
    TEMPLATES = 2
    BASELINE = 3
    AGENTS = 4
    ALERTS = 5


class Around(IntEnum):
    """The reads of an evidence round, by position."""

    TEMPLATES = 1
    BASELINE = 2
    ANCHOR_LINES = 3
    ALERTS = 4
    ERRORS = 5
    OTHERS = 6


def rows_of[M: BaseModel](
    done: Round, outputs: dict[str, list[dict]], i: int, model: type[M]
) -> list[M]:
    """What the round's ``i``-th read returned (nothing if it wasn't approved or failed)."""
    qid = f"r{done.n}q{i}"
    return [model.model_validate(r) for r in outputs.get(qid) or []]


def rca_first_round(spec: QuerySpec, ctx: Ctx) -> Outcome:
    cfg = ctx.rca
    if spec.anchor_time:
        anchor = Anchor(
            kind="user",
            at=spec.anchor_time,
            service=spec.services[0] if spec.services else None,
            why="the time given in the question",
        )
        return Outcome(
            evidence_round(2, anchor, ctx, spec), state=_state("evidence", anchor=anchor)
        )
    start, end = time_range(spec, ctx.now)
    queries = [
        log_read(
            ctx,
            round_no=1,
            query_no=Discover.PER_MINUTE,
            tool="buckets",
            title="Lines per service per minute",
            why="to see whether a service that was logging went quiet",
            filters=ctx.scope(spec),
            step=60,
            by="service",
            **span(start - cfg.before, end),
        ),
        log_read(
            ctx,
            round_no=1,
            query_no=Discover.TEMPLATES,
            tool="template_counts",
            title="Templates in the range",
            why="first time each template appeared: shutdown messages, new errors",
            filters=ctx.scope(spec),
            **span(start, end),
        ),
        log_read(
            ctx,
            round_no=1,
            query_no=Discover.BASELINE,
            tool="template_counts",
            title=f"Templates in the {cfg.baseline_hours:g} h before",
            why="which templates are normal, so new ones stand out",
            filters=ctx.scope(spec),
            **span(start - cfg.baseline, start),
        ),
        local(
            round_no=1,
            query_no=Discover.AGENTS,
            tool="agents",
            title="Agent heartbeats",
            why="an agent that stopped reporting marks when a host went quiet",
            describe="server.agents(): hostname, last_seen, online (control store)",
        ),
    ]
    if ctx.use_logbert:
        queries.append(
            local(
                round_no=1,
                query_no=Discover.ALERTS,
                tool="alerts",
                title="LogBERT alerts",
                why="windows the anomaly model flagged in the range",
                describe="reads.alerts(500) filtered to the range (anomaly_alerts)",
                **span(start, end),
            )
        )
    return Outcome(
        Round(1, "find when it broke", queries),
        state=_state("discover", range=TimeSpan(start=start, end=end)),
    )


def summary_round(spec: QuerySpec, ctx: Ctx, start: datetime, end: datetime) -> Outcome:
    anchor = Anchor(kind="range", at=end, why="a summary covers the whole range")
    return Outcome(
        evidence_round(1, anchor, ctx, spec, window=(start, end)),
        state=_state("evidence", anchor=anchor),
    )


def evidence_round(
    n: int,
    anchor: Anchor,
    ctx: Ctx,
    spec: QuerySpec,
    wide: bool = False,
    window: tuple[datetime, datetime] | None = None,
) -> Round:
    cfg = ctx.rca
    at = anchor.at
    start, end = window or (
        at - (cfg.wide_before if wide else cfg.before),
        min(at + (cfg.wide_after if wide else cfg.after), ctx.now),
    )
    length = max(int((end - start).total_seconds()), 60)
    f = ctx.scope(spec)
    queries = [
        log_read(
            ctx,
            round_no=n,
            query_no=Around.TEMPLATES,
            tool="template_counts",
            title="Templates around the anchor",
            why="what was logged, and when each kind first appeared",
            filters=f,
            **span(start, end),
        ),
        log_read(
            ctx,
            round_no=n,
            query_no=Around.BASELINE,
            tool="buckets",
            title=f"The same window, over the {cfg.baseline_hours:g} h before",
            why=f"counts per template in {length // 60}-minute slices of the previous day: the baseline",
            filters=f,
            step=length,
            **span(start - cfg.baseline, start),
        ),
    ]
    if anchor.service and anchor.kind != "range":
        lb = ctx.use_logbert and ctx.scorer_for(anchor.service) is not None
        queries.append(
            log_read(
                ctx,
                round_no=n,
                query_no=Around.ANCHOR_LINES,
                tool="rows",
                title=f"{anchor.service} lines just before the anchor",
                why="its last lines before the anchor"
                + (", which LogBERT re-reads to mark lines it did not expect" if lb else ""),
                filters={"services": [anchor.service]},
                limit=2000,
                **span(at - timedelta(minutes=5), min(at + timedelta(minutes=1), ctx.now)),
            )
        )
    if ctx.use_logbert:
        queries.append(
            local(
                round_no=n,
                query_no=Around.ALERTS,
                tool="alerts",
                title="LogBERT alerts in the window",
                why="windows the anomaly model flagged",
                describe="reads.alerts(500) filtered to the window (anomaly_alerts)",
                **span(start, end),
            )
        )
    # hints describe the moment it broke: a widened window would mix in older, unrelated trouble
    near = window or (at - cfg.before, min(at + cfg.after, ctx.now))
    for i, levels, title in (
        (Around.ERRORS, ["error", "warn"], "Error and warning lines near the anchor"),
        (Around.OTHERS, ["info", "debug"], "The other lines near the anchor"),
    ):
        queries.append(
            log_read(
                ctx,
                round_no=n,
                query_no=i,
                tool="rows",
                title=title,
                why="with their extracted fields: which host or value the errors "
                "concentrate on, compared with the other lines",
                filters=f | {"levels": levels},
                limit=ctx.ask.row_cap,
                newest_first=True,
                payload=True,
                **span(*near),
            )
        )
    return Round(
        n,
        "read around "
        + ("the range" if anchor.kind == "range" else "the anchor")
        + (" (wider)" if wide else ""),
        queries,
    )


def _state(phase: str, **fields) -> dict:
    return RcaState(phase=phase, **fields).model_dump(mode="json")  # type: ignore[arg-type]


def rca_advance(spec: QuerySpec, raw_state: dict, done: Round, outputs: dict, ctx: Ctx) -> Outcome:
    state = RcaState.model_validate(raw_state)
    if state.phase == "discover":
        assert state.range is not None
        anchor = find_anchor(spec, done, outputs, ctx, state.range)
        if anchor is None:
            return Outcome(
                result=ClarifyResult(
                    message=(
                        "I couldn't tell when it broke: no service went quiet, no shutdown or new error message, no "
                        "anomaly alerts in that range. When did it happen (e.g. 'around 14:05'), and which service?"
                    )
                ),
                state=_state("clarify"),
            )
        return Outcome(
            evidence_round(done.n + 1, anchor, ctx, spec), state=_state("evidence", anchor=anchor)
        )
    if state.phase in ("evidence", "wide"):
        assert state.anchor is not None
        anchor = state.anchor
        cands, lines, notes = rank(spec, anchor, done, outputs, ctx)
        hints = value_hints(
            rows_of(done, outputs, Around.ERRORS, LogLine),
            rows_of(done, outputs, Around.OTHERS, LogLine),
            ctx.rca,
        )
        strong = [c for c in cands if c.score >= ctx.rca.strong]
        if (
            state.phase == "evidence"
            and len(strong) < ctx.rca.min_strong
            and anchor.kind != "range"
        ):
            return Outcome(
                evidence_round(done.n + 1, anchor, ctx, spec, wide=True),
                state=_state("wide", anchor=anchor),
            )
        report = Report(
            intent=spec.intent,
            anchor=anchor,
            candidates=cands,
            evidence=lines,
            notes=notes,
            hints=hints,
            groups=ctx.groups_of({c.service for c in cands} | {e.service for e in lines}),
        )
        return synthesize_step(spec, report, ctx, state, done.n)
    if state.phase == "follow_up":
        assert state.report is not None
        from .moves import absorb

        report = absorb(state.report, done, outputs, ctx)
        return synthesize_step(spec, report, ctx, state, done.n)
    raise ValueError(f"unknown phase {state.phase!r}")


def synthesize_step(
    spec: QuerySpec, report: Report, ctx: Ctx, state: RcaState, last_round: int
) -> Outcome:
    from .moves import move_round
    from .synth import synthesize

    answer, moves = synthesize(spec, report, ctx)
    report.answer = answer
    if moves and state.follow_ups < ctx.rca.follow_ups:
        purpose = f"follow-up {state.follow_ups + 1} of {ctx.rca.follow_ups}"
        return Outcome(
            move_round(last_round + 1, moves, ctx, purpose),
            state=_state("follow_up", report=report, follow_ups=state.follow_ups + 1),
        )
    return Outcome(result=report, state=_state("done"))


def find_anchor(
    spec: QuerySpec, done: Round, outputs: dict, ctx: Ctx, rng: TimeSpan
) -> Anchor | None:
    focus = set(spec.services)
    counts = rows_of(done, outputs, Discover.TEMPLATES, TemplateCount)
    if found := silence(
        rows_of(done, outputs, Discover.PER_MINUTE, Bucket), counts, rng, focus, ctx
    ):
        return found
    for a in rows_of(done, outputs, Discover.AGENTS, AgentSeen):
        if a.last_seen and not a.online and rng.start <= a.last_seen <= rng.end:
            return Anchor(
                kind="silence",
                at=a.last_seen,
                host=a.host,
                why=f"agent {a.host} stopped sending heartbeats at {a.last_seen:%H:%M:%S}",
            )
    shutdown = [c for c in counts if ctx.canonical(c.template_id) in ctx.shutdown]
    if shutdown:
        c = min(
            [c for c in shutdown if not focus or c.service in focus] or shutdown,
            key=lambda c: c.first,
        )
        tid = ctx.canonical(c.template_id)
        return Anchor(
            kind="shutdown",
            at=c.first,
            service=c.service,
            host=c.host,
            template_id=tid,
            message=c.example[: ctx.rca.line_chars],
            why=f"first shutdown-type message: {ctx.templates.get(tid, '')[:80]}",
        )
    alerts = sorted(rows_of(done, outputs, Discover.ALERTS, Alert), key=lambda a: a.window_start)
    for a, b in pairwise(alerts):
        if a.key == b.key and b.window_start - a.window_end <= timedelta(minutes=5):
            host, _, service = a.key.partition("/")
            return Anchor(
                kind="logbert",
                at=a.window_start,
                host=host,
                service=service or None,
                why=f"LogBERT flagged {a.key} repeatedly from {a.window_start:%H:%M:%S} (miss ratio {a.miss_ratio:.2f})",
            )
    before = {
        (c.template_id, c.service) for c in rows_of(done, outputs, Discover.BASELINE, TemplateCount)
    }
    new = [
        c
        for c in counts
        if c.level in ("error", "warn")
        and (c.template_id, c.service) not in before
        and (not focus or c.service in focus)
    ]
    if new:
        c = min(new, key=lambda c: c.first)
        tid = ctx.canonical(c.template_id)
        return Anchor(
            kind="new_error",
            at=c.first,
            service=c.service,
            host=c.host,
            template_id=tid,
            message=c.example[: ctx.rca.line_chars],
            why=f"first {c.level} message not seen in the {ctx.rca.baseline_hours:g} h before: {ctx.templates.get(tid, '')[:80]}",
        )
    return None


def silence(
    buckets: list[Bucket], counts: list[TemplateCount], rng: TimeSpan, focus: set[str], ctx: Ctx
) -> Anchor | None:
    """A service that kept up ``silence_min_rate`` lines/minute, then logged nothing for at least
    ``silence_minutes`` minutes up to the end of the range."""
    cfg = ctx.rca
    origin = rng.start - cfg.before  # bucket 0 of the per-minute read
    total = int((rng.end - origin).total_seconds() // 60)
    per: dict[str | int | None, dict[int, int]] = defaultdict(dict)
    for b in buckets:
        per[b.key][b.bucket] = b.n
    last_words: dict[str | None, TemplateCount] = {}
    for c in counts:
        if c.service not in last_words or c.last >= last_words[c.service].last:
            last_words[c.service] = c
    found = []
    for service, by in per.items():
        last = max(by)
        quiet = total - 1 - last  # whole minutes with nothing, up to the end
        before = [by.get(i, 0) for i in range(max(0, last - 29), last)]
        rate = sum(before) / len(before) if before else 0
        if quiet < cfg.silence_minutes or rate < cfg.silence_min_rate:
            continue
        service = None if service is None else str(service)  # the buckets were by service
        words = last_words.get(service)
        loud = words is not None and (
            words.level in ("error", "warn") or ctx.canonical(words.template_id) in ctx.shutdown
        )
        found.append(
            Anchor(
                kind="silence",
                at=words.last if words else origin + timedelta(minutes=last + 1),
                service=service,
                loud=loud,
                why=f"{service} logged {rate:.0f} lines/minute, then nothing for {quiet} minutes"
                + (", its last line an error" if loud else ""),
            )
        )
    # A service that died saying why beats one that went quiet because nothing reached it
    # (a database with no callers); then the one asked about; then the first.
    return min(found, key=lambda a: (not a.loud, a.service not in focus, a.at)) if found else None


def rank(
    spec: QuerySpec, anchor: Anchor, done: Round, outputs: dict, ctx: Ctx
) -> tuple[list[Candidate], list[Evidence], list[str]]:
    cfg = ctx.rca
    counts = rows_of(done, outputs, Around.TEMPLATES, TemplateCount)
    rows = rows_of(done, outputs, Around.ANCHOR_LINES, LogLine)
    window = next(q for q in done.queries if q.tool == "template_counts").args
    length = datetime.fromisoformat(window["end"]) - datetime.fromisoformat(window["start"])
    n_slices = max(1, int(cfg.baseline / max(length, timedelta(minutes=1))))
    base: dict[int, list[int]] = defaultdict(lambda: [0] * n_slices)
    # slices the store has any line in: before its first line there's no baseline
    covered: set[int] = set()
    for b in rows_of(done, outputs, Around.BASELINE, Bucket):
        if 0 <= b.bucket < n_slices:
            base[ctx.canonical(int(b.key or 0))][b.bucket] += b.n
            covered.add(b.bucket)

    notes = []
    surprise = logbert_surprise(rows, anchor, ctx)
    if surprise is None and ctx.use_logbert and anchor.service:
        notes.append(f"no LogBERT model covers {anchor.service}: ranked without it")
    near = (
        dict(ctx.neighbours(anchor.template_id, 5))
        if ctx.use_semantic and anchor.template_id
        else {}
    )
    described = {}
    if ctx.use_semantic and (spec.anchor_event or spec.semantic_query):
        described = {
            t: s
            for t, s in ctx.search_templates(spec.semantic_query or spec.anchor_event or "", 5)
            if s >= 0.3
        }

    cands = []
    for c in counts:
        tid = ctx.canonical(c.template_id)
        history = [base[tid][i] for i in sorted(covered)] if tid in base else [0] * len(covered)
        med = median(history) if history else 0.0
        # a sporadic template: its mean
        expected = med or (sum(history) / len(history) if history else 0.0)
        is_new = bool(covered) and expected == 0
        ratio = c.n / expected if expected else math.inf
        rarity = (
            1.0
            if is_new
            else 0.5
            if not covered  # no baseline at all: neither rare nor common
            else min(1.0, max(0.02, (ratio - 1) / 9))
        )
        parts = {"level": cfg.level_weight(c.level)}
        if tid in ctx.shutdown:
            parts["shutdown template"] = cfg.shutdown_weight
        if surprise and tid in surprise:
            parts["LogBERT did not expect it"] = surprise[tid]
        if tid in near:
            parts["LogBERT neighbour of the anchor"] = cfg.related_weight
        if tid in described:
            parts["matches the question"] = cfg.related_weight
        anomaly = max(parts.values())
        proximity = (
            1.0
            if anchor.kind == "range"
            else math.exp(-abs((c.first - anchor.at).total_seconds()) / cfg.proximity_seconds)
        )
        why = [
            f"anomaly {anomaly:.2f} ({max(parts, key=lambda k: parts[k])})",
            "new: not in the 24 h before"
            if is_new
            else "no baseline: the store has no lines from the 24 h before"
            if not covered
            else f"{c.n} lines vs {expected:.3g} expected per window (x{ratio:.1f})",
            f"first seen {_ago(c.first, anchor.at)}"
            if anchor.kind != "range"
            else f"first seen {c.first:%H:%M:%S}",
        ]
        cands.append(
            Candidate(
                template_id=tid,
                service=c.service,
                host=c.host,
                count=c.n,
                first=c.first,
                last=c.last,
                level=c.level,
                example=c.example,
                log_id=c.log_id,
                new=is_new,
                baseline_median=med,
                anomaly=round(anomaly, 3),
                rarity=round(rarity, 3),
                proximity=round(proximity, 3),
                score=round(anomaly * rarity * proximity, 4),
                why="; ".join(why),
                template=ctx.templates.get(tid, "not parsed yet"),
            )
        )
    cands.sort(key=lambda c: (-round(c.score, 2), c.first))
    # precursors: new lines just before the strongest one
    if cands and cands[0].score >= cfg.strong:
        lead = cands[0]
        for c in cands[1:]:
            gap = (lead.first - c.first).total_seconds()
            if c.new and 0 < gap <= cfg.precursor_seconds and c.anomaly < cfg.error_weight:
                c.anomaly = cfg.error_weight
                c.score = round(c.anomaly * c.rarity * c.proximity, 4)
                c.why += f"; new, {gap:.0f} s before the top line: a likely cause"
        cands.sort(key=lambda c: (-round(c.score, 2), c.first))
    top = cands[: cfg.max_candidates]
    if top:
        chain = [c for c in top if c.score >= 0.5 * top[0].score]
        min(chain, key=lambda c: c.first).earliest_in_chain = True

    evidence: list[Evidence] = []
    if anchor.message:
        evidence = add_lines(
            evidence,
            [
                LogLine(
                    timestamp=anchor.at,
                    service=anchor.service,
                    host=anchor.host,
                    level="error",
                    template_id=anchor.template_id,
                    message=anchor.message,
                )
            ],
            ctx,
            why="the anchor: " + anchor.why,
        )
    for i, c in enumerate(top, 1):
        evidence = add_lines(
            evidence,
            [
                LogLine(
                    log_id=c.log_id,
                    timestamp=c.first,
                    service=c.service,
                    host=c.host,
                    level=c.level,
                    template_id=c.template_id,
                    message=c.example,
                )
            ],
            ctx,
            why=f"#{i} by score {c.score:.3f}: {c.why}",
            score=c.score,
        )
    if rows:  # context: the anchor service's last lines before the anchor
        evidence = add_lines(
            evidence,
            [r for r in rows if r.timestamp <= anchor.at][-10:],
            ctx,
            why=f"{anchor.service} just before the anchor",
        )
    return top, evidence, notes


def _ago(first: datetime, at: datetime) -> str:
    d = (first - at).total_seconds()
    return (
        "at the anchor"
        if abs(d) < 1
        else f"{abs(d) / 60:.1f} min {'before' if d < 0 else 'after'} the anchor"
    )


def logbert_surprise(rows: list[LogLine], anchor: Anchor, ctx: Ctx) -> dict[int, float] | None:
    """Per template, the share of its lines (in the anchor service's sequence) that LogBERT
    marked unseen or unexpected."""
    scorer = ctx.scorer_for(anchor.service) if anchor.service and ctx.use_logbert else None
    if scorer is None or not rows:
        return None
    ids = [ctx.canonical(int(r.template_id or 0)) for r in rows]
    verdicts = [
        v for i in range(0, len(ids), ctx.window) for v in scorer.explain(ids[i : i + ctx.window])
    ]
    hit, total = defaultdict(int), defaultdict(int)
    for tid, v in zip(ids, verdicts):
        total[tid] += 1
        hit[tid] += v["status"] != "expected"
    return {t: hit[t] / total[t] for t in total}


def add_lines(
    evidence: list[Evidence],
    rows: list[LogLine],
    ctx: Ctx,
    why: str,
    score: float | None = None,
    cap: int | None = None,
) -> list[Evidence]:
    """The bundle with ``rows`` added (deduplicated, at most ``cap`` lines, ``max_lines`` by default), in time
    order, refs renumbered so L1 is the earliest."""
    out = list(evidence)
    seen = {(e.timestamp, e.message) for e in out}
    for r in rows:
        if len(out) >= (cap or ctx.rca.max_lines):
            break
        message = r.message[: ctx.rca.line_chars]
        if (r.timestamp, message) in seen:
            continue
        seen.add((r.timestamp, message))
        out.append(
            Evidence(
                ref="",
                timestamp=r.timestamp,
                service=r.service,
                host=r.host,
                level=r.level,
                template_id=ctx.canonical(r.template_id) if r.template_id is not None else None,
                log_id=r.log_id,
                message=message,
                why=why,
                score=score,
            )
        )
    out.sort(key=lambda e: e.timestamp)
    for i, e in enumerate(out, 1):
        e.ref = f"L{i}"
    return out
