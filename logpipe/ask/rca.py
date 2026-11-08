"""Root-cause analysis without the LLM: find when things broke (the anchor), read around it,
and rank what changed. Every number that decides a line's place is kept next to it, so the
report can say why each line is there.

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
Ties go to the earlier template: a cause comes before its effects.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from statistics import median

from .engine import Ctx, Outcome, Round, local, log_read, span
from .spec import QuerySpec, time_range

BEFORE, AFTER = timedelta(minutes=30), timedelta(minutes=2)
WIDE_BEFORE, WIDE_AFTER = timedelta(hours=2), timedelta(minutes=5)
BASELINE = timedelta(hours=24)
SILENCE_MIN_RATE = 2.0  # lines/minute a service must have kept up before going quiet
SILENCE_MINUTES = 3
LEVEL_WEIGHT = {"error": 0.8, "warn": 0.5}
MAX_CANDIDATES = 15
STRONG = 0.15  # a candidate that stands out: a new warning near the anchor scores 0.5, a new info line 0.1
MAX_LINES = 40
LINE_CHARS = 300
FOLLOW_UPS = 3


def _t(s: str) -> datetime:
    t = datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def _iso(t: datetime) -> str:
    return t.astimezone(UTC).isoformat()


# --- rounds ---------------------------------------------------------------------------------
def rca_first_round(spec: QuerySpec, ctx: Ctx) -> Outcome:
    if spec.anchor_time:
        anchor = {"kind": "user", "at": _iso(spec.anchor_time), "service": spec.services[0] if spec.services else None,
                  "why": "the time given in the question"}
        return Outcome(evidence_round(2, anchor, ctx, spec), state={"anchor": anchor, "phase": "evidence"})
    start, end = time_range(spec, ctx.now)
    queries = [
        log_read(ctx, 1, 1, "buckets", "Lines per service per minute",
                 "to see whether a service that was logging went quiet", filters={},
                 step=60, by="service", **span(start - BEFORE, end)),
        log_read(ctx, 1, 2, "template_counts", "Templates in the range",
                 "first time each template appeared: shutdown messages, new errors", filters={}, **span(start, end)),
        log_read(ctx, 1, 3, "template_counts", "Templates in the 24 h before",
                 "which templates are normal, so new ones stand out", filters={}, **span(start - BASELINE, start)),
        local(1, 4, "agents", "Agent heartbeats", "an agent that stopped reporting marks when a host went quiet",
              "server.agents(): hostname, last_seen, online (control store)"),
    ]
    if ctx.use_logbert:
        queries.append(local(1, 5, "alerts", "LogBERT alerts", "windows the anomaly model flagged in the range",
                             "reads.alerts(500) filtered to the range (anomaly_alerts)", **span(start, end)))
    return Outcome(Round(1, "find when it broke", queries), state={"phase": "discover", "range": span(start, end)})


def summary_round(spec: QuerySpec, ctx: Ctx, start: datetime, end: datetime) -> Outcome:
    anchor = {"kind": "range", "at": _iso(end), "why": "a summary covers the whole range"}
    return Outcome(evidence_round(1, anchor, ctx, spec, window=(start, end)), state={"anchor": anchor, "phase": "evidence"})


def evidence_round(n: int, anchor: dict, ctx: Ctx, spec: QuerySpec, wide: bool = False,
                   window: tuple[datetime, datetime] | None = None) -> Round:
    at = _t(anchor["at"])
    start, end = window or (at - (WIDE_BEFORE if wide else BEFORE), min(at + (WIDE_AFTER if wide else AFTER), ctx.now))
    length = max(int((end - start).total_seconds()), 60)
    f = {"hosts": list(spec.hosts)} if spec.hosts and not spec.services else {}
    queries = [
        log_read(ctx, n, 1, "template_counts", "Templates around the anchor", "what was logged, and when each kind first appeared",
                 filters=f, **span(start, end)),
        log_read(ctx, n, 2, "buckets", "The same window, over the 24 h before",
                 f"counts per template in {length // 60}-minute slices of the previous day: the baseline",
                 filters=f, step=length, **span(start - BASELINE, start)),
    ]
    service = anchor.get("service")
    if service and anchor["kind"] != "range":
        lb = ctx.use_logbert and ctx.scorer_for(service) is not None
        queries.append(log_read(ctx, n, 3, "rows", f"{service} lines just before the anchor",
                                "its last lines before the anchor" + (", which LogBERT re-reads to mark lines it did not expect" if lb else ""),
                                filters={"services": [service]}, limit=2000, **span(at - timedelta(minutes=5), min(at + timedelta(minutes=1), ctx.now))))
    if ctx.use_logbert:
        queries.append(local(n, 4, "alerts", "LogBERT alerts in the window", "windows the anomaly model flagged",
                             "reads.alerts(500) filtered to the window (anomaly_alerts)", **span(start, end)))
    return Round(n, "read around " + ("the range" if anchor["kind"] == "range" else "the anchor") + (" (wider)" if wide else ""), queries)


def rca_advance(spec: QuerySpec, state: dict, done: Round, outputs: dict, ctx: Ctx) -> Outcome:
    phase = state.get("phase")
    out = {q.tool + str(i): outputs.get(q.id) for i, q in enumerate(done.queries, 1)}
    if phase == "discover":
        anchor = find_anchor(spec, out, ctx, _t(state["range"]["start"]), _t(state["range"]["end"]))
        if anchor is None:
            return Outcome(result={"kind": "clarify", "anchor": None, "message": (
                "I couldn't tell when it broke: no service went quiet, no shutdown or new error message, no "
                "anomaly alerts in that range. When did it happen (e.g. 'around 14:05'), and which service?")},
                state={"phase": "clarify"})
        return Outcome(evidence_round(done.n + 1, anchor, ctx, spec), state={"anchor": anchor, "phase": "evidence"})
    if phase in ("evidence", "wide"):
        anchor = state["anchor"]
        cands, lines, notes = rank(spec, anchor, done, outputs, ctx)
        strong = [c for c in cands if c["score"] >= STRONG]
        if phase == "evidence" and len(strong) < 3 and anchor["kind"] != "range":
            return Outcome(evidence_round(done.n + 1, anchor, ctx, spec, wide=True),
                           state={"phase": "wide", "first_pass": {"candidates": len(strong)}})
        report = {"kind": "report", "intent": spec.intent, "anchor": anchor, "candidates": cands, "evidence": lines,
                  "notes": notes, "window": state.get("window")}
        return synthesize_step(spec, report, ctx, state, done.n)
    if phase == "follow_up":
        report = state["report"]
        extra = [r for q in done.queries if q.id in outputs for r in outputs[q.id]]
        add_lines(report, extra, ctx, why="asked for by the follow-up search")
        return synthesize_step(spec, report, ctx, state, done.n)
    raise ValueError(f"unknown phase {phase!r}")


def synthesize_step(spec: QuerySpec, report: dict, ctx: Ctx, state: dict, last_round: int) -> Outcome:
    from .synth import synthesize

    answer, follow = synthesize(spec, report, ctx)
    report["answer"] = answer
    used = state.get("follow_ups", 0)
    if follow and used < FOLLOW_UPS:
        at = _t(report["anchor"]["at"])
        n = last_round + 1
        queries = [log_read(ctx, n, i, "rows", f"Follow-up: {f.describe()}", "the model asked to look for this",
                            filters=f.__dict__, limit=50, **span(at - WIDE_BEFORE, min(at + WIDE_AFTER, ctx.now)))
                   for i, f in enumerate(follow[:2], 1)]
        return Outcome(Round(n, f"follow-up {used + 1} of {FOLLOW_UPS}", queries),
                       state={"phase": "follow_up", "report": report, "follow_ups": used + 1})
    return Outcome(result=report, state={"phase": "done"})


# --- anchor ---------------------------------------------------------------------------------
def find_anchor(spec: QuerySpec, out: dict, ctx: Ctx, start: datetime, end: datetime) -> dict | None:
    focus = set(spec.services)
    counts = out.get("template_counts2") or []
    found = silence(out.get("buckets1") or [], counts, start, end, focus, ctx)
    if found:
        return found
    for a in out.get("agents4") or []:
        seen = a.get("last_seen")
        if seen and not a.get("online") and start.isoformat() <= seen <= end.isoformat():
            return {"kind": "silence", "at": seen, "host": a["host"], "service": None,
                    "why": f"agent {a['host']} stopped sending heartbeats at {seen}"}
    shutdown = [c for c in counts if ctx.canonical(int(c["template_id"])) in ctx.shutdown and (not focus or c["service"] in focus)]
    shutdown = shutdown or [c for c in counts if ctx.canonical(int(c["template_id"])) in ctx.shutdown]
    if shutdown:
        c = min(shutdown, key=lambda c: c["first"])
        tid = ctx.canonical(int(c["template_id"]))
        return {"kind": "shutdown", "at": c["first"], "service": c["service"], "host": c["host"], "template_id": tid,
                "message": c["example"][:LINE_CHARS], "why": f"first shutdown-type message: {ctx.templates.get(tid, '')[:80]}"}
    alerts = sorted(out.get("alerts5") or [], key=lambda a: a["window_start"])
    for a, b in pairwise(alerts):
        if a["key"] == b["key"] and _t(b["window_start"]) - _t(a["window_end"]) <= timedelta(minutes=5):
            host, _, service = a["key"].partition("/")
            return {"kind": "logbert", "at": a["window_start"], "host": host, "service": service or None,
                    "why": f"LogBERT flagged {a['key']} repeatedly from {a['window_start']} (miss ratio {a['miss_ratio']:.2f})"}
    before = {(int(c["template_id"]), c["service"]) for c in out.get("template_counts3") or []}
    new = [c for c in counts if c.get("level") in ("error", "warn") and (int(c["template_id"]), c["service"]) not in before
           and (not focus or c["service"] in focus)]
    if new:
        c = min(new, key=lambda c: c["first"])
        tid = ctx.canonical(int(c["template_id"]))
        return {"kind": "new_error", "at": c["first"], "service": c["service"], "host": c["host"], "template_id": tid,
                "message": c["example"][:LINE_CHARS],
                "why": f"first {c['level']} message not seen in the 24 h before: {ctx.templates.get(tid, '')[:80]}"}
    return None


def silence(buckets: list[dict], counts: list[dict], start: datetime, end: datetime, focus: set[str], ctx: Ctx) -> dict | None:
    """A service that kept up SILENCE_MIN_RATE lines/minute, then logged nothing for at least
    SILENCE_MINUTES minutes up to the end of the range."""
    origin = start - BEFORE  # bucket 0 of the per-minute query
    total = int((end - origin).total_seconds() // 60)
    per: dict[str, dict[int, int]] = defaultdict(dict)
    for b in buckets:
        per[b["key"]][int(b["bucket"])] = int(b["n"])
    last_seen = {}
    for c in counts:
        last_seen[c["service"]] = max(last_seen.get(c["service"], ""), c["last"])
    found = []
    for service, by in per.items():
        last = max(by)
        quiet = total - 1 - last  # whole minutes with nothing, up to the end
        before = [by.get(i, 0) for i in range(max(0, last - 29), last)]
        if quiet < SILENCE_MINUTES or not before or sum(before) / len(before) < SILENCE_MIN_RATE:
            continue
        at = last_seen.get(service) or _iso(origin + timedelta(minutes=last + 1))
        found.append({"kind": "silence", "at": at, "service": service, "host": None,
                      "why": f"{service} logged {sum(before) / len(before):.0f} lines/minute, then nothing for {quiet} minutes"})
    if focus and (mine := [f for f in found if f["service"] in focus]):
        found = mine
    return min(found, key=lambda f: f["at"]) if found else None


# --- ranking --------------------------------------------------------------------------------
def rank(spec: QuerySpec, anchor: dict, done: Round, outputs: dict, ctx: Ctx) -> tuple[list[dict], list[dict], list[str]]:
    by_tool = {q.tool + str(i): outputs.get(q.id) for i, q in enumerate(done.queries, 1)}
    counts = by_tool.get("template_counts1") or []
    buckets = by_tool.get("buckets2") or []
    rows = by_tool.get("rows3") or []
    notes = []
    at = _t(anchor["at"])
    window = next(q for q in done.queries if q.tool == "template_counts").args
    length = _t(window["end"]) - _t(window["start"])
    n_buckets = max(1, int(BASELINE / max(length, timedelta(minutes=1))))
    base: dict[int, list[int]] = defaultdict(lambda: [0] * n_buckets)
    for b in buckets:
        if 0 <= int(b["bucket"]) < n_buckets:
            base[ctx.canonical(int(b["key"]))][int(b["bucket"])] += int(b["n"])

    surprise = logbert_surprise(rows, anchor, ctx)
    if surprise is None and ctx.use_logbert and anchor.get("service"):
        notes.append(f"no LogBERT model covers {anchor['service']}: ranked without it")
    near = dict(ctx.neighbours(anchor["template_id"], 5)) if ctx.use_semantic and anchor.get("template_id") else {}
    described = {}
    if ctx.use_semantic and (spec.anchor_event or spec.semantic_query):
        described = {t: s for t, s in ctx.search_templates(spec.semantic_query or spec.anchor_event or "", 5) if s >= 0.3}

    cands = []
    for c in counts:
        tid = ctx.canonical(int(c["template_id"]))
        history = base.get(tid)
        med = median(history) if history else 0.0
        is_new = not history or sum(history) == 0
        n = int(c["n"])
        ratio = n / med if med else math.inf
        rarity = 1.0 if is_new else min(1.0, max(0.02, (ratio - 1) / 9))
        level_w = LEVEL_WEIGHT.get(c.get("level") or "", 0.1)
        lb = (surprise or {}).get(tid)
        parts = {"level": level_w}
        if tid in ctx.shutdown:
            parts["shutdown template"] = 0.9
        if lb is not None:
            parts["LogBERT did not expect it"] = lb
        if tid in near:
            parts["LogBERT neighbour of the anchor"] = 0.5
        if tid in described:
            parts["matches the question"] = 0.5
        anomaly = max(parts.values())
        first = _t(c["first"])
        proximity = 1.0 if anchor["kind"] == "range" else math.exp(-abs((first - at).total_seconds()) / 600)
        score = anomaly * rarity * proximity
        why = [f"anomaly {anomaly:.2f} ({max(parts, key=lambda k: parts[k])})",
               "new: not in the 24 h before" if is_new else f"{n} lines vs baseline median {med:g} per window (x{ratio:.1f})",
               f"first seen {_ago(first, at)}" if anchor["kind"] != "range" else f"first seen {c['first'][11:19]}"]
        cands.append({"template_id": tid, "service": c["service"], "host": c["host"], "count": n, "first": c["first"],
                      "last": c["last"], "level": c.get("level"), "example": c["example"], "log_id": c["log_id"],
                      "new": is_new, "baseline_median": med, "anomaly": round(anomaly, 3), "rarity": round(rarity, 3),
                      "proximity": round(proximity, 3), "score": round(score, 4), "why": "; ".join(why),
                      "template": ctx.templates.get(tid, "not parsed yet")})
    cands.sort(key=lambda c: (-round(c["score"], 2), c["first"]))
    top = cands[:MAX_CANDIDATES]
    if top:
        best = top[0]["score"]
        chain = [c for c in top if c["score"] >= 0.5 * best]
        if chain:
            min(chain, key=lambda c: c["first"])["earliest_in_chain"] = True
    report = {"evidence": []}
    if anchor.get("message"):
        add_lines(report, [{"log_id": None, "timestamp": anchor["at"], "service": anchor.get("service"), "host": anchor.get("host"),
                            "level": "error", "template_id": anchor.get("template_id"), "message": anchor["message"]}],
                  ctx, why="the anchor: " + anchor["why"])
    for c in top:
        add_lines(report, [{"log_id": c["log_id"], "timestamp": c["first"], "service": c["service"], "host": c["host"],
                            "level": c["level"], "template_id": c["template_id"], "message": c["example"]}],
                  ctx, why=f"#{top.index(c) + 1} by score {c['score']:.3f}: {c['why']}", score=c["score"])
    if rows:  # context: the anchor service's last lines before the anchor, LogBERT's surprises first
        before = [r for r in rows if _t(r["timestamp"]) <= at][-10:]
        add_lines(report, before, ctx, why=f"{anchor.get('service')} just before the anchor")
    return top, report["evidence"], notes


def _ago(first: datetime, at: datetime) -> str:
    d = (first - at).total_seconds()
    return "at the anchor" if abs(d) < 1 else f"{abs(d) / 60:.1f} min {'before' if d < 0 else 'after'} the anchor"


def logbert_surprise(rows: list[dict], anchor: dict, ctx: Ctx) -> dict[int, float] | None:
    """Per template, the share of its lines (in the anchor service's sequence) that LogBERT
    marked unseen or unexpected."""
    scorer = ctx.scorer_for(anchor["service"]) if anchor.get("service") and ctx.use_logbert else None
    if scorer is None or not rows:
        return None
    ids = [ctx.canonical(int(r["template_id"])) for r in rows]
    verdicts = [v for i in range(0, len(ids), ctx.window) for v in scorer.explain(ids[i:i + ctx.window])]
    hit, total = defaultdict(int), defaultdict(int)
    for tid, v in zip(ids, verdicts):
        total[tid] += 1
        hit[tid] += v["status"] != "expected"
    return {t: hit[t] / total[t] for t in total}


def add_lines(report: dict, rows: list[dict], ctx: Ctx, why: str, score: float | None = None) -> None:
    """Append lines to the bundle as L<n> (deduplicated, capped, chronological order kept)."""
    ev = report.setdefault("evidence", [])
    seen = {(e["timestamp"], e["message"]) for e in ev}
    for r in rows:
        if len(ev) >= MAX_LINES:
            break
        key = (r["timestamp"], r["message"][:LINE_CHARS])
        if key in seen:
            continue
        seen.add(key)
        tid = ctx.canonical(int(r["template_id"])) if r.get("template_id") is not None else None
        ev.append({"ref": f"L{len(ev) + 1}", "timestamp": r["timestamp"], "service": r.get("service"), "host": r.get("host"),
                   "level": r.get("level"), "template_id": tid, "log_id": r.get("log_id"),
                   "message": r["message"][:LINE_CHARS], "why": why, "score": score})
    ev.sort(key=lambda e: e["timestamp"])
    for i, e in enumerate(ev, 1):  # refs follow time order, so L1 is the earliest
        e["ref"] = f"L{i}"
