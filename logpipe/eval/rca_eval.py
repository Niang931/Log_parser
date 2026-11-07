"""Root-cause eval on injected faults (``python -m logpipe.eval.rca_eval``).

One ``logpipe server`` (memory store) learns a web shop's normal traffic, then each fault from
:data:`~logpipe.showcase.generator.FAULTS` is injected into its own time slot, after an hour of
normal traffic. For each slot an on-call question is asked and two things are measured apart:

* **evidence recall**: is the root-cause line (``truth``) in the evidence bundle at all?
* **answer correctness**: does the top hypothesis cite it (or, on a no-fault control, does the
  answer say there isn't enough evidence)?

Ablations: without LogBERT, without semantic search, and a naive baseline that hands the last
200 lines before the question to the model. LLM replies are recorded once and replayed
(``--record`` / ``--replay``), so the regression test (``tests/test_rca_replay.py``) needs no model.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..showcase.generator import FAULTS, FILES, ShopGenerator

ORDER_EVERY = timedelta(seconds=20)
NORMAL_BEFORE = timedelta(minutes=60)  # normal traffic before each fault
ASK_AFTER = timedelta(minutes=5)  # the question comes this long after the fault starts, while it lasts
TRAIN_FOR = timedelta(hours=2)
NAIVE_LINES = 200

# The question the on-call engineer asks, per fault; {at} is the fault's start (HH:MM UTC).
QUESTIONS = {
    "sigterm": "why did checkout start failing around {at}?",
    "oom": "why is checkout returning 502s?",
    "segfault": "what caused the checkout errors around {at}?",
    "disk_full": "why did the database go down?",
    "db_port_blocked": "why are orders failing in the last 10 minutes?",
    "pool_exhausted": "why did shop-api fail around {at}?",
    "gateway_timeout": "why are checkouts failing?",
    "dns_failure": "why did payments break around {at}?",
    "tls_expired": "why are orders failing in the last 15 minutes?",
    "config_reload": "why did the shop go down around {at}?",
    "dependency_503": "what caused the failed orders in the last 10 minutes?",
    "gc_thrash": "why is shop-api slow around {at}?",
    "lock_contention": "why are database statements timing out?",
    "clock_skew": "why did logins break around {at}?",
    "api_hang": "why did shop-api go down?",
    "control_quiet": "why did shop-api go down in the last 10 minutes?",
    "control_busy": "what caused the outage around {at}?",
}


@dataclass
class Slot:
    mode: str
    start: datetime  # the fault's start
    question: str
    truth: str | None
    now: datetime  # when the question is asked


@dataclass
class World:
    server: object
    client: object
    slots: list[Slot] = field(default_factory=list)
    seconds: float = 0.0


@contextmanager
def utc_clock():
    """The generator writes local wall-clock times into some lines; pin them to UTC so the
    timestamps the server reads match the ones the eval knows."""
    old = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time.tzset()
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


def _send(server, gen: ShopGenerator, events, when: datetime) -> None:
    from ..sources.file_replay import LogEvent

    for e in events:
        server.process(LogEvent(device_id="db-1" if e.service == "postgres" else "web-1", source=FILES[e.service][1],
                                line_no=0, line=e.message, service=e.service, time=when))


def _traffic(server, gen: ShopGenerator, start: datetime, end: datetime) -> None:
    t = start
    while t < end:
        _send(server, gen, gen.order(t), t)
        t += ORDER_EVERY


def build_world(modes: list[str] | None = None, data_dir: Path | None = None, seed: int = 7,
                adapter_factory=None, epochs: int = 4, log=print) -> World:
    """Train on normal traffic, then inject each fault in its own slot ending just before now."""
    import tempfile

    from fastapi.testclient import TestClient

    from ..enrich.llm import get_adapter
    from ..server.app import create_app
    from ..server.control import ControlStore
    from ..server.core import Server
    from ..storage.memory import MemoryStore

    started = time.perf_counter()
    modes = modes or list(FAULTS)
    data_dir = data_dir or Path(tempfile.mkdtemp(prefix="rca-eval-"))
    server = Server(MemoryStore(), ControlStore(), data_dir=data_dir,
                    adapter_factory=adapter_factory or (lambda: get_adapter("offline")), epochs=epochs,
                    start_threads=False, min_train_steps=0)
    client = TestClient(create_app(server))
    gen = ShopGenerator(data_dir / "unused", seed=seed)
    slot_len = NORMAL_BEFORE + ASK_AFTER
    end = (datetime.now(UTC) - timedelta(minutes=1)).replace(microsecond=0)
    t0 = end - slot_len * len(modes) - TRAIN_FOR

    log(f"normal traffic for {TRAIN_FOR}, then calibration and training")
    _traffic(server, gen, t0, t0 + TRAIN_FOR)
    _calibrate(server, client)
    world = World(server, client)
    for i, mode in enumerate(modes):
        slot = t0 + TRAIN_FOR + slot_len * i
        fault = slot + NORMAL_BEFORE
        _traffic(server, gen, slot, fault)
        t, step = fault, 0
        while t < fault + ASK_AFTER:
            _send(server, gen, gen.fault(mode, t, step), t)
            t, step = t + ORDER_EVERY, step + 1
        server.enrich()  # mint templates for the new lines, as the learner does every 30 s
        world.slots.append(Slot(mode, fault, QUESTIONS[mode].format(at=f"{fault:%H:%M}"), FAULTS[mode]["truth"],
                                fault + ASK_AFTER))
        log(f"  injected {mode} at {fault:%H:%M}")
    for _ in range(3):
        server.enrich()
    world.seconds = time.perf_counter() - started
    return world


def _calibrate(server, client) -> None:
    def run_jobs():
        while not server.jobs.empty():
            job = server.run_job(server.jobs.get_nowait())
            if job["status"] != "done":
                raise RuntimeError(f"{job['kind']} job failed: {job.get('error')}")

    client.post("/api/review/generate", json={"sample_cap": 400}).raise_for_status()
    run_jobs()
    bid = client.get("/api/setup").json()["review"]["id"]
    client.post(f"/api/review/{bid}/decide-all", json={"status": "approved"}).raise_for_status()
    client.post(f"/api/review/{bid}/apply").raise_for_status()
    run_jobs()


# --- metrics ------------------------------------------------------------------------------
def naive_bundle(world: World, slot: Slot) -> list[str]:
    """The baseline: the last 200 lines before the question, whatever they are."""
    rows = world.server.reads.between(slot.now - timedelta(hours=2), slot.now, 100_000)
    return [r["message"] for r in rows[-NAIVE_LINES:]]


def contains_truth(lines: list[str], truth: str | None) -> bool:
    return truth is not None and not truth.startswith("silence:") and any(truth in line for line in lines)


def score_report(report: dict, slot: Slot) -> dict:
    """Evidence recall and correctness for one Ask report (see logpipe.ask.engine)."""
    lines = {e["ref"]: e for e in report.get("evidence", [])}
    texts = [e["message"] for e in lines.values()]
    silence = slot.truth and slot.truth.startswith("silence:")
    anchor = report.get("anchor") or {}
    if slot.truth is None:
        recall = None
    elif silence:
        recall = anchor.get("kind") == "silence" and anchor.get("service") == slot.truth.split(":", 1)[1]
    else:
        recall = contains_truth(texts, slot.truth)
    answer = report.get("answer") or {}
    hyps = answer.get("hypotheses") or []
    if slot.truth is None:
        correct = bool(answer.get("insufficient_evidence")) or report.get("status") == "clarify"
    elif not hyps:
        correct = False
    else:
        top = hyps[0]
        cited = [lines[c]["message"] for c in top.get("cites", []) if c in lines]
        cited += [e["message"] for e in lines.values() if f"T{e.get('template_id')}" in top.get("cites", [])]
        correct = (anchor.get("kind") == "silence" and slot.truth.split(":", 1)[1] in json.dumps(top)) if silence \
            else contains_truth(cited, slot.truth)
    return {"recall": recall, "correct": correct}


def evaluate(world: World, investigate, variants: dict[str, dict], log=print) -> dict:
    """``investigate(world, slot, **variant) -> report`` per slot and variant, plus the naive baseline."""
    rows = []
    for slot in world.slots:
        row = {"mode": slot.mode, "question": slot.question,
               "naive": {"recall": None if slot.truth is None or slot.truth.startswith("silence:")
                         else contains_truth(naive_bundle(world, slot), slot.truth)}}
        for name, kw in variants.items():
            report = investigate(world, slot, **kw)
            row[name] = score_report(report, slot) | {"lines": len(report.get("evidence", [])),
                                                      "anchor": (report.get("anchor") or {}).get("kind")}
        rows.append(row)
        log(f"  {slot.mode:16} " + "  ".join(f"{k}={v}" for k, v in row.items() if isinstance(v, dict)))
    return {"rows": rows, "summary": summarize(rows, ["naive", *variants])}


def summarize(rows: list[dict], names: list[str]) -> dict:
    out = {}
    for name in names:
        rec = [r[name]["recall"] for r in rows if r[name].get("recall") is not None]
        cor = [r[name]["correct"] for r in rows if r[name].get("correct") is not None]
        out[name] = {"evidence_recall": round(sum(rec) / len(rec), 3) if rec else None,
                     "answer_correct": round(sum(cor) / len(cor), 3) if cor else None, "n": len(rows)}
    return out


def table(summary: dict) -> str:
    lines = ["| variant | evidence recall | answer correct |", "|---|---|---|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['evidence_recall'] if s['evidence_recall'] is not None else '-'} | "
                     f"{s['answer_correct'] if s['answer_correct'] is not None else '-'} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modes", nargs="*", help="faults to inject (default: all)")
    ap.add_argument("--out", type=Path, default=Path("artifacts/eval/rca.json"))
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args(argv)
    with utc_clock():
        world = build_world(args.modes, epochs=args.epochs)
        print(f"world built in {world.seconds:.0f} s")
        result = evaluate(world, _investigate_or_none(), VARIANTS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(table(result["summary"]))


VARIANTS: dict[str, dict] = {}


def _investigate_or_none():
    return lambda world, slot, **kw: {}


if __name__ == "__main__":
    main()
