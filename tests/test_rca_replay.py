"""Regression harness: saved fault scenarios replayed through the whole Ask path (router, RCA,
synthesizer). Rule-based by default; an LLM's recorded replies replay without the model."""
from __future__ import annotations

import pytest

from logpipe.eval.rca_eval import (
    VARIANTS,
    Recorded,
    build_world,
    evaluate,
    investigate,
    score_report,
    utc_clock,
)

MODES = ["sigterm", "db_port_blocked", "lock_contention", "dns_failure", "api_hang", "control_quiet", "control_busy"]


@pytest.fixture(scope="module")
def world():
    with utc_clock():
        yield build_world(MODES, epochs=1, log=lambda *_: None)


def test_scenarios_keep_their_answers(world):
    with utc_clock():
        result = evaluate(world, investigate, VARIANTS, log=lambda *_: None)
    s = result["summary"]
    assert s["full"]["evidence_recall"] == 1.0 and s["full"]["answer_correct"] == 1.0, result["rows"]
    assert s["no_logbert"]["answer_correct"] >= 0.8 and s["no_semantic"]["answer_correct"] >= 0.8
    by = {r["mode"]: r for r in result["rows"]}
    assert by["api_hang"]["full"]["anchor"] == "silence" and by["control_quiet"]["full"]["anchor"] is None


def test_every_claim_cites_the_bundle(world):
    with utc_clock():
        for slot in world.slots:
            report = investigate(world, slot)
            refs = {e["ref"] for e in report.get("evidence", [])} | {f"T{c['template_id']}" for c in report.get("candidates", [])}
            refs |= {h["ref"] for h in report.get("hints", [])} | {s["ref"] for s in report.get("series", [])}
            for lead in (report.get("answer") or {}).get("leads", []):
                assert set(lead["cites"]) <= refs, (slot.mode, lead)
            assert len(report.get("evidence", [])) <= 40
            for r in report.get("rounds", []):  # nothing ran that wasn't approved (the eval approves all)
                assert all(q["status"] == "done" for q in r["queries"])


class FakeLLM:
    """Stands in for a chat model: routes like the fast path and blames the top-ranked line."""
    can_ask = True

    def __init__(self):
        self.calls = 0

    def ask(self, schema, prompt):
        self.calls += 1
        if schema.__name__ == "RouteOut":
            return schema(intent="rca", anchor_event="checkout failing")
        first = next(line for line in prompt.splitlines() if line.startswith("T"))
        tid = first.split()[0]
        return schema(summary="from the top template", hypotheses=[{"cause": "top", "confidence": "low", "reasoning": "",
                                                                     "cites": [tid]}])


def test_record_then_replay_without_the_model(world, tmp_path):
    path = tmp_path / "replies.json"
    slot = next(s for s in world.slots if s.mode == "db_port_blocked")
    fake = FakeLLM()
    with utc_clock():
        recorded = investigate(world, slot, adapter=Recorded(path, fake))
        calls = fake.calls
        replayed = investigate(world, slot, adapter=Recorded(path))
    assert calls >= 1 and path.exists()
    assert replayed["answer"]["hypotheses"] == recorded["answer"]["hypotheses"] and recorded["answer"]["by"] == "llm"
    assert score_report(replayed, slot)["correct"]
    other = next(s for s in world.slots if s.mode == "sigterm")
    with utc_clock(), pytest.raises(KeyError, match="--record"):
        investigate(world, other, adapter=Recorded(path))  # an unseen prompt fails loudly
