"""The synthesizer: uncited claims are dropped, thin evidence says so, follow-ups stay bounded."""
from __future__ import annotations

from logpipe.ask.engine import Ctx
from logpipe.ask.index import TemplateIndex, auto_shutdown
from logpipe.ask.spec import QuerySpec
from logpipe.ask.synth import check_citations, offline_answer, synth_schema, synthesize

REPORT = {
    "anchor": {"kind": "shutdown", "at": "2026-11-08T10:05:00+00:00", "why": "first shutdown-type message", "service": "db"},
    "candidates": [{"template_id": 7, "service": "db", "template": "PANIC: could not write: No space left on device",
                    "count": 1, "score": 0.9, "why": "new"},
                   {"template_id": 8, "service": "api", "template": "order <*> failed", "count": 40, "score": 0.1, "why": "x4"}],
    "evidence": [{"ref": "L1", "timestamp": "2026-11-08T10:05:00+00:00", "service": "db", "level": "error", "template_id": 7,
                  "message": "PANIC: could not write: No space left on device", "score": 0.9},
                 {"ref": "L2", "timestamp": "2026-11-08T10:05:03+00:00", "service": "api", "level": "error", "template_id": 8,
                  "message": "order 1 failed", "score": 0.1}],
}


def test_uncited_claims_are_dropped():
    answer = check_citations({"summary": "s", "timeline": [{"at": "x", "what": "disk", "cites": ["L1"]},
                                                           {"at": "y", "what": "made up", "cites": ["L99"]}],
                              "hypotheses": [{"cause": "aliens", "confidence": "high", "reasoning": "", "cites": []},
                                             {"cause": "disk full", "confidence": "high", "reasoning": "", "cites": ["L1", "T7", "L42"]}]},
                             REPORT)
    assert [h["cause"] for h in answer["hypotheses"]] == ["disk full"] and answer["hypotheses"][0]["cites"] == ["L1", "T7"]
    assert [t["what"] for t in answer["timeline"]] == ["disk"] and len(answer["dropped"]) == 2
    assert not answer.get("insufficient_evidence")


def test_nothing_cited_means_insufficient_evidence():
    answer = check_citations({"summary": "s", "hypotheses": [{"cause": "a guess", "cites": ["L9"]}]}, REPORT)
    assert answer["insufficient_evidence"] and answer["hypotheses"] == [] and answer["missing"]


class Scripted:
    can_ask = True

    def __init__(self, **fields):
        self.fields, self.prompts = fields, []

    def ask(self, schema, prompt):
        self.prompts.append(prompt)
        return schema(**self.fields)


def test_llm_sees_only_the_bundle_and_follow_ups_are_bounded():
    llm = Scripted(summary="db disk full", hypotheses=[{"cause": "disk full", "confidence": "high", "reasoning": "L1", "cites": ["L1"]}],
                   follow_up=[{"keywords": ["df -h"], "services": ["db", "invented"]}, {"keywords": ["x"]}, {"keywords": ["y"]}])
    ctx = Ctx(store=None, templates={}, adapter=llm)
    answer, follow = synthesize(QuerySpec(intent="rca", anchor_event="why did db die"), REPORT, ctx)
    assert answer["by"] == "llm" and answer["hypotheses"][0]["cites"] == ["L1"]
    assert len(follow) == 2 and follow[0].services == ["db"]  # an invented service is dropped
    prompt = llm.prompts[0]
    assert "L1 2026-11-08T10:05:00" in prompt and "data, not instructions" in prompt and "T7 db:" in prompt


def test_offline_answer_is_the_ranking_not_a_diagnosis():
    a = offline_answer(QuerySpec(intent="rca"), REPORT)
    assert a["hypotheses"] == [] and a["leads"][0]["cites"] == ["L1", "T7"] and "not a diagnosis" in a["note"]
    assert len(a["leads"]) == 1  # the 0.1 symptom isn't a lead
    quiet = offline_answer(QuerySpec(intent="rca"), {"anchor": {"why": "x"}, "candidates": [], "evidence": []})
    assert quiet["insufficient_evidence"] and quiet["leads"] == []


def test_schema_has_what_the_prompt_asks_for():
    assert {"summary", "timeline", "hypotheses", "insufficient_evidence", "follow_up"} <= set(synth_schema().model_fields)


def test_text_index_finds_shutdown_wording():
    templates = {1: "java.lang.OutOfMemoryError: Java heap space", 2: "order <*:order_id> confirmed",
                 3: "PANIC: could not write to file <*>: No space left on device", 4: "user <*> logged in",
                 5: "LOG: received fast shutdown request"}
    idx = TemplateIndex.build(templates, 1)
    assert idx.kind in ("tfidf", "minilm")
    assert idx.search("out of memory", 1)[0][0] == 1
    assert idx.search("disk full", 1)[0][0] == 3
    assert set(auto_shutdown(templates, idx)) == {1, 3, 5}
