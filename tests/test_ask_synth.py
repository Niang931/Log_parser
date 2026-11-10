"""The synthesizer: uncited claims are dropped, thin evidence says so, follow-ups stay bounded."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from logpipe.ask.engine import Ctx
from logpipe.ask.index import TemplateIndex, auto_shutdown
from logpipe.ask.models import Anchor, Candidate, Evidence, Report
from logpipe.ask.spec import QuerySpec
from logpipe.ask.synth import check_citations, offline_answer, synth_schema, synthesize

AT = datetime(2026, 11, 8, 10, 5, tzinfo=UTC)


def cand(tid: int, service: str, template: str, count: int, score: float, **kw) -> Candidate:
    return Candidate(template_id=tid, service=service, count=count, first=AT, last=AT, example=template, template=template,
                     new=True, baseline_median=0, anomaly=score, rarity=1, proximity=1, score=score, why="new", **kw)


REPORT = Report(
    intent="rca",
    anchor=Anchor(kind="shutdown", at=AT, why="first shutdown-type message", service="db"),
    candidates=[cand(7, "db", "PANIC: could not write: No space left on device", 1, 0.9),
                cand(8, "api", "order <*> failed", 40, 0.1)],
    evidence=[Evidence(ref="L1", timestamp=AT, service="db", level="error", template_id=7,
                       message="PANIC: could not write: No space left on device", why="#1", score=0.9),
              Evidence(ref="L2", timestamp=AT + timedelta(seconds=3), service="api", level="error", template_id=8,
                       message="order 1 failed", why="#2", score=0.1)],
    groups={"shop": ["api", "db"]},
)


def test_uncited_claims_are_dropped():
    answer = check_citations({"summary": "s", "timeline": [{"at": "x", "what": "disk", "cites": ["L1"]},
                                                           {"at": "y", "what": "made up", "cites": ["L99"]}],
                              "hypotheses": [{"cause": "aliens", "confidence": "high", "reasoning": "", "cites": []},
                                             {"cause": "disk full", "confidence": "high", "reasoning": "", "cites": ["L1", "T7", "L42"]}]},
                             REPORT)
    assert [h.cause for h in answer.hypotheses] == ["disk full"] and answer.hypotheses[0].cites == ["L1", "T7"]
    assert [t.what for t in answer.timeline] == ["disk"] and len(answer.dropped) == 2
    assert not answer.insufficient_evidence and answer.by == "llm"


def test_nothing_cited_means_insufficient_evidence():
    answer = check_citations({"summary": "s", "hypotheses": [{"cause": "a guess", "cites": ["L9"]}]}, REPORT)
    assert answer.insufficient_evidence and answer.hypotheses == [] and answer.missing


class Scripted:
    can_ask = True

    def __init__(self, **fields):
        self.fields, self.prompts = fields, []

    def ask(self, schema, prompt):
        self.prompts.append(prompt)
        return schema(**self.fields)


def test_llm_sees_only_the_bundle_with_groups_and_follow_ups_are_bounded():
    llm = Scripted(summary="db disk full", hypotheses=[{"cause": "disk full", "confidence": "high", "reasoning": "L1", "cites": ["L1"]}],
                   follow_up=[{"kind": "lines", "keywords": ["df -h"], "services": ["db", "invented"]}, {"kind": "lines", "keywords": ["x"]},
                              {"kind": "lines", "keywords": ["y"]}, {"kind": "lines", "keywords": ["z"]}])
    ctx = Ctx(store=None, templates={}, adapter=llm, question="why did the shop's database die?")  # type: ignore[arg-type]
    answer, follow = synthesize(QuerySpec(intent="rca", anchor_event="why did db die"), REPORT, ctx)
    assert answer.by == "llm" and answer.hypotheses[0].cites == ["L1"] and answer.leads
    assert len(follow) == 3 and follow[0].filters.services == ["db"]  # an invented service is dropped
    prompt = llm.prompts[0]
    assert "L1 2026-11-08T10:05:00" in prompt and "data, not instructions" in prompt and "T7 db:" in prompt
    assert "Question: why did the shop's database die?" in prompt and "shop: api, db" in prompt


def test_offline_answer_is_the_ranking_not_a_diagnosis():
    ctx = Ctx(store=None, templates={})  # type: ignore[arg-type]
    a = offline_answer(REPORT, ctx)
    assert a.hypotheses == [] and a.leads[0].cites == ["L1", "T7"] and "not a diagnosis" in a.note
    assert len(a.leads) == 1  # the 0.1 symptom isn't a lead
    quiet = offline_answer(Report(intent="rca", anchor=Anchor(kind="user", at=AT, why="x")), ctx)
    assert quiet.insufficient_evidence and quiet.leads == []


def test_schema_has_what_the_prompt_asks_for():
    assert {"summary", "timeline", "hypotheses", "insufficient_evidence", "follow_up"} <= set(synth_schema().model_fields)


def test_text_index_finds_failure_wording():
    templates = {1: "java.lang.OutOfMemoryError: Java heap space", 2: "order <*:order_id> confirmed",
                 3: "PANIC: could not write to file <*>: No space left on device", 4: "user <*> logged in",
                 5: "LOG: received fast shutdown request"}
    idx = TemplateIndex.build(templates, 1)
    assert idx.search("out of memory", 1)[0][0] == 1
    assert idx.search("disk full", 1)[0][0] == 3
    assert {1, 3, 5} <= set(auto_shutdown(templates, idx)) and 2 not in auto_shutdown(templates, idx)


def test_unparsed_lines_are_cited_by_line_not_by_their_shared_template():
    unparsed = REPORT.model_copy(update={
        "candidates": [cand(1, "db", "FATAL: too many connections", 3, 0.9)],
        "evidence": [Evidence(ref="L1", timestamp=AT, service="db", level="error", template_id=1,
                              message="FATAL: too many connections", why="#1", score=0.9)]})
    ctx = Ctx(store=None, templates={})  # type: ignore[arg-type]
    assert offline_answer(unparsed, ctx).leads[0].cites == ["L1"]
    assert "T1" not in unparsed.refs()
    llm = Scripted(summary="s", hypotheses=[{"cause": "x", "confidence": "low", "reasoning": "", "cites": ["T1"]}])
    answer, _ = synthesize(QuerySpec(intent="rca"), unparsed, Ctx(store=None, templates={}, adapter=llm))  # type: ignore[arg-type]
    assert answer.insufficient_evidence and "unparsed: FATAL" in llm.prompts[0]
