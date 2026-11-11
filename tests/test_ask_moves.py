"""Value hints (where errors concentrate) and the typed follow-up moves the LLM may ask for."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import get_args

import pytest
from pydantic import ValidationError

from logpipe.ask.engine import LOG_READS, Ctx, Round
from logpipe.ask.hints import value_hints
from logpipe.ask.models import Anchor, Evidence, LogLine, Report
from logpipe.ask.moves import MoveKind, MoveRequest, absorb, check_moves, move_round
from logpipe.ask.query import LogTool
from logpipe.ask.synth import check_citations, leads
from logpipe.server.stores.memory import MemoryReads
from logpipe.settings import get_settings
from logpipe.storage.memory import MemoryStore

AT = datetime(2026, 11, 8, 10, 5, tzinfo=UTC)
CFG = get_settings().rca


def line(i: int, host: str, level: str, **payload) -> LogLine:
    return LogLine(
        timestamp=AT - timedelta(seconds=i),
        host=host,
        service="api",
        level=level,
        message=f"{level} {i}",
        payload=payload,
    )


def test_a_host_the_errors_pile_up_on_is_a_hint():
    errors = [line(i, "web-3", "error", upstream="pay-2", order=str(i)) for i in range(20)] + [
        line(99, "web-1", "error", upstream="pay-1")
    ]
    others = [
        line(
            i,
            "web-3" if i % 10 == 0 else f"web-{1 + i % 2}",
            "info",
            upstream=f"pay-{1 + i % 2}",
            order=str(100 + i),
        )
        for i in range(60)
    ]
    hints = value_hints(errors, others, CFG)
    found = {(h.field, h.value) for h in hints}
    assert ("host", "web-3") in found and (
        "@upstream",
        "pay-2",
    ) not in found  # pay-2 is as common among the others
    assert not any(h.field == "@order" for h in hints)  # an id: every line its own value
    h = next(h for h in hints if h.value == "web-3")
    assert h.ref == "H1" and h.error_share > 0.9 and h.other_share == 0.1 and h.ratio >= 5
    assert value_hints(errors, others[:5], CFG) == []  # nothing to contrast with
    # a field only error lines carry is the error template's own slot: no hint
    assert value_hints([line(i, "web-1", "error", code="503") for i in range(9)], others, CFG) == []


def test_payload_comes_back_as_fields_from_json_text():
    assert LogLine.model_validate(
        {"timestamp": AT, "message": "x", "payload": '{"a": 1}'}
    ).payload == {"a": 1}
    assert LogLine.model_validate({"timestamp": AT, "message": "x", "payload": None}).payload == {}


REPORT = Report(
    intent="rca",
    anchor=Anchor(kind="new_error", at=AT, why="x", service="api"),
    evidence=[
        Evidence(
            ref="L1",
            timestamp=AT,
            service="api",
            message="order 7 failed (request req-8f3a)",
            why="#1",
            score=0.9,
        )
    ],
)


def ctx(store=None) -> Ctx:
    services = {"api", "db", "dns"}
    return Ctx(store=store, templates={}, services=services, now=AT + timedelta(minutes=10))  # type: ignore[arg-type]


def test_moves_are_checked_before_they_become_reads():
    raw = [
        {"kind": "trace", "value": "req-8f3a"},
        {"kind": "trace", "value": "req-0000"},  # not in the bundle: a guess
        {"kind": "rate", "services": ["db", "invented"], "minutes_before": 10_000},
        {"kind": "lines"},  # nothing to search for
        {"kind": "neighbours"},
        {"kind": "lines", "keywords": ["x"]},
    ]
    moves = check_moves([MoveRequest.model_validate(m) for m in raw], REPORT, ctx())
    assert [m.kind for m in moves] == ["trace", "rate", "neighbours"]  # capped at moves_per_round
    trace, rate, near = moves
    # a trace reads every service
    assert (
        trace.kind is MoveKind.TRACE
        and trace.filters.keywords == ["req-8f3a"]
        and trace.filters.services == []
    )
    assert rate.filters.services == ["db"] and AT - rate.start == timedelta(
        hours=CFG.move_max_hours
    )
    assert near.filters.levels == ["error", "warn"] and near.end == AT + timedelta(minutes=2)


def test_a_kind_that_is_not_a_move_is_refused_when_the_answer_is_read():
    with pytest.raises(ValidationError):
        MoveRequest.model_validate({"kind": "drop table"})


def test_follow_up_reads_compile_and_come_back_as_lines_and_rates():
    store = MemoryStore()
    reads = MemoryReads(store)
    c = ctx(reads)
    requests = [
        MoveRequest(kind=MoveKind.TRACE, value="req-8f3a", why="follow the request"),
        MoveRequest(kind=MoveKind.RATE, services=["db"], minutes_before=30, minutes_after=1),
        MoveRequest(kind=MoveKind.NEIGHBOURS),
    ]
    moves = check_moves(requests, REPORT, c)
    r = move_round(3, moves, c, "follow-up 1 of 4")
    trace, rate, near = r.queries
    assert (
        trace.tool == "rows"
        and "req-8f3a" in trace.compiled
        and "service" not in trace.compiled.split("WHERE")[1]
    )
    assert rate.tool == "buckets" and rate.args["step"] == 60 and near.tool == "template_counts"
    outputs = {
        trace.id: [
            {
                "timestamp": (AT - timedelta(minutes=1)).isoformat(),
                "service": "db",
                "level": "warn",
                "message": "req-8f3a waited 30 s for a row lock held by autovacuum",
            }
        ],
        rate.id: [{"key": "db", "bucket": 28, "n": 40}],
        near.id: [
            {
                "template_id": 9,
                "service": "dns",
                "n": 3,
                "first": AT.isoformat(),
                "last": AT.isoformat(),
                "example": "SERVFAIL payments.internal",
                "level": "error",
            }
        ],
    }
    report = absorb(REPORT.model_copy(deep=True), Round(3, "x", r.queries), outputs, c)
    msgs = [e.message for e in report.evidence]
    assert msgs[0].startswith("req-8f3a waited") and "SERVFAIL payments.internal" in msgs
    assert report.evidence[0].why.startswith("follow-up trace") and report.evidence[0].ref == "L1"
    s1 = report.series[0]
    assert s1.ref == "S1" and s1.service == "db" and len(s1.counts) == 31 and s1.counts[28] == 40
    answer = check_citations(
        {"summary": "s", "hypotheses": [{"cause": "lock", "cites": ["L1", "S1", "H9"]}]}, report
    )
    assert answer.hypotheses[0].cites == ["L1", "S1"]


def test_a_strong_hint_is_a_lead_without_an_llm():
    from logpipe.ask.models import Hint

    report = REPORT.model_copy(
        update={
            "hints": [
                Hint(
                    ref="H1",
                    field="host",
                    value="web-3",
                    errors=40,
                    error_share=0.95,
                    other_share=0.02,
                    ratio=40,
                )
            ]
        }
    )
    found = leads(report, ctx())
    assert found[0].cause == "errors concentrate on host=web-3" and found[0].cites == ["H1"]


def test_follow_up_lines_are_varied_not_forty_copies():
    from logpipe.ask.moves import varied

    same = [
        LogLine(
            timestamp=AT + timedelta(seconds=i),
            service="api",
            template_id=7,
            message=f"order {i} failed",
        )
        for i in range(40)
    ]
    other = [LogLine(timestamp=AT, service="db", template_id=8, message="lock wait")]
    assert len(varied(same + other)) == 4


def test_every_log_read_has_a_compiler():
    assert set(LOG_READS) == set(get_args(LogTool))
