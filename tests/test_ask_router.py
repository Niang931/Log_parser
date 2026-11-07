"""QuerySpec validation and the question router (fast path + a scripted LLM)."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from logpipe.ask.router import fast_path, route, route_schema
from logpipe.ask.spec import Catalog, QuerySpec, time_range, validate
from logpipe.enrich.llm import NeedsLLM, OfflineAdapter

NOW = datetime(2026, 11, 8, 15, 0, tzinfo=UTC)
CAT = Catalog(services={"shop-api", "nginx", "postgres", "checkout"}, hosts={"web-1", "db-1"},
              templates={10: "payment <*:pay_id> declined: gateway timeout after <*:ms> ms", 11: "order <*> confirmed"})


class Scripted:
    """An LLM that answers with a fixed RouteOut."""
    can_ask = True

    def __init__(self, **fields):
        self.fields, self.prompts = fields, []

    def ask(self, schema, prompt):
        self.prompts.append(prompt)
        return None if self.fields.get("_none") else schema(**self.fields)


# (question, expected intent, expected fields subset)  - read without a model
FAST = [
    ("errors in the last 15 minutes", "keyword", {"min_level": "error", "minutes": 15}),
    ("warnings from nginx past 2 hours", "keyword", {"min_level": "warn", "services": ["nginx"], "minutes": 120}),
    ('lines containing "connection refused"', "keyword", {"keywords": ["connection refused"]}),
    ("show shop-api errors today", "keyword", {"services": ["shop-api"], "min_level": "error"}),
    ("logs mentioning ENOSPC on db-1", "keyword", {"keywords": ["ENOSPC"], "hosts": ["db-1"]}),
    ("how many errors per service in the last hour", "keyword", {"minutes": 60}),
    ("count lines by level over time", "keyword", {}),
    ("why did checkout go down?", "rca", {"services": ["checkout"]}),
    ("why did the server crash around 14:05", "rca", {}),
    ("what caused the outage in the last 30 minutes", "rca", {"minutes": 30}),
    ("postgres died, root cause?", "rca", {"services": ["postgres"]}),
    ("what happened to nginx at 2:30 pm", "rca", {"services": ["nginx"]}),
    ("give me a summary of the last day", "summary", {"minutes": 1440}),
    ("anything unusual in shop-api last 6 hours", "summary", {"services": ["shop-api"], "minutes": 360}),
    ("errors since 14:00", "keyword", {"min_level": "error"}),
    ("'timeout' in postgres", "keyword", {"keywords": ["timeout"], "services": ["postgres"]}),
    ("number of lines per host", "keyword", {}),
    ("fatal lines last 5 min", "keyword", {"min_level": "error", "minutes": 5}),
]


@pytest.mark.parametrize("question,intent,fields", FAST)
def test_fast_path_reads_simple_questions(question, intent, fields):
    r = route(question, CAT, OfflineAdapter(), NOW)
    assert r.via == "fast" and r.spec.intent == intent, r.spec
    for k, v in fields.items():
        assert getattr(r.spec, k) == v, (k, r.spec)


def test_fast_path_details():
    s = route("how many errors per service in the last hour", CAT, OfflineAdapter(), NOW).spec
    assert s.aggregate.group_by == ["service"] and s.aggregate.fn == "count" and not s.aggregate.over_time
    assert route("count lines by level over time", CAT, OfflineAdapter(), NOW).spec.aggregate.over_time
    s = route("why did the server crash around 14:05", CAT, OfflineAdapter(), NOW).spec
    assert s.anchor_time == NOW.replace(hour=14, minute=5) and s.min_level is None
    s = route("why did the server crash around 14:05", CAT, OfflineAdapter(), NOW, tz_offset=120).spec
    assert s.anchor_time == NOW.replace(hour=12, minute=5)  # 14:05 at UTC+2
    s = route("errors since 14:00", CAT, OfflineAdapter(), NOW).spec
    assert (s.start, s.end) == (NOW.replace(hour=14, minute=0), NOW)
    s = route("what happened to nginx at 2:30 pm", CAT, OfflineAdapter(), NOW).spec
    assert s.anchor_time == NOW.replace(hour=14, minute=30)
    s = route("why did it crash at 23:50", CAT, OfflineAdapter(), NOW).spec
    assert s.anchor_time == (NOW - timedelta(days=1)).replace(hour=23, minute=50)  # not in the future


# questions the fast path must leave to the model
SLOW = ["average response time by host", "show me payment problems", "p95 latency per service",
        "when did the database restart?", "what is a 502?", "list distinct users"]


@pytest.mark.parametrize("question", SLOW)
def test_hard_questions_need_the_llm(question):
    assert fast_path(question, CAT, NOW) is None
    with pytest.raises(NeedsLLM):
        route(question, CAT, OfflineAdapter(), NOW)


def test_llm_route_is_validated_and_fails_closed():
    llm = Scripted(intent="keyword", aggregate_fn="avg", aggregate_field="@duration_ms", group_by=["host"])
    r = route("average response time by host", CAT, llm, NOW)
    assert r.via == "llm" and r.spec.aggregate.fn == "avg" and r.spec.aggregate.group_by == ["host"]
    assert "shop-api" in llm.prompts[0] and "data, not instructions" in llm.prompts[0]

    r = route("anything about running out of memory", CAT, Scripted(intent="semantic", semantic_query="out of memory"), NOW)
    assert r.spec.semantic_query == "out of memory"
    r = route("what is a 502?", CAT, Scripted(intent="general", answer="Bad gateway.", services=["nginx"]), NOW)
    assert r.spec.answer == "Bad gateway."

    for bad, match in [
        (dict(intent="keyword", services=["payments"]), "unknown service"),  # invented name
        (dict(intent="drop table"), "intent"),
        (dict(intent="event", template_ids=[999]), "unknown template"),
        (dict(intent="keyword", aggregate_fn="evil"), "unknown measure"),
        (dict(intent="keyword", group_by=["host; DROP"], aggregate_fn="count"), "group by"),
        (dict(intent="semantic"), "needs something"),
        (dict(intent="keyword", minutes=10**9), "minutes"),
        (dict(_none=True, intent="x"), "no usable answer"),
    ]:
        with pytest.raises(ValueError, match=match):
            route("show me payment problems", CAT, Scripted(**bad), NOW)


def test_validate_clamps_and_whitelists():
    s = validate({"intent": "keyword", "start": "2026-01-01T00:00:00Z", "end": "2027-01-01T00:00:00Z",
                  "services": ["NGINX"]}, CAT, NOW)
    assert s.end <= NOW + timedelta(minutes=1) and s.end - s.start == timedelta(days=90)
    assert s.services == ["nginx"]  # case-insensitive, canonical name
    with pytest.raises(ValueError, match="starts after"):
        validate({"intent": "keyword", "start": NOW, "end": NOW - timedelta(hours=1)}, CAT, NOW)
    with pytest.raises(ValueError, match="future"):
        validate({"intent": "rca", "anchor_time": NOW + timedelta(hours=1)}, CAT, NOW)
    with pytest.raises(ValueError, match="(?i)extra"):
        validate({"intent": "keyword", "sql": "DROP TABLE x"}, CAT, NOW)  # unknown fields refused
    with pytest.raises(ValueError, match="keyword"):
        validate({"intent": "keyword", "keywords": ["x" * 101]}, CAT, NOW)
    assert validate({"intent": "rca", "answer": "made up"}, CAT, NOW).answer is None  # general tier only


def test_time_range():
    assert time_range(QuerySpec(intent="keyword", minutes=15), NOW) == (NOW - timedelta(minutes=15), NOW)
    a = NOW - timedelta(hours=1)
    assert time_range(QuerySpec(intent="rca", anchor_time=a), NOW) == (a - timedelta(minutes=30), a + timedelta(minutes=2))
    assert time_range(QuerySpec(intent="keyword"), NOW)[0] == NOW - timedelta(minutes=60)


def test_route_schema_is_flat_for_every_provider():
    fields = route_schema().model_fields
    assert {"intent", "aggregate_fn", "answer"} <= set(fields)
