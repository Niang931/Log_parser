"""The Mongo Query-page builder and its read-only guard (no database needed)."""
from __future__ import annotations

import json

import pytest

from logpipe.server.querybuild_mongo import build, check


@pytest.mark.parametrize("stage", [
    {"$out": "x"}, {"$merge": {"into": "x"}}, {"$lookup": {"from": "control_docs"}},
    {"$unionWith": "control_docs"}, {"$collStats": {}}])
def test_check_rejects_write_and_foreign_stages(stage):
    with pytest.raises(ValueError, match="not allowed"):
        check([{"$match": {}}, stage])


@pytest.mark.parametrize("bad", [
    {"$match": {"$where": "1"}}, {"$match": {"$expr": {"$function": {"body": "x", "args": [], "lang": "js"}}}},
    {"$group": {"_id": None, "a": {"$accumulator": {}}}}, {"$addFields": {"x": [{"$where": "1"}]}}])
def test_check_rejects_code_operators_at_any_depth(bad):
    with pytest.raises(ValueError, match="not allowed"):
        check([bad])


def test_check_accepts_read_only_pipeline():
    pipeline = [{"$match": {"service": "api"}}, {"$group": {"_id": "$host", "n": {"$sum": 1}}}, {"$limit": 5}]
    assert check(pipeline) is pipeline


@pytest.mark.parametrize("not_pipeline", [{}, [], "x", [{"$match": {}, "$limit": 1}], [1]])
def test_check_rejects_non_pipelines(not_pipeline):
    with pytest.raises(ValueError):
        check(not_pipeline)


def test_build_shapes():
    p = json.loads(build({"minutes": 30, "group_by": ["service"], "filters": [
        {"field": "host", "op": "!=", "value": "h1"}, {"field": "message", "op": "contains", "value": "a.b"}]}))
    assert check(p) == p
    flt = p[0]["$match"]["$and"]
    assert flt[1] == {"device_id": {"$ne": "h1"}} and flt[2] == {"raw_message": {"$regex": r"a\.b", "$options": "i"}}
    assert p[1]["$group"] == {"_id": {"service": "$service"}, "lines": {"$sum": 1}}
    assert list(p[2]["$project"]) == ["_id", "service", "lines"]  # column order is the spec order
    over = json.loads(build({"over_time": True, "measure": {"fn": "p95", "field": "@took_ms"}}))
    assert "$percentile" in json.dumps(over) and list(over[-3]["$project"])[1:] == ["time", "p95_took_ms"]


@pytest.mark.parametrize("bad", [
    {"group_by": ["host; DROP"]}, {"filters": [{"field": "@x'y", "value": "1"}]}, {"measure": {"fn": "evil"}},
    {"filters": [{"field": "host", "op": ">", "value": "abc"}]}, {"filters": [{"field": "level", "op": ">", "value": "error"}]},
    {"filters": [{"field": "template_id", "value": "x"}]}, {"minutes": 10**9}, {"measure": {"fn": "avg"}}])
def test_build_rejects_bad_specs(bad):
    with pytest.raises(ValueError):
        build(bad)
