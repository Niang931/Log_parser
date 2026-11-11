"""Threshold alerts on the built-in metrics: fire after the breach lasts, resolve when back."""

from __future__ import annotations

import pytest

from logpipe.server.control import ControlStore
from logpipe.server.metrics import DEFAULT_RULES, MetricRules
from tests.test_server import env  # noqa: F401 - fixture


def test_defaults_are_seeded_once():
    control = ControlStore()
    rules = MetricRules(control)
    assert len(rules.rules()) == len(DEFAULT_RULES)
    rules.delete("default-cpu_pct")
    # a restart doesn't re-add it
    assert len(MetricRules(control).rules()) == len(DEFAULT_RULES) - 1


def test_breach_must_last_then_resolves():
    rules = MetricRules(ControlStore())
    t = 1_000_000.0
    assert rules.evaluate({"cpu_pct": 95}, t) == []
    assert rules.evaluate({"cpu_pct": 97}, t + 300) == []  # 5 of the 10 minutes
    rules.evaluate({"cpu_pct": 50}, t + 400)  # dipped: the clock restarts
    rules.evaluate({"cpu_pct": 95}, t + 500)
    fired = rules.evaluate({"cpu_pct": 99}, t + 500 + 600)
    assert [a["metric"] for a in fired] == ["cpu_pct"] and fired[0]["started_at"] == t + 500
    assert rules.evaluate({"cpu_pct": 98}, t + 1200) == []  # already firing: no second alert
    firing = rules.alerts()[0]
    assert firing["status"] == "firing" and firing["peak"] == 99
    rules.evaluate({"cpu_pct": 10}, t + 1300)
    done = rules.alerts()
    assert len(done) == 1 and done[0]["status"] == "resolved" and done[0]["resolved_at"] == t + 1300


def test_below_rules_missing_metrics_and_disabled_rules():
    rules = MetricRules(ControlStore())
    rule = rules.save({"metric": "lines_per_s", "op": "<", "threshold": 1, "for_minutes": 0})
    assert [a["rule"] for a in rules.evaluate({"lines_per_s": 0}, 0)] == [rule["id"]]
    rules.evaluate({}, 10)  # metric not sampled: not a breach
    assert all(a["status"] == "resolved" for a in rules.alerts())
    rules.save({"enabled": False}, rule["id"])
    assert rules.evaluate({"lines_per_s": 0}, 20) == []
    with pytest.raises(ValueError):
        rules.save({"metric": "nope", "op": ">", "threshold": 1})
    with pytest.raises(ValueError):
        rules.save({"metric": "cpu_pct", "op": "!=", "threshold": 1})


def test_rules_api_and_server_sampling(env):  # noqa: F811
    server, client, _ = env
    body = client.get("/api/system/alerts").json()
    assert len(body["rules"]) == len(DEFAULT_RULES) and body["alerts"] == []
    rid = client.post(
        "/api/system/rules", json={"metric": "threads", "op": ">", "threshold": 0, "for_minutes": 0}
    ).json()["id"]
    server.sample_metrics()
    alerts = client.get("/api/system/alerts").json()["alerts"]
    assert alerts[0]["rule"] == rid and alerts[0]["status"] == "firing"
    assert (
        client.put(f"/api/system/rules/{rid}", json={"threshold": 10_000}).json()["threshold"]
        == 10_000
    )
    server.sample_metrics()
    assert client.get("/api/system/alerts").json()["alerts"][0]["status"] == "resolved"
    assert client.put("/api/system/rules/missing", json={}).status_code == 404
    assert client.delete(f"/api/system/rules/{rid}").json() == {"ok": True}
