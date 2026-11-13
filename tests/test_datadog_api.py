import json
from pathlib import Path
from types import SimpleNamespace

import httpx

from logpipe.datadog_api import DatadogAPI, DatadogSync, dd_rules, to_dd_rule
from logpipe.parsing.grok import rule_from_template
from logpipe.server.control import ControlStore
from logpipe.settings import DatadogSettings


def _api(handler, **kw):
    settings = DatadogSettings(api_key="k", app_key="a", **kw)
    return DatadogAPI(settings, transport=httpx.MockTransport(handler)), settings


def test_rules_translate_to_datadog_grok():
    rule = "[%{integer:pid}] GET %{path:url} took %{number:ms}ms from %{notspace:ip} %{quoted:ua}"
    assert to_dd_rule(rule) == (
        r"\[%{integer:pid}\] GET %{notSpace:url} took %{number:ms}ms from %{notSpace:ip} "
        r"%{quotedString:ua}"
    )
    assert to_dd_rule('%{regex("[a-z]+"):w} x.y') == r'%{regex("[a-z]+"):w} x\.y'
    assert "regex(" in to_dd_rule("%{timestamp:ts} up")
    rules = dd_rules([{"template_id": 7, "rule": rule_from_template(["user", "<*>"], ["name"])}])
    assert rules == ["t7 user %{notSpace:name}"]


def test_search_maps_logs_to_intake_events():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["headers"] = request.headers
        attrs = {
            "message": "boom",
            "host": "web-1",
            "service": "shop",
            "tags": ["source:nginx", "env:prod"],
            "timestamp": "2026-11-12T10:00:00.123Z",
        }
        return httpx.Response(200, json={"data": [{"attributes": attrs}]})

    api, _ = _api(handler)
    events = api.search_logs("service:shop", "2026-11-12T09:00:00Z")
    assert seen["body"]["filter"] == {
        "query": "service:shop",
        "from": "2026-11-12T09:00:00Z",
        "to": "now",
    }
    assert seen["headers"]["DD-APPLICATION-KEY"] == "a"
    assert events == [
        {
            "message": "boom",
            "host": "web-1",
            "service": "shop",
            "source": "nginx",
            "tags": ["source:nginx", "env:prod"],
            "ts": "2026-11-12T10:00:00.123Z",
        }
    ]


class FakeServer:
    def __init__(self):
        self.control = ControlStore()
        self.ingested = []
        self.incident_list = []
        self.explained = []
        self.plans = {}
        self.ask = SimpleNamespace(get=lambda plan_id: self.plans[plan_id])

    def ingest(self, events):
        self.ingested += events

    def incidents(self):
        return self.incident_list

    def explain_incident(self, incident_id, require_approval=None):
        assert require_approval is False
        self.explained.append(incident_id)
        doc = self.control.get("incident", incident_id) or {"id": incident_id}
        self.control.put("incident", incident_id, doc | {"ask_plan": "p1"})


INC = {
    "id": "a1",
    "key": "web-1/shop",
    "host": "web-1",
    "service": "shop",
    "start": "2026-11-12T10:00:00+00:00",
    "end": "2026-11-12T10:05:00+00:00",
    "alerts": 3,
    "peak_miss": 0.4,
}


def _sync(settings_kw=None, logs=()):
    posted = []

    def handler(request):
        if request.url.host.startswith("event-management-intake."):
            e = json.loads(request.content)["data"]["attributes"]
            posted.append(e | {"text": e["message"]})
            return httpx.Response(202, json={"status": "ok"})
        data = [{"attributes": {"message": m, "timestamp": ts}} for m, ts in logs]
        return httpx.Response(200, json={"data": data})

    api, settings = _api(handler, pull_every_s=0, **(settings_kw or {}))
    return DatadogSync(settings, api=api), posted


def test_pull_ingests_and_advances_the_cursor():
    sync, _ = _sync({"pull_query": "service:shop"}, [("a", "2026-11-12T10:00:00.000+00:00")])
    server = FakeServer()
    sync.tick(server)
    assert [e["message"] for e in server.ingested] == ["a"]
    assert (server.control.get("datadog", "pull") or {})[
        "since"
    ] == "2026-11-12T10:00:00.001000+00:00"


def test_new_incident_posts_an_event_then_its_root_cause():
    sync, posted = _sync({"public_url": "https://lp.example"})
    server = FakeServer()
    server.incident_list = [INC | {"id": "old"}]
    sync.tick(server)  # history is not posted
    assert posted == [] and server.explained == []

    server.incident_list.append(INC)
    sync.tick(server)
    assert server.explained == ["a1"]
    assert posted[0]["title"] == "logpipe: anomaly on web-1/shop"
    assert posted[0]["aggregation_key"] == "a1"
    assert "https://lp.example/alerts/a1" in posted[0]["text"]
    assert "service:shop" in posted[0]["tags"]

    sync.tick(server)  # no explanation yet: nothing new
    assert len(posted) == 1
    doc = server.control.get("incident", "a1") or {}
    answer = {"summary": "payment gateway timed out", "hypotheses": [{"cause": "gw down"}]}
    server.control.put("incident", "a1", doc | {"explanation": {"answer": answer}})
    sync.tick(server)
    sync.tick(server)
    assert len(posted) == 2
    assert posted[1]["title"] == "logpipe RCA: web-1/shop"
    assert "payment gateway timed out" in posted[1]["text"] and "gw down" in posted[1]["text"]


def test_webhook_runs_an_rca_and_posts_it_when_done():
    sync, posted = _sync()
    server = FakeServer()
    asked = []

    def ask(question, spec, require_approval):
        asked.append(spec)
        server.plans["p9"] = {"id": "p9", "status": "running"}
        return server.plans["p9"]

    server.ask.ask = ask
    sync.alert(
        server,
        {"title": "High 5xx", "date": "1762941600000", "tags": "service:shop,env:prod", "id": "e1"},
    )
    assert asked[0]["services"] == ["shop"] and asked[0]["anchor_time"].startswith("2025-11-12")
    sync.tick(server)
    server.plans["p9"] = {"status": "waiting_approval"}
    sync.tick(server)
    assert posted == []
    server.plans["p9"] = {"status": "done", "result": {"answer": {"summary": "db pool exhausted"}}}
    sync.tick(server)
    assert posted[0]["aggregation_key"] == "e1" and "db pool exhausted" in posted[0]["text"]
    sync.tick(server)
    assert len(posted) == 1


def test_server_routes(tmp_path):
    from fastapi.testclient import TestClient

    from logpipe.enrich.llm import get_adapter
    from logpipe.server.app import create_app
    from logpipe.server.core import Server
    from logpipe.storage.memory import MemoryStore

    server = Server(
        MemoryStore(),
        ControlStore(),
        data_dir=tmp_path,
        adapter_factory=lambda: get_adapter("offline"),
        start_threads=False,
    )
    client = TestClient(create_app(server))
    key = {"X-API-Key": client.post("/api/keys", json={"name": "dd"}).json()["token"]}
    hook = {"title": "High 5xx", "tags": "service:shop"}
    assert client.post("/intake/v1/datadog/webhook", json=hook).status_code == 401
    assert client.post("/intake/v1/datadog/webhook", json=hook, headers=key).status_code == 409

    server.dd_sync, _ = _sync()
    r = client.post("/intake/v1/datadog/webhook", json=hook, headers=key)
    assert r.status_code == 202, r.text
    docs = [d for d in server.control.all("datadog") if d.get("plan") == r.json()["id"]]
    assert docs and docs[0]["service"] == "shop"
    assert client.post("/api/datadog/pipeline").status_code == 409


def test_a_deleted_plan_does_not_block_incidents():
    sync, posted = _sync()
    server = FakeServer()
    server.control.put(
        "datadog",
        "alert-x",
        {"id": "alert-x", "plan": "gone", "title": "t", "key": "k", "host": "", "service": ""},
    )
    sync.tick(server)
    server.incident_list = [INC]
    sync.tick(server)
    assert [p["aggregation_key"] for p in posted] == ["k", "a1"]


def test_showcase_creates_what_is_missing_and_deletes_only_that(monkeypatch, tmp_path):
    from logpipe.showcase import datadog as sd

    calls = []
    pipelines = []

    def handler(request):
        calls.append((request.method, request.url.path))
        path = request.url.path
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        if path == "/api/v1/dashboard" and request.method == "GET":
            return httpx.Response(200, json={"dashboards": []})
        if path == "/api/v1/dashboard":
            return httpx.Response(200, json={"id": "d1", "url": "/dashboard/d1"})
        if path == "/api/v1/monitor" and request.method == "GET":
            exists = request.url.params["name"] == "logpipe: anomaly spike"
            return httpx.Response(200, json=[{"id": 1}] if exists else [])
        if path == "/api/v1/monitor":
            return httpx.Response(200, json={"id": len(calls)})
        if path == "/api/v1/logs/config/pipelines":
            return httpx.Response(200, json=pipelines)
        return httpx.Response(404)

    monkeypatch.setenv("DD_API_KEY", "k")
    monkeypatch.setenv("DD_APP_KEY", "a")
    monkeypatch.setattr(sd.subprocess, "run", lambda *a, **kw: None)
    real = sd.DatadogAPI
    monkeypatch.setattr(sd, "DatadogAPI", lambda s: real(s, transport=httpx.MockTransport(handler)))
    dd = sd.ShowcaseDatadog("shop-web-1")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs").symlink_to(Path(__file__).parents[1] / "docs")
    dd.start()
    assert dd.links["dashboard"] == "https://app.datadoghq.com/dashboard/d1"
    assert dd.created[0] == "/api/v1/dashboard/d1" and len(dd.created) == 4  # 3 new monitors

    pipelines.append({"id": "p1", "name": "logpipe (auto)"})  # exported during the tour
    dd.cleanup()
    deleted = [p for m, p in calls if m == "DELETE"]
    assert "/api/v1/logs/config/pipelines/p1" in deleted and "/api/v1/dashboard/d1" in deleted
    assert len(deleted) == 5
