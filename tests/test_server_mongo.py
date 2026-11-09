"""``logpipe server --store mongodb`` on a live MongoDB: intake, UI reads, a restart, metrics
(``pytest -m integration``; ``MONGO_URI``). Mirrors ``test_server_clickhouse.py``, minus raw SQL."""
from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from logpipe.enrich.llm import get_adapter
from logpipe.server.app import create_app
from logpipe.server.control import ControlStore
from logpipe.server.core import Server
from logpipe.storage.mongo import MongoDocs, MongoStore

pytestmark = pytest.mark.integration


@pytest.fixture
def store(monkeypatch):
    monkeypatch.delenv("MONGO_READONLY_URI", raising=False)  # .env may leak in (load_env); free-form queries then use the main user
    probe = MongoClient(os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017"), serverSelectionTimeoutMS=2000)
    try:
        probe.admin.command("ping")
        probe[f"logpipe_probe_{uuid.uuid4().hex[:8]}"].list_collection_names()  # ping works without auth; this needs rights
    except PyMongoError:
        pytest.skip("no MongoDB reachable or not allowed to use it (set MONGO_URI with a user that can create databases)")
    finally:
        probe.close()
    st = MongoStore.from_env(database=f"logpipe_server_test_{uuid.uuid4().hex[:8]}")
    yield st
    st.db.client.drop_database(st.db.name)
    st.close()


def make(store, tmp_path):
    server = Server(store, ControlStore(MongoDocs(store.db)), data_dir=tmp_path,
                    adapter_factory=lambda: get_adapter("offline"), epochs=1, start_threads=False)
    return server, TestClient(create_app(server))


def test_server_on_mongodb(store, tmp_path):
    server, client = make(store, tmp_path)
    token = client.post("/api/keys", json={"name": "ci"}).json()["token"]
    headers = {"X-API-Key": token}
    lines = [f"job {i} finished in {i % 50} ms on worker {i % 4}" for i in range(57)] + [
        "ERROR [db] lost connection", "WARN disk 91% full",
        '10.0.0.1 - - [x] "GET /a HTTP/1.1" 503 12']
    r = client.post("/intake/v1/logs", headers=headers, json={"events": [
        {"host": "h1", "service": "jobs", "source": "go", "tags": ["env:ci"], "message": m} for m in lines]})
    assert r.status_code == 202
    while not server.inbox.empty():
        server.process(server.inbox.get_nowait())
    server.persist()

    logs = client.get("/api/logs", params={"service": "jobs", "limit": 3}).json()["logs"]
    assert len(logs) == 3 and logs[0]["host"] == "h1" and logs[0]["tags"] == ["env:ci"]
    assert client.get("/api/stats").json()["services"] == [["jobs", 60]]
    facets = client.get("/api/logs/facets", params={"minutes": 60}).json()
    assert dict(facets["level"]) == {"info": 57, "error": 2, "warn": 1}
    errors = client.get("/api/logs", params={"level": "error", "minutes": 60}).json()["logs"]
    assert len(errors) == 2 and {r["level"] for r in errors} == {"error"}
    hist = client.get("/api/logs/histogram", params={"minutes": 15}).json()
    assert sum(p["error"] for p in hist["points"]) == 2 and len(hist["points"]) == 90

    # Raw SQL is the one feature MongoDB does not have: a clear 400, not a crash.
    bad = client.post("/api/query", json={"sql": "SELECT 1"})
    assert bad.status_code == 400 and "not supported on MongoDB" in bad.text

    # The Query page: the builder's pipeline text runs through /api/query; stages that write are refused.
    by_service = client.post("/api/query/build", json={"minutes": 60, "group_by": ["service"]}).json()
    assert by_service["result"]["columns"] == ["service", "lines"] and by_service["result"]["rows"] == [["jobs", 60]]
    took = client.post("/api/query/build", json={"minutes": 60, "measure": {"fn": "max", "field": "@nope"}}).json()
    assert took["result"]["rows"] == [[None]]
    for fn, field, want in (("count_distinct", "service", 1), ("p95", "template_id", None), ("sum", "template_id", None)):
        r = client.post("/api/query/build", json={"minutes": 60, "measure": {"fn": fn, "field": field}}).json()["result"]
        assert len(r["rows"]) == 1 and (want is None or r["rows"] == [[want]]), (fn, r)
    hot = client.post("/api/query/build", json={"minutes": 60, "filters": [
        {"field": "message", "op": "contains", "value": "disk"}, {"field": "host", "value": "h1"}]}).json()
    assert hot["result"]["rows"] == [[1]]
    over = client.post("/api/query/build", json={"minutes": 60, "over_time": True}).json()
    assert over["result"]["columns"] == ["time", "lines"] and sum(r[-1] for r in over["result"]["rows"]) == 60
    quoted = client.post("/api/query/build", json={"filters": [{"field": "host", "value": "x' OR '1'='1"}]}).json()
    assert quoted["result"]["rows"] == []  # the value stays a value (SQL stores answer [[0]])
    assert client.post("/api/query", json={"sql": by_service["sql"]}).json()["rows"] == [["jobs", 60]]
    for bad in ('[{"$out": "x"}]', '[{"$match": {"$where": "1"}}]', '[{"$lookup": {"from": "control_docs"}}]'):
        assert client.post("/api/query", json={"sql": bad}).status_code == 400, bad
    for spec in ({"group_by": ["host; DROP"]}, {"measure": {"fn": "evil"}}, {"filters": [{"field": "level", "op": ">", "value": "e"}]}):
        assert client.post("/api/query/build", json=spec).status_code == 400, spec
    assert client.get("/api/query/schema").json()["engine"] == "mongodb"
    plan = client.post("/api/ask", json={"question": "how many lines per service in the last hour"}).json()
    assert plan["status"] == "waiting_approval" and plan["rounds"][0]["queries"][0]["engine"] == "mongodb"
    client.post(f"/api/ask/{plan['id']}/approve")
    while not server.jobs.empty():
        server.run_job(server.jobs.get_nowait())
    done = client.get(f"/api/ask/{plan['id']}").json()
    assert done["status"] == "done" and done["result"]["rows"] == [["jobs", 60]]

    # A restart reads keys, sources and streams back from control_docs.
    server2, client2 = make(store, tmp_path)
    assert client2.get("/intake/v1/ping", headers=headers).status_code == 200
    assert [s["id"] for s in client2.get("/api/sources").json()["sources"]] == ["h1|jobs|go"]
    assert server2.sources["h1|jobs|go"]["count"] == 60


def test_metrics_on_mongodb(store, tmp_path):
    server, client = make(store, tmp_path)
    server.metrics.parsed(5000, unknown=False)
    server.sample_metrics()
    server.sample_metrics()
    body = client.get("/api/system?minutes=5").json()
    assert body["current"]["threads"] >= 1
    assert len(body["series"]["rss_mb"]) >= 1 and body["series"]["lines_per_s"][0][0] > 1.7e9
