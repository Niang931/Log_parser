"""``logpipe server`` on a live ClickHouse: the agent columns migration, the control docs
surviving a restart, and the UI reads (``pytest -m integration``)."""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from logpipe.enrich.llm import get_adapter
from logpipe.server.app import create_app
from logpipe.server.control import ClickHouseDocs, ControlStore
from logpipe.server.core import Server

pytestmark = pytest.mark.integration


@pytest.fixture
def store():
    from logpipe.storage.clickhouse import ClickHouseStore

    db = f"logpipe_server_test_{uuid.uuid4().hex[:8]}"
    st = ClickHouseStore.from_env(database=db, batch_size=10)
    yield st
    st.client.command(f"DROP DATABASE IF EXISTS `{db}`")
    st.client.close()


def make(store, tmp_path):
    server = Server(store, ControlStore(ClickHouseDocs(store.client)), data_dir=tmp_path,
                    adapter_factory=lambda: get_adapter("offline"), epochs=1, start_threads=False)
    return server, TestClient(create_app(server))


@pytest.mark.parametrize("sql", ["SELECT 1; DROP TABLE iot_logs", "WITH x AS (SELECT 1) SELECT * FROM x; TRUNCATE iot_logs",
                                 "ALTER USER default IDENTIFIED BY 'pwned'", "INSERT INTO iot_logs (raw_message) VALUES ('x')",
                                 "SELECT * FROM url('http://169.254.169.254/latest/meta-data', 'RawBLOB')",
                                 "SELECT * FROM remote('127.0.0.1', system.one)", "SELECT * FROM file('/etc/passwd', 'RawBLOB')"])
def test_query_runs_read_only(store, tmp_path, sql):
    server, _ = make(store, tmp_path)
    with pytest.raises(ValueError):
        server.reads.query(sql)
    assert store.client.query("EXISTS TABLE iot_logs").result_rows == [(1,)]


def test_server_on_clickhouse(store, tmp_path):
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
    q = client.post("/api/query", json={"sql": "SELECT service, count() FROM iot_logs GROUP BY service"}).json()
    assert q["rows"] == [["jobs", 60]]
    assert client.post("/api/query", json={"sql": "SELECT 1; DROP TABLE iot_logs"}).status_code == 400
    assert client.get("/api/stats").json()["services"] == [["jobs", 60]]
    # Level is SQL here and Python for the rows: both must agree.
    facets = client.get("/api/logs/facets", params={"minutes": 60}).json()
    assert dict(facets["level"]) == {"info": 57, "error": 2, "warn": 1}
    errors = client.get("/api/logs", params={"level": "error", "minutes": 60}).json()["logs"]
    assert len(errors) == 2 and {r["level"] for r in errors} == {"error"}
    hist = client.get("/api/logs/histogram", params={"minutes": 15}).json()
    assert sum(p["error"] for p in hist["points"]) == 2 and len(hist["points"]) == 90

    # The Query page's builder writes ClickHouse SQL that runs.
    built = client.post("/api/query/build", json={"minutes": 60, "group_by": ["level"], "over_time": True}).json()
    assert sum(r[-1] for r in built["result"]["rows"]) == 60 and {r[1] for r in built["result"]["rows"]} == {"info", "error", "warn"}
    errors = client.post("/api/query/build", json={"minutes": 60, "filters": [
        {"field": "level", "value": "error"}, {"field": "host", "value": "h1' OR 1=1 --"}]}).json()
    assert errors["result"]["rows"] == [[0]]
    p95 = client.post("/api/query/build", json={"minutes": 60, "measure": {"fn": "count_distinct", "field": "host"}}).json()
    assert p95["result"]["rows"] == [[1]]

    # A restart reads keys, sources and streams back from control_docs.
    server2, client2 = make(store, tmp_path)
    assert client2.get("/intake/v1/ping", headers=headers).status_code == 200
    assert [s["id"] for s in client2.get("/api/sources").json()["sources"]] == ["h1|jobs|go"]
    assert server2.sources["h1|jobs|go"]["count"] == 60


def test_one_client_serves_concurrent_threads(store):
    from concurrent.futures import ThreadPoolExecutor

    docs = ClickHouseDocs(store.client)

    def work(i):
        docs.write("agent", f"h{i}", {"hostname": f"h{i}"})
        return store.client.query("SELECT sleep(0.05), count() FROM control_docs").result_rows

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(work, range(32)))  # used to raise "concurrent queries within the same session"
    assert len(docs.load()) == 32


def test_metrics_on_clickhouse(store, tmp_path):
    server, client = make(store, tmp_path)
    server.metrics.parsed(5000, unknown=False)
    first = server.sample_metrics()
    assert first["store_mb"] >= 0 and 0 <= first["disk_used_pct"] <= 100
    server.sample_metrics()
    body = client.get("/api/system?minutes=5").json()
    assert body["current"]["threads"] >= 1
    assert len(body["series"]["rss_mb"]) >= 1 and body["series"]["lines_per_s"][0][0] > 1.7e9
