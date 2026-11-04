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


def test_server_on_clickhouse(store, tmp_path):
    server, client = make(store, tmp_path)
    token = client.post("/api/keys", json={"name": "ci"}).json()["token"]
    headers = {"X-API-Key": token}
    lines = [f"job {i} finished in {i % 50} ms on worker {i % 4}" for i in range(60)]
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

    # A restart reads keys, sources and streams back from control_docs.
    server2, client2 = make(store, tmp_path)
    assert client2.get("/intake/v1/ping", headers=headers).status_code == 200
    assert [s["id"] for s in client2.get("/api/sources").json()["sources"]] == ["h1|jobs|go"]
    assert server2.sources["h1|jobs|go"]["count"] == 60
