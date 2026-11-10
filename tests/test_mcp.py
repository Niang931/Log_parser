"""The /mcp endpoint and the mcp-scoped API keys that guard it, on the memory store."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from logpipe.enrich.llm import get_adapter
from logpipe.server.app import create_app
from logpipe.server.control import ControlStore
from logpipe.server.core import Server
from logpipe.storage.memory import MemoryStore


@pytest.fixture
def client(tmp_path):
    server = Server(MemoryStore(), ControlStore(), data_dir=tmp_path,
                    adapter_factory=lambda: get_adapter("offline"), start_threads=False)
    return TestClient(create_app(server))


def new_key(client: TestClient, scope: str) -> dict[str, str]:
    token = client.post("/api/keys", json={"name": f"{scope}-key", "scope": scope}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_keys_only_work_for_their_scope(client):
    intake, mcp = new_key(client, "intake"), new_key(client, "mcp")
    assert client.get("/intake/v1/ping", headers=intake).status_code == 200
    assert client.get("/intake/v1/ping", headers=mcp).status_code == 401
    assert {k["scope"] for k in client.get("/api/keys").json()["keys"]} == {"intake", "mcp"}
    assert client.post("/api/keys", json={"name": "x", "scope": "admin"}).status_code == 400
