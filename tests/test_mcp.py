"""The /mcp endpoint and the mcp-scoped API keys that guard it, on the memory store."""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from logpipe.enrich.llm import get_adapter
from logpipe.server.app import create_app
from logpipe.server.auth import Auth
from logpipe.server.control import ControlStore
from logpipe.server.core import KeyScope, Server
from logpipe.storage.memory import MemoryStore


@pytest.fixture
def server(tmp_path):
    return Server(MemoryStore(), ControlStore(), data_dir=tmp_path,
                  adapter_factory=lambda: get_adapter("offline"), start_threads=False)


@pytest.fixture
def client(server):
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


def test_mcp_keys_reach_only_allowlisted_routes(server, client):
    _, token = server.create_key("assistant", KeyScope.MCP)
    mcp = {"Authorization": f"Bearer {token}"}
    Auth(server.control).set_password("correct horse")
    assert client.get("/api/logs").status_code == 401
    assert client.get("/api/logs", headers=mcp).status_code == 200
    assert client.get("/api/review", headers=mcp).status_code == 200
    assert client.post("/api/review/b1/apply", headers=mcp).status_code == 403
    assert client.post("/api/review/b1/discard", headers=mcp).status_code == 403
    assert client.get("/api/keys", headers=mcp).status_code == 403
    assert client.put("/api/settings", headers=mcp, json={}).status_code == 403


def test_mcp_keys_cannot_change_incident_status(client):
    mcp = new_key(client, "mcp")
    resolve = client.put("/api/incidents/i1", headers=mcp, json={"status": "resolved", "note": "x"})
    assert resolve.status_code == 403
    note = client.put("/api/incidents/i1", headers=mcp, json={"status": "open", "note": "x"})
    assert note.status_code == 404  # past the guard; i1 just doesn't exist


def test_mcp_endpoint_needs_an_mcp_key(client):
    assert client.post("/mcp", json={}).status_code == 401
    assert client.post("/mcp", json={}, headers=new_key(client, "intake")).status_code == 401


def rpc(client: TestClient, headers: dict, method: str, params: dict | None = None) -> dict:
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    response = client.post("/mcp", headers=headers | {"Accept": "application/json, text/event-stream"}, json=body)
    assert response.status_code == 200, response.text
    return response.json()["result"]


def call_tool(client: TestClient, headers: dict, tool: str, **arguments) -> dict:
    result = rpc(client, headers, "tools/call", {"name": tool, "arguments": arguments})
    text = result["content"][0]["text"]
    return {"error": text} if result["isError"] else json.loads(text)


def test_tools_answer_from_the_logs(server, client):
    server.ingest([{"ts": datetime.now(UTC).isoformat(), "host": "web-1", "service": "api",
                    "source": "python", "message": f"user {i} logged in"} for i in range(5)])
    while not server.inbox.empty():
        server.process(server.inbox.get_nowait())
    with client:
        mcp = new_key(client, "mcp")
        names = {t["name"] for t in rpc(client, mcp, "tools/list")["tools"]}
        assert {"search_logs", "get_incident", "update_draft", "save_query"} <= names
        assert not names & {"apply_batch", "discard_batch"}

        found = call_tool(client, mcp, "search_logs", where={"service": "api"}, limit=3)
        assert len(found["rows"]) == 3 and found["truncated"] and found["hint"]

        call_tool(client, mcp, "save_query", name="logins", sql="SELECT * FROM logs")
        saved = call_tool(client, mcp, "list_saved_queries")["queries"]
        assert [(q["name"], q["by"]) for q in saved] == [("logins", "mcp-key")]

        assert call_tool(client, mcp, "get_incident", incident_id="nope")["error"].endswith("404: not found: nope")


def test_prompts_are_listed(client):
    with client:
        prompts = {p["name"] for p in rpc(client, new_key(client, "mcp"), "prompts/list")["prompts"]}
    assert prompts == {"investigate_incident", "review_batch"}
