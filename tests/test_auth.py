"""The admin password: the UI's /api needs a session once a password is set; agents don't."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from logpipe.enrich.llm import get_adapter
from logpipe.server import auth as auth_mod
from logpipe.server.app import create_app
from logpipe.server.auth import Auth, hash_password, verify_password
from logpipe.server.control import ControlStore
from logpipe.server.core import Server
from logpipe.settings import get_settings
from logpipe.storage.memory import MemoryStore


@pytest.fixture
def env(tmp_path):
    control = ControlStore()
    server = Server(
        MemoryStore(),
        control,
        data_dir=tmp_path,
        adapter_factory=lambda: get_adapter("offline"),
        start_threads=False,
    )
    open_client = TestClient(create_app(server))
    # before a password exists
    key = open_client.post("/api/keys", json={"name": "a"}).json()["token"]
    Auth(control).set_password("correct horse")
    return control, TestClient(create_app(server)), {"X-API-Key": key}


def test_hash_round_trip():
    stored = hash_password("s3cret-pass")
    assert verify_password("s3cret-pass", stored)
    assert not verify_password("s3cret-pasS", stored)
    assert not verify_password("x", "garbage")


def test_api_needs_login_but_intake_and_ui_shell_do_not(env):
    _, client, agent = env
    assert client.get("/api/overview").status_code == 401
    assert client.get("/api/docs").status_code == 401
    assert client.get("/api/auth/status").json() == {"required": True, "authenticated": False}
    assert client.get("/intake/v1/ping", headers=agent).status_code == 200
    assert client.get("/").status_code in (200, 503)  # the SPA shell renders the login page
    assert client.post("/api/auth/login", json={"password": "nope"}).status_code == 401
    r = client.post("/api/auth/login", json={"password": "correct horse"})
    assert r.status_code == 200 and "httponly" in r.headers["set-cookie"].lower()
    assert client.get("/api/overview").status_code == 200
    client.post("/api/auth/logout")
    assert client.get("/api/overview").status_code == 401


def test_forged_and_expired_sessions_are_rejected(env):
    control, client, _ = env
    a = Auth(control)
    good = a.login("t", "correct horse")
    assert a.valid(good)
    exp, _, sig = good.partition(".")
    assert not a.valid(f"{int(exp) + 1}.{sig}")
    assert not a.valid(a._sign(control.get("auth", "admin"), int(time.time()) - 1))
    client.cookies.set(auth_mod.COOKIE, "9999999999.deadbeef")
    assert client.get("/api/overview").status_code == 401


def test_wrong_passwords_lock_the_address_out(env):
    _, client, _ = env
    for _ in range(get_settings().server.login_max_failures):
        assert client.post("/api/auth/login", json={"password": "bad"}).status_code == 401
    assert client.post("/api/auth/login", json={"password": "correct horse"}).status_code == 429


def test_changing_the_password_signs_other_sessions_out(env):
    control, client, _ = env
    other = Auth(control).login("elsewhere", "correct horse")
    client.post("/api/auth/login", json={"password": "correct horse"})
    assert (
        client.post("/api/auth/password", json={"current": "wrong", "new": "x" * 9}).status_code
        == 403
    )
    assert (
        client.post(
            "/api/auth/password", json={"current": "correct horse", "new": "short"}
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/auth/password", json={"current": "correct horse", "new": "battery staple"}
        ).status_code
        == 200
    )
    assert client.get("/api/overview").status_code == 200  # this session was renewed
    assert not Auth(control).valid(other)


def test_startup_generates_once_and_env_resets():
    control = ControlStore()
    first = Auth(control).ensure(None)
    assert first and Auth(control).login("t", first)
    assert Auth(control).ensure(None) is None  # kept across restarts
    assert Auth(control).ensure("from-the-env") is None
    assert Auth(control).login("t", "from-the-env") and not Auth(control).login("t", first)
