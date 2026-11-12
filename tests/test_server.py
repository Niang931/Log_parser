"""``logpipe server`` end to end on the memory store: intake, agents, setup, calibration,
per-stream LogBERT, warnings and their actions."""

from __future__ import annotations

import gzip
import json
import os
import random
import threading
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from logpipe.enrich.llm import get_adapter
from logpipe.server.app import create_app
from logpipe.server.control import ControlStore
from logpipe.server.core import Server, remote_hash
from logpipe.storage.memory import MemoryStore

T0 = datetime.now(UTC) - timedelta(hours=2)


def app_lines(n: int, seed: int = 1) -> list[str]:
    rng = random.Random(seed)
    shapes = [
        lambda i: f"user {rng.randint(1, 99)} logged in from 10.0.0.{rng.randint(1, 250)}",
        lambda i: f"request GET /api/items/{rng.randint(1, 999)} took {rng.randint(1, 900)} ms",
        lambda i: f"cache hit ratio={rng.random():.2f} size={rng.randint(10, 99)}",
        lambda i: f"worker {rng.randint(1, 8)} finished job {rng.randint(1000, 9999)}",
    ]
    return [shapes[i % len(shapes)](i) for i in range(n)]


def events(lines, host="web-1", service="api", source="python", start=0):
    return [
        {
            "ts": (T0 + timedelta(seconds=start + i)).isoformat(),
            "host": host,
            "service": service,
            "source": source,
            "tags": ["env:test"],
            "message": m,
        }
        for i, m in enumerate(lines)
    ]


@pytest.fixture
def env(tmp_path):
    # opt in: `STORE=mongodb pytest tests/test_server.py`
    mongo = os.environ.get("STORE") == "mongodb"
    if mongo:
        from logpipe.storage.mongo import MongoDocs, MongoStore

        store = MongoStore.from_env(database=f"logpipe_test_{uuid.uuid4().hex[:8]}")
        control = ControlStore(MongoDocs(store.db))
    else:
        store, control = MemoryStore(), ControlStore()
    # one epoch: fast, the model's quality isn't tested here
    server = Server(
        store,
        control,
        data_dir=tmp_path,
        adapter_factory=lambda: get_adapter("offline"),
        epochs=1,
        start_threads=False,
        min_train_steps=0,
    )
    client = TestClient(create_app(server))
    key = client.post("/api/keys", json={"name": "test"}).json()["token"]
    yield server, client, {"X-API-Key": key}
    if mongo:
        store.db.client.drop_database(store.db.name)
        store.close()


def pump(server: Server) -> None:
    while not server.inbox.empty():
        server.process(server.inbox.get_nowait())


def send(server, client, headers, evs, gz=False):
    body = json.dumps({"events": evs}).encode()
    h = dict(headers)
    if gz:
        body, h["Content-Encoding"] = gzip.compress(body), "gzip"
    r = client.post("/intake/v1/logs", content=body, headers=h)
    assert r.status_code == 202, r.text
    pump(server)


def run_jobs(server):
    while not server.jobs.empty():
        job = server.run_job(server.jobs.get_nowait())
        assert job["status"] == "done", job


def test_intake_requires_a_valid_key(env):
    _, client, headers = env
    assert client.post("/intake/v1/logs", json={"events": []}).status_code == 401
    assert client.get("/intake/v1/ping", headers={"X-API-Key": "lp_wrong"}).status_code == 401
    assert client.get("/intake/v1/ping", headers=headers).json()["ok"]
    key_id = client.get("/api/keys").json()["keys"][0]["id"]
    assert "hash" not in client.get("/api/keys").json()["keys"][0]
    client.delete(f"/api/keys/{key_id}")
    assert client.get("/intake/v1/ping", headers=headers).status_code == 401


def test_intake_rejects_bad_bodies_and_backpressures(env):
    server, client, headers = env
    assert server.inbox.empty()
    assert client.post("/intake/v1/logs", content=b"nope", headers=headers).status_code == 400
    server.inbox.maxsize = 2
    r = client.post("/intake/v1/logs", json={"events": events(["a", "b", "c"])}, headers=headers)
    assert r.status_code == 429


def test_heartbeat_and_remote_config(env):
    _, client, headers = env
    status = {"hostname": "web-1", "version": "1.0", "sources": [], "remote_hash": ""}
    assert client.post("/intake/v1/heartbeat", json=status, headers=headers).json() == {
        "remote_hash": ""
    }
    assert (
        client.put(
            "/api/agents/web-1/remote-config", json={"yaml": "logs: [{type: nope}]"}
        ).status_code
        == 400
    )
    text = "logs:\n  - type: file\n    path: /var/log/x.log\n"
    assert client.put("/api/agents/web-1/remote-config", json={"yaml": text}).status_code == 200
    answer = client.post("/intake/v1/heartbeat", json=status, headers=headers).json()
    assert answer["remote_hash"] == remote_hash(text)
    assert (
        client.get("/intake/v1/config", params={"hostname": "web-1"}, headers=headers).json()[
            "yaml"
        ]
        == text
    )
    agent = client.get("/api/agents/web-1").json()
    assert agent["online"] and not agent["remote_applied"]


def test_full_lifecycle(env):
    server, client, headers = env
    # 1. before calibration: stored PENDING, nothing learned
    send(server, client, headers, events(app_lines(300)), gz=True)
    send(server, client, headers, events(app_lines(200, seed=2), host="web-2"))
    logs = client.get("/api/logs", params={"host": "web-1", "limit": 5}).json()["logs"]
    assert len(logs) == 5 and logs[0]["status"] == "PENDING" and logs[0]["service"] == "api"
    setup = client.get("/api/setup").json()
    assert len(setup["sources"]) == 2 and not setup["settings"]["calibrated"]

    # 2. drafts -> review (edit, merge, decide) -> apply: templates, backfill, a model
    assert client.post("/api/review/generate", json={"sample_cap": 150}).json()["kind"] == "draft"
    run_jobs(server)
    assert client.get("/api/overview").json()["templates"] == 0  # nothing live before review
    detail = client.get("/api/jobs").json()["jobs"][0]["detail"]  # the UI's detail panel
    assert detail["Input"]["sampled lines"] > 0 and detail["Output"]["templates"]
    assert {c["call"] for c in detail["LLM calls"]} >= {"synth_masks", "name_keys"}
    batch = client.get("/api/setup").json()["review"]
    assert batch["initial"] and batch["pending"] == batch["drafts"] >= 4
    drafts = client.get(f"/api/review/{batch['id']}").json()["drafts"]
    assert client.post(f"/api/review/{batch['id']}/apply").status_code == 400  # undecided drafts
    login = next(d for d in drafts if d["tokens"][:2] == ["user", "<VAR:NUMBER>"])
    # counted over each source's 150 most recent lines (a quarter are logins), not the 4-line sample
    assert login["count"] == 74 and len(login["samples"]) == 5
    edited = client.put(f"/api/drafts/{login['id']}", json={"keys": ["user_id", "ip"]}).json()
    assert edited["keys"] == ["user_id", "ip"] and edited["count"] == login["count"]
    assert client.put(f"/api/drafts/{login['id']}", json={"keys": ["x"]}).status_code == 400
    assert (
        client.post(
            f"/api/review/{batch['id']}/check",
            json={"tokens": ["user", "<*>", "logged", "out"], "keys": ["u"]},
        ).json()["count"]
        == 0
    )
    assert (
        client.post(
            f"/api/review/{batch['id']}/check",
            json={  # widened by hand: still matches
                "tokens": ["user", "<*>", "logged", "in", "from", "<*>"],
                "keys": ["u", "ip"],
            },
        ).json()["count"]
        == 74
    )
    picked = [d["id"] for d in drafts[:2]]  # ticked in the UI: just those, and back again
    url = f"/api/review/{batch['id']}/decide-all"
    assert client.post(url, json={"status": "rejected", "ids": picked}).json()["changed"] == 2
    assert client.post(url, json={"status": "pending", "ids": picked}).json()["changed"] == 2
    # "undecided" -> undecided: no
    assert client.post(url, json={"status": "pending"}).status_code == 400
    assert (
        client.post(f"/api/review/{batch['id']}/decide-all", json={"status": "approved"}).json()[
            "changed"
        ]
        > 0
    )
    assert client.post(f"/api/review/{batch['id']}/apply").json()["kind"] == "apply"
    run_jobs(server)
    assert client.post("/api/review/generate", json={}).status_code == 400  # only once
    overview = client.get("/api/overview").json()
    assert overview["settings"]["calibrated"] and overview["templates"] >= 4
    assert overview["status"].get("FINAL", 0) >= 450 and overview["models"] == 1
    streams = client.get("/api/streams").json()["streams"]
    # web-2 joined web-1's stream
    assert [s["id"] for s in streams] == ["api"] and streams[0]["has_model"]
    trained = next(j for j in client.get("/api/jobs").json()["jobs"] if j["kind"] == "apply")[
        "detail"
    ]
    assert trained["api: input"]["sources"] == 2 and trained["api: output"]["model version"]
    tpl = client.get("/api/templates").json()["templates"]
    assert any(t["keys"] == ["user_id", "ip"] for t in tpl)  # the reviewer's field names
    assert all(t["seen_by"] == ["api"] for t in tpl if t["count"])
    assert sum(t["count"] for t in tpl) == overview["status"]["FINAL"]  # back-filled lines counted

    # 3. a new source after calibration -> warning; "same as" resolves it
    send(server, client, headers, events(app_lines(20, seed=3), host="web-3"))
    warns = {w["id"]: w for w in client.get("/api/warnings").json()["warnings"]}
    wid = "new_source:web-3|api|python"
    assert warns[wid]["status"] == "open" and warns[wid]["count"] == 20
    assert (
        client.post(f"/api/warnings/{wid}/same-as", json={"target": "web-1|api|python"}).json()[
            "status"
        ]
        == "resolved"
    )
    assert client.get("/api/overview").json()["warnings"] == {}

    # 4. a different app -> drafted on its own at the threshold -> reviewed -> appended, ids kept
    before = {t["template_id"]: t["text"] for t in client.get("/api/templates").json()["templates"]}
    server.update_settings({"draft_lines": 100})
    other = [f"disk /dev/sd{'abc'[i % 3]} usage {i % 97} percent" for i in range(120)]
    send(server, client, headers, events(other[:60], host="db-1", service="disk", source="agent"))
    server.pipeline.enrich()  # the online learner must not learn a new source before its review
    assert not any(t["tokens"][0] == "disk" for t in server.auto_templates())
    assert server.due_drafts() == []
    assert client.get("/api/maintenance").json()["collecting"][0]["have"] == 60
    send(server, client, headers, events(other[60:], host="db-1", service="disk", source="agent"))
    wid = "new_source:db-1|disk|agent"
    assert server.due_drafts() == ["db-1|disk|agent"]
    server.enqueue("draft", sources=server.due_drafts())
    run_jobs(server)
    assert server.due_drafts() == []  # in review now
    batch = client.get("/api/maintenance").json()["review"][0]
    assert not batch["initial"] and batch["sources"] == ["db-1|disk|agent"] and batch["drafts"] >= 1
    drafts = client.get(f"/api/review/{batch['id']}").json()["drafts"]
    if len(drafts) > 1:  # merge whatever Drain split
        same = [d["id"] for d in drafts if len(d["tokens"]) == len(drafts[0]["tokens"])]
        if len(same) > 1:
            merged = client.post(f"/api/drafts/{same[0]}/merge", json={"others": same[1:]}).json()
            assert merged["status"] == "approved"
    client.post(f"/api/review/{batch['id']}/decide-all", json={"status": "approved"})
    client.post(f"/api/review/{batch['id']}/apply")
    run_jobs(server)
    after = {t["template_id"]: t["text"] for t in client.get("/api/templates").json()["templates"]}
    assert all(after[k] == v for k, v in before.items()) and len(after) > len(before)
    assert client.get("/api/logs?host=db-1&limit=1").json()["logs"][0]["status"] == "FINAL"
    warns = {w["id"]: w for w in client.get("/api/warnings").json()["warnings"]}
    assert warns[wid]["status"] == "resolved"

    # 5. online learning finds a new template -> novelty warning (not an alert) -> retrain clears it
    novel = [f"payment {i} refused by bank code {i % 7}" for i in range(40)]
    send(server, client, headers, events(novel, start=1000))
    server.pipeline.enrich()
    send(server, client, headers, events(novel, start=2000))
    novel_warns = [
        w
        for w in client.get("/api/warnings").json()["warnings"]
        if w["kind"] == "novel_template" and w["status"] == "open"
    ]
    assert novel_warns and novel_warns[0]["streams"] == ["api"]
    assert all("novelty" not in a["reasons"] for a in client.get("/api/alerts").json()["alerts"])
    # due only once the stream's retrain period has passed
    assert server.due_retrains() == []
    server.control.update(
        "stream", "api", trained_at=(datetime.now(UTC) - timedelta(days=8)).isoformat()
    )
    assert server.due_retrains() == ["api"]
    client.post("/api/streams/api/train")
    run_jobs(server)
    assert not [
        w
        for w in client.get("/api/warnings").json()["warnings"]
        if w["kind"] == "novel_template" and w["status"] == "open"
    ]


def test_each_source_is_in_one_stream_ticked_sources_win(env):
    server, client, headers = env
    send(server, client, headers, events(app_lines(30)))
    # same service and type: joins
    send(server, client, headers, events(app_lines(5), host="web-2"))
    # same name, other type
    send(server, client, headers, events(app_lines(5), host="web-3", source="go"))
    streams = {x["id"]: x for x in client.get("/api/streams").json()["streams"]}
    assert streams["api"]["sources"] == ["web-1|api|python", "web-2|api|python"]
    assert streams["api-go"]["name"] == "api (go)" and streams["api-go"]["sources"] == [
        "web-3|api|go"
    ]
    client.post(
        "/api/streams", json={"name": "api", "picked": ["web-3|api|go", "web-2|api|python"]}
    )
    streams = {x["id"]: x for x in client.get("/api/streams").json()["streams"]}
    assert "api-go" not in streams and streams["api-2"]["sources"] == [
        "web-2|api|python",
        "web-3|api|go",
    ]
    client.put("/api/streams/api-2", json={"name": "api", "picked": ["web-3|api|go"]})
    streams = {x["id"]: x for x in client.get("/api/streams").json()["streams"]}
    # unticked: back with a source like it (web-1), never left without a stream
    assert streams["api"]["sources"] == ["web-1|api|python", "web-2|api|python"]
    client.post("/api/streams", json={"name": "Web tier", "selectors": [{"host": "web-*"}]})
    send(server, client, headers, events(app_lines(5), host="web-9", service="cron", source="go"))
    streams = {x["id"]: x for x in client.get("/api/streams").json()["streams"]}
    # a pattern catches sources that come later
    assert streams["web-tier"]["sources"] == ["web-9|cron|go"]
    send(
        server,
        client,
        headers,
        events(["db up"] * 5, host="db-1", service="pg", source="postgresql"),
    )
    db = "db-1|pg|postgresql"
    # ticked: wins
    client.post("/api/streams", json={"name": "Data", "picked": [db, "web-1|api|python"]})
    streams = {x["id"]: x for x in client.get("/api/streams").json()["streams"]}
    assert sorted(streams["data"]["sources"]) == [db, "web-1|api|python"] and "pg" not in streams
    # moves back
    client.put(
        "/api/streams/web-tier",
        json={"name": "Web tier", "picked": ["web-1|api|python"], "selectors": [{"host": "web-*"}]},
    )
    streams = {x["id"]: x for x in client.get("/api/streams").json()["streams"]}
    assert streams["data"]["picked"] == [db] and streams["web-tier"]["sources"] == [
        "web-1|api|python",
        "web-9|cron|go",
    ]
    assert client.post("/api/streams", json={"name": "X", "picked": ["nope"]}).status_code == 400
    # pattern gone
    client.put("/api/streams/web-tier", json={"name": "Web tier", "picked": ["web-1|api|python"]})
    streams = {x["id"]: x for x in client.get("/api/streams").json()["streams"]}
    assert streams["cron"]["sources"] == ["web-9|cron|go"]  # web-9 got a stream of its own


@pytest.mark.skipif(os.environ.get("STORE") == "mongodb", reason="raw SQL: not on MongoDB")
def test_streams_settings_query_stats(env):
    server, client, headers = env
    send(server, client, headers, events(app_lines(30)))
    s = client.post(
        "/api/streams",
        json={"name": "Web tier", "selectors": [{"host": "web-*"}], "retrain_days": 3},
    ).json()
    assert s["id"] == "web-tier"
    streams = {x["id"]: x for x in client.get("/api/streams").json()["streams"]}
    assert streams["web-tier"]["sources"] == [] and streams["api"]["sources"] == [
        "web-1|api|python"
    ]
    assert client.delete("/api/streams/api").status_code == 400  # still has a source
    assert client.delete("/api/streams/web-tier").json() == {"ok": True}
    assert client.put("/api/settings", json={"sample_cap": 5}).status_code == 400
    assert client.put("/api/settings", json={"retrain_days": 14}).json()["retrain_days"] == 14
    about = client.get("/api/about").json()
    assert about["engine"] == "sqlite" and about["version"] and about["password_set"] is False
    q = client.post(
        "/api/query", json={"sql": "SELECT device_id, count(*) FROM logs GROUP BY 1"}
    ).json()
    assert q["rows"] == [["web-1", 30]]
    assert client.post("/api/query", json={"sql": "DELETE FROM logs"}).status_code == 400
    stats = client.get("/api/stats", params={"minutes": 24 * 60}).json()
    assert stats["hosts"] == [["web-1", 30]]
    hist = client.get("/api/logs/histogram", params={"minutes": 24 * 60, "by": "status"}).json()
    assert hist["step"] == 1800 and len(hist["points"]) == 48
    assert sum(p["FINAL"] + p["PENDING"] for p in hist["points"]) == 30
    facets = client.get("/api/logs/facets", params={"minutes": 24 * 60, "host": "web-1"}).json()
    assert facets["host"] == [["web-1", 30]] and sum(n for _, n in facets["level"]) == 30
    assert all(
        r["level"] for r in client.get("/api/logs", params={"minutes": 24 * 60}).json()["logs"]
    )
    assert client.get("/demo/").status_code == 404
    assert client.get("/api/nope").status_code == 404


def test_zone_less_line_times_use_the_agents_offset(env):
    server, client, headers = env
    send(
        server,
        client,
        headers,
        [
            {
                "ts": "2026-01-01T10:00:05+08:00",
                "host": "h",
                "service": "s",
                "source": "x",
                "message": "2026-01-01 10:00:00 started",
            },
            {
                "ts": "2026-01-01T10:00:06+08:00",
                "host": "h",
                "service": "s",
                "source": "x",
                "message": "no time here",
            },
        ],
    )
    got = {r["message"]: r["timestamp"] for r in client.get("/api/logs").json()["logs"]}
    assert got["2026-01-01 10:00:00 started"] == "2026-01-01T02:00:00+00:00"
    assert got["no time here"] == "2026-01-01T02:00:06+00:00"


def test_wait_long_poll_and_instant_presence(env, monkeypatch):
    import time

    server, client, headers = env
    client.post("/intake/v1/heartbeat", json={"hostname": "gw-9"}, headers=headers)
    server.set_remote("gw-9", "logs:\n  - type: udp\n    port: 5514\n")
    want = server.remote_hash_of("gw-9")
    # A config the agent doesn't have answers at once; the current one is held until the timeout.
    start = time.monotonic()
    assert client.get("/intake/v1/wait", params={"hostname": "gw-9"}, headers=headers).json() == {
        "remote_hash": want
    }
    assert time.monotonic() - start < 1
    held = client.get(
        "/intake/v1/wait",
        params={"hostname": "gw-9", "remote_hash": want, "timeout": 1.2},
        headers=headers,
    )
    assert held.json() == {"remote_hash": want} and time.monotonic() - start >= 1.2
    # A job or the online learner holding the server lock (an LLM call, training) must not stall
    # the event loop that serves every request.
    answered = threading.Event()
    with server.lock:
        threading.Thread(
            target=lambda: (
                client.get(
                    "/intake/v1/wait", headers=headers, params={"hostname": "gw-9", "timeout": 0}
                ),
                answered.set(),
            ),
            daemon=True,
        ).start()
        assert answered.wait(2)

    # Held open -> online; closed and not reopened -> offline without waiting for ONLINE_SECONDS.
    server.connected("gw-9", +1)
    assert server.agents()[0]["online"]
    server.connected("gw-9", -1)
    assert server.agents()[0]["online"]  # within the reconnect grace
    cfg = server.config
    monkeypatch.setattr(
        server,
        "config",
        cfg.model_copy(update={"server": cfg.server.model_copy(update={"gone_seconds": 0})}),
    )
    time.sleep(0.01)
    assert not server.agents()[0]["online"]


def test_tracebacks_go_to_the_log_file_not_the_console(tmp_path, capsys):
    import logging

    from logpipe.server import setup_logging

    log_file = setup_logging(tmp_path)
    try:
        raise ValueError("stream All logs: 1 windows")
    except ValueError:
        logging.getLogger("logpipe.server").exception("job j1 failed")
    logging.getLogger("logpipe.server").info("quiet")
    for h in logging.getLogger().handlers:
        h.flush()
    err = capsys.readouterr().err
    assert (
        "job j1 failed: stream All logs: 1 windows" in err
        and "Traceback" not in err
        and "quiet" not in err
    )
    assert "Traceback" in log_file.read_text() and "quiet" in log_file.read_text()
    logging.basicConfig(force=True)  # leave pytest's logging as it was


def test_level_rules_agree_in_python_and_explicit_levels_win():
    from logpipe.server.stores import level

    assert level("2026-11-05 10:00:00 WARN  [db] slow query") == "warn"
    assert level("INFO request failed java.lang.IllegalStateException") == "info"  # declared level
    assert level('10.0.0.1 - - [x] "GET /a HTTP/1.1" 503 12') == "error"
    assert level('10.0.0.1 - - [x] "GET /a HTTP/1.1" 404 12') == "warn"
    assert level("ERR_CONNECTION refused") == "info" and level("no level here") == "info"


def calibrated(server, client, headers, lines):
    send(server, client, headers, events(lines))
    client.post("/api/review/generate", json={"sample_cap": 150})
    run_jobs(server)
    bid = client.get("/api/setup").json()["review"]["id"]
    client.post(f"/api/review/{bid}/decide-all", json={"status": "approved"})
    client.post(f"/api/review/{bid}/apply")
    run_jobs(server)


def test_too_little_data_waits_instead_of_failing(env):
    server, client, headers = env
    send(server, client, headers, events(app_lines(10)))
    assert client.post("/api/review/generate", json={}).status_code == 400  # under 50 lines
    calibrated(server, client, headers, app_lines(40, seed=5))  # 50 lines: 7 windows of 20, need 8
    stream = client.get("/api/streams").json()["streams"][0]
    jobs = client.get("/api/jobs").json()["jobs"]
    assert all(j["status"] == "done" for j in jobs)  # nothing failed
    assert stream["waiting"] and stream["data"]["have"] < stream["data"]["need"] == 8
    assert not stream["has_model"]
    assert client.post("/api/streams/api/train").json() == {"waiting": ["api"]}
    assert server.jobs.empty()
    # more logs arrive -> housekeeping notices and trains by itself
    send(server, client, headers, events(app_lines(200, seed=6), start=500))
    server.refresh_readiness()
    run_jobs(server)
    stream = client.get("/api/streams").json()["streams"][0]
    assert stream["has_model"] and not stream["waiting"]
    assert client.get("/api/maintenance").json()["waiting"] == []


def test_jobs_cancel_retry_and_log(env):
    server, client, _ = env
    job = server.enqueue("retrain", streams=["missing"])
    assert client.post(f"/api/jobs/{job['id']}/retry").status_code == 400  # still queued
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["status"] == "cancelled"
    assert server.run_job(server.jobs.get_nowait())["status"] == "cancelled"  # skipped
    again = client.post(f"/api/jobs/{job['id']}/retry").json()
    assert again["id"] != job["id"] and again["status"] == "queued"
    failed = server.run_job(server.jobs.get_nowait())
    assert failed["status"] == "failed" and failed["log"] == []
    assert client.post(f"/api/jobs/{again['id']}/cancel").status_code == 400  # not queued


def test_online_templates_are_listed_for_review(env):
    server, client, headers = env
    calibrated(server, client, headers, app_lines(120))
    assert client.get("/api/review").json()["auto_templates"] == []  # reviewed ones aren't listed
    novel = [f"payment {i} refused by bank code {i % 7}" for i in range(40)]
    send(server, client, headers, events(novel, start=1000))
    server.pipeline.enrich()
    auto = client.get("/api/review").json()["auto_templates"]
    assert auto and client.get("/api/maintenance").json()["auto_templates"] == len(auto)
    tid = auto[0]["template_id"]
    slots = sum(t.startswith("<") for t in auto[0]["tokens"])
    keys = [f"f{i}" for i in range(slots)]
    assert (
        client.post(f"/api/templates/{tid}/review", json={"keys": ["Bad Name"] * slots}).status_code
        == 400
    )
    assert client.post(f"/api/templates/{tid}/review", json={"keys": keys}).json()["reviewed"]
    entry = server.registry.current.templates[tid]
    assert list(entry.keys) == keys and tid not in [
        t["template_id"] for t in client.get("/api/review").json()["auto_templates"]
    ]


def test_lines_that_parse_keep_flowing_while_the_llm_thinks(env):
    server, client, headers = env
    calibrated(server, client, headers, app_lines(120))
    worker = server.pipeline.worker
    thinking, answer = threading.Event(), threading.Event()
    name_keys = worker.adapter.name_keys

    def slow_name_keys(tokens, samples):  # a chat model taking its time
        thinking.set()
        answer.wait(10)
        return name_keys(tokens, samples)

    worker.adapter.name_keys = slow_name_keys
    send(
        server,
        client,
        headers,
        events([f"payment {i} refused by bank code {i % 7}" for i in range(40)], start=1000),
    )
    cycle = threading.Thread(target=server.pipeline.enrich)  # as housekeeping runs it
    cycle.start()
    assert thinking.wait(5)
    parsed = threading.Thread(
        target=send, args=(server, client, headers, events(app_lines(20), start=2000))
    )
    parsed.start()
    parsed.join(2)
    alive = parsed.is_alive()
    answer.set()
    cycle.join(10)
    assert not alive, "a line that parses waited for the LLM"
    assert client.get("/api/review").json()["auto_templates"]  # and the LLM's template still landed


def test_parsing_rules_on_drafts_and_live_templates(env):
    server, client, headers = env
    send(server, client, headers, events(app_lines(200)))
    client.post("/api/review/generate", json={"sample_cap": 150})
    run_jobs(server)
    bid = client.get("/api/setup").json()["review"]["id"]
    drafts = client.get(f"/api/review/{bid}").json()["drafts"]
    login = next(d for d in drafts if d["tokens"][0] == "user")
    # a starting rule
    assert login["rule"].startswith("user %{number:") and not login["rule_edited"]

    rule = "user %{integer:user_id} logged in from %{ip:client_ip}"
    checked = client.post(f"/api/drafts/{login['id']}/check", json={"rule": rule}).json()
    assert checked["count"] == login["count"] > 30 and checked["keys"] == ["user_id", "client_ip"]
    assert checked["rows"][0]["fields"]["client_ip"].startswith("10.0.0.")
    narrow = client.post(
        f"/api/drafts/{login['id']}/check",
        json={"rule": "user %{integer:u} logged in from 10.0.0.1"},
    ).json()
    # the draft's lines it misses
    assert narrow["missed"] > 0 and narrow["rows"][-1]["fields"] is None
    assert (
        client.post(f"/api/drafts/{login['id']}/check", json={"rule": "%{data:x}"}).status_code
        == 400
    )
    saved = client.put(f"/api/drafts/{login['id']}", json={"rule": rule}).json()
    assert saved["rule_edited"] and saved["keys"] == ["user_id", "client_ip"]
    client.post(f"/api/review/{bid}/decide-all", json={"status": "approved"})
    client.post(f"/api/review/{bid}/apply")
    run_jobs(server)
    row = client.get("/api/logs", params={"q": "logged in", "limit": 1}).json()["logs"][0]
    assert set(row["payload"]) == {"user_id", "client_ip"}  # back-filled with the rule

    # a live template: new rule -> stored lines re-extracted by a job
    worker = next(
        t
        for t in client.get("/api/templates").json()["templates"]
        if t["text"].startswith("worker")
    )
    assert not worker["rule_edited"]
    rule = "worker %{integer:worker_id} finished job %{integer:job_id}"
    assert (
        client.post(f"/api/templates/{worker['template_id']}/check", json={"rule": rule}).json()[
            "count"
        ]
        == 50
    )
    job = client.post(f"/api/templates/{worker['template_id']}/review", json={"rule": rule}).json()[
        "job"
    ]
    assert job["kind"] == "reextract"
    run_jobs(server)
    row = client.get("/api/logs", params={"q": "finished job", "limit": 1}).json()["logs"][0]
    assert set(row["payload"]) == {"worker_id", "job_id"}
    send(server, client, headers, events(["worker 3 finished job 77"], start=900))
    row = client.get("/api/logs", params={"q": "job 77", "limit": 1}).json()["logs"][0]
    assert row["payload"] == {"worker_id": "3", "job_id": "77"}
    assert server.registry.current.templates[worker["template_id"]].rule == rule


def test_template_links_show_who_made_them_and_can_be_undone(env):
    server, client, headers = env
    calibrated(server, client, headers, app_lines(120))
    tpl = {t["text"].split()[0]: t for t in client.get("/api/templates").json()["templates"]}
    a, b = tpl["worker"]["template_id"], tpl["cache"]["template_id"]
    assert client.post(f"/api/templates/{a}/link", json={"target": b}).json()["alias_of"] == b
    tpl = {t["template_id"]: t for t in client.get("/api/templates").json()["templates"]}
    assert tpl[a]["linked_by"] == "you" and tpl[a]["canonical"] == b and tpl[b]["linked"] == [a]
    send(server, client, headers, events(["worker 1 finished job 5"], start=900))
    assert (
        client.get("/api/logs", params={"q": "job 5", "limit": 1}).json()["logs"][0]["template_id"]
        == b
    )
    assert client.post(f"/api/templates/{a}/unlink").json()["alias_of"] is None
    tpl = {t["template_id"]: t for t in client.get("/api/templates").json()["templates"]}
    assert tpl[a]["linked_by"] is None and tpl[b]["linked"] == []
    assert client.post(f"/api/templates/{a}/unlink").status_code == 400
    snap = server.registry.alias(a, b)  # as the enrichment worker's LLM alias check does it
    server.pipeline.handle.swap(snap)
    assert {t["template_id"]: t for t in client.get("/api/templates").json()["templates"]}[a][
        "linked_by"
    ] == "llm"


@pytest.mark.skipif(
    os.environ.get("STORE") == "mongodb",
    reason="SQL query builder: MongoDB pipeline builder is Phase 6",
)
def test_query_builder_schema_saved_and_ask(env):
    server, client, headers = env
    now = datetime.now(UTC)
    lines = [f"request GET /api/items/{i} took {10 * (i % 5)} ms" for i in range(60)] + [
        "ERROR db down"
    ] * 3
    send(
        server,
        client,
        headers,
        [e | {"ts": (now - timedelta(minutes=5)).isoformat()} for e in events(lines)],
    )
    send(
        server,
        client,
        headers,
        [
            e | {"ts": (now - timedelta(minutes=5)).isoformat()}
            for e in events(["ERROR disk full"], service="db")
        ],
    )
    calibrated(server, client, headers, [])
    built = client.post("/api/query/build", json={"minutes": 60, "group_by": ["service"]}).json()
    assert dict(built["result"]["rows"]) == {"api": 63, "db": 1}
    errors = client.post(
        "/api/query/build",
        json={"minutes": 60, "filters": [{"field": "level", "op": "=", "value": "error"}]},
    ).json()
    assert errors["result"]["rows"] == [[4]]
    took = next(t for t in client.get("/api/templates").json()["templates"] if "took" in t["text"])
    assert set(took["keys"]) <= set(client.get("/api/query/schema").json()["fields"])
    avg = client.post(
        "/api/query/build",
        json={
            "minutes": 60,
            "measure": {"fn": "avg", "field": f"@{took['keys'][-1]}"},
            "filters": [{"field": "message", "op": "contains", "value": "took"}],
        },
    ).json()
    assert avg["result"]["rows"] == [[20.0]]  # 0,10,20,30,40 ms evenly
    over = client.post("/api/query/build", json={"minutes": 60, "over_time": True}).json()
    assert (
        sum(r[-1] for r in over["result"]["rows"]) == 64 and over["result"]["columns"][0] == "time"
    )
    for bad in (
        {"group_by": ["host; DROP TABLE logs"]},
        {"filters": [{"field": "@x'y", "value": "1"}]},
        {"measure": {"fn": "evil"}},
        {"filters": [{"field": "host", "op": ">", "value": "abc"}]},
    ):
        assert client.post("/api/query/build", json=bad).status_code == 400, bad
    quoted = client.post(
        "/api/query/build", json={"filters": [{"field": "host", "value": "x' OR '1'='1"}]}
    ).json()
    assert quoted["result"]["rows"] == [[0]]  # the value stays a value

    saved = client.post("/api/queries", json={"name": "errors", "sql": errors["sql"]}).json()
    assert [q["name"] for q in client.get("/api/queries").json()["queries"]] == ["errors"]
    assert client.delete(f"/api/queries/{saved['id']}").json()["ok"]
    # now /api/ask
    assert client.post("/api/query/ask", json={"question": "errors per service"}).status_code in (
        404,
        405,
    )


def test_a_zone_written_in_the_line_wins_over_the_agents_offset():
    from datetime import timezone

    from logpipe.sources.timestamps import event_time

    host = timezone(timedelta(hours=8))
    utc = {
        line: event_time(line, {}, local=host).astimezone(UTC).strftime("%H:%M")
        for line in (
            "2026-11-05 09:20:32.599 UTC [24] LOG: x",
            "2026-11-05 17:20:32 +08 [2] LOG: x",
            "2026-11-05T17:20:32+08:00 x",
            "2026-11-05 17:20:32.599 INFO x",
            "time=2026-11-05T09:20:32Z level=info",
            "2026-11-05 17:20:32 -5 ms late",
        )
    }
    assert set(utc.values()) == {"09:20"}, utc
    # a rule's extracted `ts` without the zone that follows it in the line
    line = "2026-11-05 09:20:32.599 UTC [24] LOG: x"
    assert (
        event_time(line, {"ts": "2026-11-05 09:20:32.599"}, local=host)
        .astimezone(UTC)
        .strftime("%H:%M")
        == "09:20"
    )


def test_otlp_logs_json_and_protobuf(env):
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
    from opentelemetry.proto.logs.v1.logs_pb2 import (
        LogRecord,
        ResourceLogs,
        ScopeLogs,
        SeverityNumber,
    )
    from opentelemetry.proto.resource.v1.resource_pb2 import Resource

    server, client, headers = env
    t = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    s = lambda k, v: {"key": k, "value": {"stringValue": v}}
    # As logpipe-agent sends it: OTLP/HTTP JSON, gzip, observed time, the host's offset.
    agent = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        s("host.name", "gw-1"),
                        s("service.name", "api"),
                        s("logpipe.source", "java"),
                        s("logpipe.host.utc_offset", "+08:00"),
                    ]
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "logpipe-agent"},
                        "logRecords": [
                            {
                                "observedTimeUnixNano": str(int(t.timestamp() * 1e9)),
                                "body": {"stringValue": "user 7 logged in"},
                                "attributes": [
                                    s("log.file.path", "/var/log/app.log"),
                                    {
                                        "key": "logpipe.tags",
                                        "value": {
                                            "arrayValue": {"values": [{"stringValue": "env:prod"}]}
                                        },
                                    },
                                ],
                            },
                            {
                                "observedTimeUnixNano": str(int(t.timestamp() * 1e9)),
                                "body": {"stringValue": "2026-01-02 11:04:06 job done"},
                            },
                        ],
                    }
                ],
            }
        ]
    }
    r = client.post(
        "/v1/logs",
        content=gzip.compress(json.dumps(agent).encode()),
        headers=headers | {"Content-Type": "application/json", "Content-Encoding": "gzip"},
    )
    assert r.status_code == 200 and r.json() == {}
    # As the OpenTelemetry Collector / an SDK sends it: protobuf, a severity, bearer auth.
    req = ExportLogsServiceRequest(
        resource_logs=[
            ResourceLogs(
                resource=Resource(
                    attributes=[
                        KeyValue(key="service.name", value=AnyValue(string_value="checkout")),
                        KeyValue(key="host.name", value=AnyValue(string_value="pod-3")),
                    ]
                ),
                scope_logs=[
                    ScopeLogs(
                        log_records=[
                            LogRecord(
                                time_unix_nano=int(t.timestamp() * 1e9),
                                severity_number=SeverityNumber.SEVERITY_NUMBER_ERROR,
                                severity_text="ERROR",
                                body=AnyValue(string_value="payment refused"),
                            )
                        ]
                    )
                ],
            )
        ]
    )
    r = client.post(
        "/v1/logs",
        content=req.SerializeToString(),
        headers={
            "Content-Type": "application/x-protobuf",
            "Authorization": "Bearer " + headers["X-API-Key"],
        },
    )
    assert r.status_code == 200 and r.headers["content-type"] == "application/x-protobuf"
    pump(server)

    logs = {l["message"]: l for l in client.get("/api/logs", params={"limit": 10}).json()["logs"]}
    assert (
        logs["user 7 logged in"]["host"] == "gw-1" and logs["user 7 logged in"]["service"] == "api"
    )
    assert logs["user 7 logged in"]["source"] == "java" and logs["user 7 logged in"]["tags"] == [
        "env:prod"
    ]
    # zone-less time in the line = the agent host's clock (+08:00)
    assert logs["2026-01-02 11:04:06 job done"]["timestamp"].startswith("2026-01-02T03:04:06")
    refused = logs["payment refused"]
    assert (refused["host"], refused["service"], refused["source"]) == ("pod-3", "checkout", "otlp")
    assert refused["level"] == "error"  # from the OTel severity, not the text
    assert client.post("/v1/logs", content=b"nope", headers=headers).status_code == 400
    assert client.post("/v1/logs", json={"resourceLogs": []}).status_code == 401
    # agents from before OTLP keep working
    send(server, client, headers, events(["legacy line"]))
    assert client.get("/api/logs", params={"q": "legacy"}).json()["logs"]


def test_incidents_group_alerts_explain_lines_and_learn_false_positives(env):
    from logpipe.storage.base import AnomalyAlert

    server, client, headers = env
    calibrated(server, client, headers, app_lines(300))
    version = server.scorers["api"].version
    send(
        server,
        client,
        headers,
        events([f"worker {i} finished job {i}" for i in range(30)], start=4000),
    )
    t = T0 + timedelta(seconds=4000)

    def alert(s, e):
        return AnomalyAlert(
            "web-1/api",
            t + timedelta(seconds=s),
            t + timedelta(seconds=e),
            (5,) * 20,
            0.9,
            0.0,
            None,
            ("sequence",),
            False,
            1,
            version,
        )

    # the last: >5 min later
    server.store.insert_alerts([alert(0, 19), alert(5, 24), alert(1000, 1010)])
    incs = client.get("/api/incidents").json()["incidents"]
    assert [i["alerts"] for i in incs] == [1, 2] and incs[1]["status"] == "open"
    iid = incs[1]["id"]
    inc = client.get(f"/api/incidents/{iid}").json()
    assert (
        len(inc["events"]) == 25
        and inc["model_current"]
        and inc["events"][0]["template"].startswith("worker")
    )
    assert {e["status"] for e in inc["events"]} <= {"expected", "unexpected", "unseen"}

    plan = client.post(
        f"/api/incidents/{iid}/explain"
    ).json()  # a root-cause Ask, waiting for approval
    assert plan["status"] == "waiting_approval" and plan["spec"]["anchor_time"].startswith(
        t.isoformat()[:19]
    )
    assert plan["spec"]["services"] == ["api"] and server.jobs.empty()  # nothing ran yet
    assert client.get(f"/api/incidents/{iid}").json()["ask_plan"]["id"] == plan["id"]
    for _ in range(4):  # each round waits for its own approval (a thin first pass widens once)
        client.post(f"/api/ask/{plan['id']}/approve")
        run_jobs(server)
        if client.get(f"/api/ask/{plan['id']}").json()["status"] == "done":
            break
    inc = client.get(f"/api/incidents/{iid}").json()
    assert inc["ask_plan"]["status"] == "done" and inc["explanation"]["plan"] == plan["id"]
    assert inc["explanation"]["kind"] == "report" and inc["explanation"]["anchor"]["kind"] == "user"

    before = len(server._windows("api")[0])
    client.put(f"/api/incidents/{iid}", json={"status": "false_positive", "note": "nightly batch"})
    assert len(server._windows("api")[0]) == before + 2  # the next retrain learns them as normal
    fp = client.get("/api/incidents", params={"status": "false_positive"}).json()["incidents"]
    assert [i["note"] for i in fp] == ["nightly batch"]
    assert client.put(f"/api/incidents/{iid}", json={"status": "nope"}).status_code == 400
    assert client.get("/api/incidents/missing").status_code == 404


def test_firmware_changes_a_variable_is_linked_by_itself_a_rewording_needs_the_llm(env):
    server, client, headers = env
    checkout = [
        f"checkout started for cart {1000 + i} by user {i} total={i}.50 EUR" for i in range(60)
    ]
    calibrated(server, client, headers, [x for pair in zip(app_lines(60), checkout) for x in pair])
    before = {t["template_id"]: t for t in client.get("/api/templates").json()["templates"]}
    old = next(tid for tid, t in before.items() if t["text"].startswith("checkout"))
    # the currency now varies
    v2 = [
        f"checkout started for cart {2000 + i} by user {i} total={i}.75 {('USD', 'GBP')[i % 2]}"
        for i in range(20)
    ]
    v2 += [f"worker {i} completed job {i * 7}" for i in range(20)]  # "finished" reworded
    send(server, client, headers, events(v2, start=5000))
    server.enrich()
    new = [
        t
        for t in client.get("/api/templates").json()["templates"]
        if t["template_id"] not in before
    ]
    currency = next(t for t in new if t["text"].startswith("checkout"))
    reworded = next(t for t in new if "completed" in t["text"])
    assert currency["alias_of"] == old and currency["linked_by"] == "pipeline"
    assert currency["link_reason"] == "differs only in variable slots"
    assert reworded["alias_of"] is None  # offline: rewording is never linked without an LLM
