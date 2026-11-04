import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pytest

from logpipe.demo.server import Demo, make_handler, save_registry


def _demo(**kw) -> Demo:
    return Demo(devices=1, history_hours=0.5, validation_hours=0.25, test_hours=0.5, epochs=1,
                enrich_every=50, **{"log": lambda _: None} | kw)


def test_demo_endpoints_controls_and_timeline():
    demo = _demo()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(demo))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    def get(path: str):
        with urlopen(base + path) as resp:
            return json.loads(resp.read())

    def post(query: str):
        with urlopen(Request(base + "/api/control?" + query, method="POST")) as resp:
            return json.loads(resp.read())

    try:
        with urlopen(base + "/") as resp:
            assert b"<title>" in resp.read()
        system = get("/api/system")
        plug = next(d for d, kinds in system["devices"].items() if "stuck" in kinds)
        post("action=rollout")
        post(f"action=inject&value={plug}:stuck")
        assert post("action=step")["paused"]
        while demo.step():
            pass

        stream = get("/api/stream?after=0")
        assert stream["finished"] and stream["done"] == len(stream["lines"])
        assert {"raw", "status", "template_id", "payload", "firmware"} <= stream["lines"][0].keys()
        assert any(line["anomaly"] == "stuck" and line["device_id"] == plug for line in stream["lines"])

        kinds = {e["kind"] for e in get("/api/events?after=0")["events"]}
        # The rollout changes wording, so enrichment must learn or re-associate templates.
        assert {"control", "rollout", "truth", "end"} <= kinds and kinds & {"new", "alias"}
        assert kinds & {"tp", "fn"}

        system = get("/api/system")
        assert system["hit"] + system["miss"] == system["lines"] == stream["done"]
        assert system["queue"]["kind"] == "memory" and system["queue"]["published"] == system["miss"]
        assert system["published"] >= 1 and system["mask_backend"] == "offline"

        pending = next(line["i"] for line in stream["lines"] if line["status"] == "PENDING")
        trace = get(f"/api/trace?i={pending}")
        assert "queue" in trace["path"] and "miss" in trace["trail"][0]
        assert len(trace["trail"]) >= 2  # a cycle resolved, requeued or dead-lettered it

        templates = get("/api/templates")
        first = next(t for t in templates["templates"] if t["count"])
        assert {"template_id", "tokens", "keys", "count", "learned", "sql"} <= first.keys()
        result = get("/api/query?sql=" + quote(first["sql"]))
        assert result["columns"][:2] == ["timestamp", "device_id"] and result["rows"]
        assert get("/api/metrics")["windows"]
        assert "report" in get("/api/eval")
        try:
            post("action=inject&value=nope:stuck")
        except HTTPError as exc:
            assert exc.code == 400
        else:
            raise AssertionError("unknown device must be rejected")

        # Restart: same stream from the first line (manual rollout/inject are not replayed),
        # keeping what enrichment learned.
        first = {k: stream["lines"][0][k] for k in ("time", "device_id")}
        learned = system["templates"]
        assert post("action=restart&value=same,keep")["session"] == 2
        demo.step()
        again = get("/api/stream?after=0")
        assert again["session"] == 2 and {k: again["lines"][0][k] for k in first} == first
        after = get("/api/system")
        assert after["templates"] == learned and after["queue"]["published"] <= after["miss"]
        assert get("/api/events?after=0")["events"][0]["text"].startswith("session 2 started")
        post("action=restart&value=new,queue=memory,llm=offline")
        assert get("/api/system")["templates"] == len(demo.initial.templates)
        try:
            post("action=restart&value=same,store=postgres")
        except HTTPError as exc:
            assert exc.code == 400
        else:
            raise AssertionError("an unknown store must be rejected")
        assert get("/api/system")["session"] == 3  # the running session is untouched
    finally:
        server.shutdown()


def test_saved_registry_is_loaded_with_its_mask_backend(tmp_path):
    path = tmp_path / "demo_registry.json"
    save_registry(path, devices=1, history_hours=0.5, log=lambda _: None)
    demo = _demo(registry_path=path)
    assert demo.mask_backend == "offline" and demo.registry.current.templates


@pytest.mark.integration
def test_clickhouse_store_receives_rows_and_answers_queries():
    pytest.importorskip("clickhouse_connect")
    try:
        demo = _demo(store="clickhouse")
    except RuntimeError as error:
        pytest.skip(f"ClickHouse not available: {error}")
    for _ in range(300):
        demo.step()
    first = next(t for t in demo.templates()["templates"] if t["count"])
    assert "FROM iot_logs FINAL" in first["sql"]
    result = demo.query(first["sql"])
    assert result["engine"] == "clickhouse" and result["rows"]
    total = demo.query("SELECT count() FROM iot_logs FINAL")["rows"][0][0]
    assert total == len(demo.lines)
    demo.store.close()


def test_calibration_is_cached_per_backend_and_fleet(tmp_path):
    logs: list[str] = []
    first = _demo(cache_dir=tmp_path, log=logs.append)
    assert any("calibrating" in m for m in logs)
    logs.clear()
    again = _demo(cache_dir=tmp_path, log=logs.append)
    assert not any("calibrating" in m for m in logs) and any("reusing" in m for m in logs)
    assert again.initial.templates.keys() == first.initial.templates.keys()
    logs.clear()
    _demo(cache_dir=tmp_path, recalibrate=True, log=logs.append)
    assert any("calibrating" in m for m in logs)
