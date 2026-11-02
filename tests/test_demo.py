import json
import threading
from http.server import ThreadingHTTPServer
from urllib.parse import quote
from urllib.request import Request, urlopen

from logpipe.demo.server import Demo, make_handler, save_registry


def _demo(**kw) -> Demo:
    return Demo(devices=1, history_hours=0.5, validation_hours=0.25, test_hours=0.5, epochs=1,
                enrich_every=50, log=lambda _: None, **kw)


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
        except Exception as exc:  # urllib raises on 400
            assert "400" in str(exc)
        else:
            raise AssertionError("unknown device must be rejected")
    finally:
        server.shutdown()


def test_saved_registry_is_loaded_with_its_mask_backend(tmp_path):
    path = tmp_path / "demo_registry.json"
    save_registry(path, devices=1, history_hours=0.5, log=lambda _: None)
    demo = _demo(registry_path=path)
    assert demo.mask_backend == "offline" and demo.registry.current.templates
