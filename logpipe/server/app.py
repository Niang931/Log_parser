"""HTTP surface of ``logpipe server``: ``/intake/v1/*`` for agents (API key), ``/api/*`` for the
React UI, the built UI at ``/`` and the developer demo at ``/demo/``."""

import asyncio
import json
import zlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)

from ..registry import UNK_TEMPLATE_ID
from .core import Busy, Server

STATIC = Path(__file__).with_name("static")
DEMO_INDEX = Path(__file__).parent.parent / "demo" / "index.html"
MAX_BODY = 64 << 20  # decompressed intake batch


def template_sql(entry, engine: str) -> str:
    keys = [k for k in entry.keys if k not in ("timestamp", "device_id")]
    if engine == "clickhouse":
        cols = "".join(f",\n  payload.`{k}` AS `{k}`" for k in keys)
        return (f"SELECT timestamp, device_id AS host, service{cols}\nFROM iot_logs FINAL\n"
                f"WHERE template_id = {entry.template_id}\nORDER BY timestamp DESC\nLIMIT 50")
    cols = "".join(f",\n  json_extract(payload, '$.\"{k}\"') AS \"{k}\"" for k in keys)
    return (f"SELECT timestamp, device_id AS host, service{cols}\nFROM logs\n"
            f"WHERE template_id = {entry.template_id}\nORDER BY timestamp DESC\nLIMIT 50")


def create_app(server: Server, demo: Any = None) -> FastAPI:
    app = FastAPI(title="logpipe", docs_url="/api/docs", openapi_url="/api/openapi.json")
    engine = "sqlite" if server.reads.__class__.__name__ == "MemoryReads" else "clickhouse"

    def agent_key(x_api_key: Annotated[str | None, Header()] = None) -> dict:
        key = server.check_key(x_api_key)
        if key is None:
            raise HTTPException(401, "missing or unknown API key")
        return key

    @app.exception_handler(KeyError)
    async def not_found(_request, exc: KeyError):
        return JSONResponse({"detail": f"not found: {exc.args[0] if exc.args else ''}"}, 404)

    @app.exception_handler(ValueError)
    async def bad_request(_request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, 400)

    # --- intake ---------------------------------------------------------------------
    @app.get("/intake/v1/ping")
    def ping(key: Annotated[dict, Depends(agent_key)]) -> dict:
        return {"ok": True, "key": key["name"]}

    @app.post("/intake/v1/logs", status_code=202)
    async def intake_logs(request: Request, key: Annotated[dict, Depends(agent_key)]) -> dict:
        raw = await request.body()
        if request.headers.get("content-encoding") == "gzip":
            try:
                d = zlib.decompressobj(16 + zlib.MAX_WBITS)
                raw = d.decompress(raw, MAX_BODY)
                if d.unconsumed_tail:
                    raise HTTPException(413, "batch too large")
            except zlib.error as error:
                raise HTTPException(400, f"bad gzip body: {error}") from error
        try:
            events = json.loads(raw)["events"]
            assert isinstance(events, list)
        except (ValueError, KeyError, TypeError, AssertionError) as error:
            raise HTTPException(400, "expected {\"events\": [...]}") from error
        try:
            return {"accepted": server.ingest(events)}
        except Busy as error:
            raise HTTPException(429, str(error)) from error

    @app.post("/intake/v1/heartbeat")
    def intake_heartbeat(status: Annotated[dict, Body()], key: Annotated[dict, Depends(agent_key)]) -> dict:
        return server.heartbeat(status, key)

    @app.get("/intake/v1/wait")
    async def intake_wait(request: Request, hostname: str, key: Annotated[dict, Depends(agent_key)],
                          remote_hash: str = "", timeout: float = 25) -> dict:
        """Long-poll held by every agent: answers as soon as its remote config differs from
        ``remote_hash`` (else after ``timeout``); while it's open the agent counts as online."""
        server.events.bind()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + min(max(timeout, 0), 55)
        server.connected(hostname, +1)
        try:
            while (current := server.remote_hash_of(hostname)) == remote_hash and loop.time() < deadline:
                if await request.is_disconnected():
                    break
                await server.events.wait(min(1.0, max(deadline - loop.time(), 0)))
            return {"remote_hash": current}
        finally:
            server.connected(hostname, -1)

    @app.get("/intake/v1/config")
    def intake_config(hostname: str, key: Annotated[dict, Depends(agent_key)]) -> dict:
        return {"yaml": (server.control.get("remote", hostname) or {}).get("yaml", "")}

    @app.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        """Server-sent events: one ``change`` per agent/fleet change, so the UI refetches at once."""
        server.events.bind()

        async def stream():
            yield "retry: 1000\n\n"
            while not await request.is_disconnected():
                yield "event: change\ndata: {}\n\n" if await server.events.wait(15) else ": ping\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # --- overview, logs, templates, stats --------------------------------------------
    @app.get("/api/overview")
    def overview() -> dict:
        agents = server.agents()
        warnings = [w for w in server.control.all("warning") if w["status"] == "open"]
        jobs = [j for j in server.control.all("job") if j["status"] in ("queued", "running")]
        snap = server.registry.current
        return {
            "received": server.received, "rate": round(server.rate(), 2),
            "status": server.store.count_by_status(), "inbox": server.inbox.qsize(),
            "agents": len(agents), "agents_online": sum(a["online"] for a in agents),
            "sources": len(server.sources), "templates": len(snap.templates),
            "registry_version": snap.version,
            "warnings": dict(Counter(w["kind"] for w in warnings)),
            "streams": len(server.control.all("stream")), "models": len(server.scorers),
            "settings": server.settings, "jobs": jobs,
            "recent_alerts": server.reads.alerts(5),
        }

    @app.get("/api/logs")
    def logs(host: str | None = None, service: str | None = None, source: str | None = None,
             status: str | None = None, template_id: int | None = None, q: str = "",
             limit: int = 200) -> dict:
        filters = {k: v for k, v in (("host", host), ("service", service), ("source", source),
                                      ("status", status), ("template_id", template_id)) if v is not None and v != ""}
        return {"logs": server.reads.recent(filters, min(max(limit, 1), 1000), q)}

    @app.get("/api/templates")
    def templates() -> dict:
        snap = server.registry.current
        calibrated_at = server.settings.get("calibrated_at") or ""
        vocabs = {sid: s.vocab for sid, s in server.scorers.items()}
        out = []
        for t in sorted(snap.templates.values(), key=lambda t: t.template_id):
            meta = server.tmeta.get(t.template_id, {})
            seen_by = [sid for sid, v in vocabs.items() if t.template_id in v]
            out.append(asdict(t) | {
                "text": " ".join(t.tokens), "count": meta.get("count", 0),
                "first_seen": meta.get("first_seen"), "last_seen": meta.get("last_seen"),
                "seen_by": seen_by,
                "candidate": not seen_by and (meta.get("first_seen") or "") > calibrated_at,
                "sql": template_sql(t, engine)})
        return {"version": snap.version, "masks": [m.to_dict() for m in snap.masks],
                "templates": out, "engine": engine,
                "pending": server.store.count_by_status().get("PENDING", 0)}

    @app.post("/api/query")
    def query(body: Annotated[dict, Body()]) -> dict:
        sql = str(body.get("sql", "")).strip().rstrip(";")
        if not sql.lower().startswith(("select", "with")):
            raise HTTPException(400, "only SELECT / WITH queries")
        return server.reads.query(sql)

    @app.get("/api/stats")
    def stats(minutes: int = 60) -> dict:
        minutes = min(max(minutes, 1), 7 * 24 * 60)
        snap = server.registry.current
        top_t = server.reads.top("template_id", minutes, 10)
        return {
            "per_minute": server.reads.per_minute(minutes),
            "hosts": server.reads.top("host", minutes, 10),
            "services": server.reads.top("service", minutes, 10),
            "sources": server.reads.top("source", minutes, 10),
            "templates": [[tid, n, " ".join(snap.templates[tid].tokens) if tid in snap.templates
                           else ("unparsed (pending)" if tid == UNK_TEMPLATE_ID else f"#{tid}")]
                          for tid, n in top_t],
        }

    @app.get("/api/alerts")
    def alerts(limit: int = 200) -> dict:
        return {"alerts": server.reads.alerts(min(max(limit, 1), 1000)),
                "streams": {sid: {"miss_threshold": s.settings.thresholds.miss_ratio,
                                  "version": s.version} for sid, s in server.scorers.items()}}

    # --- agents, keys -------------------------------------------------------------------
    @app.get("/api/agents")
    def agents() -> dict:
        return {"agents": server.agents()}

    @app.get("/api/agents/{host}")
    def agent(host: str) -> dict:
        found = next((a for a in server.agents() if a["hostname"] == host), None)
        if found is None:
            raise KeyError(host)
        return found

    @app.put("/api/agents/{host}/remote-config")
    def put_remote(host: str, body: Annotated[dict, Body()]) -> dict:
        return server.set_remote(host, str(body.get("yaml", "")))

    @app.delete("/api/agents/{host}")
    def forget_agent(host: str) -> dict:
        server.control.delete("agent", host)
        server.events.notify()
        return {"ok": True}

    @app.get("/api/keys")
    def keys() -> dict:
        return {"keys": sorted(({k: v for k, v in d.items() if k != "hash"}
                                for d in server.control.all("key")), key=lambda d: d["created_at"])}

    @app.post("/api/keys")
    def create_key(body: Annotated[dict, Body()]) -> dict:
        doc, token = server.create_key(str(body.get("name", "")))
        return {k: v for k, v in doc.items() if k != "hash"} | {"token": token}

    @app.delete("/api/keys/{key_id}")
    def revoke_key(key_id: str) -> dict:
        server.revoke_key(key_id)
        return {"ok": True}

    # --- learning lifecycle -------------------------------------------------------------
    @app.get("/api/sources")
    def sources() -> dict:
        return {"sources": sorted(({**s, "streams": [x["id"] for x in server.streams_for(s["id"])]}
                                   for s in server.sources.values()), key=lambda s: s["id"])}

    @app.get("/api/setup")
    def setup() -> dict:
        return {
            "keys": sum(1 for k in server.control.all("key") if not k.get("revoked")),
            "agents": [{"hostname": a["hostname"], "online": a["online"],
                        "sources": len(a.get("status", {}).get("sources") or [])}
                       for a in server.agents()],
            "sources": [{**s, "streams": [x["id"] for x in server.streams_for(s["id"])]}
                        for s in server.sources.values()],
            "settings": server.settings,
            "streams": server.control.all("stream"),
            "jobs": sorted(server.control.all("job"), key=lambda j: j["created_at"])[-5:],
        }

    @app.post("/api/setup/calibrate", status_code=202)
    def start_calibration(body: Annotated[dict | None, Body()] = None) -> dict:
        if body and "sample_cap" in body:
            update_settings({"sample_cap": body["sample_cap"]})
        return server.enqueue("calibrate")

    @app.get("/api/settings")
    def get_settings() -> dict:
        return server.settings

    @app.put("/api/settings")
    def update_settings(body: Annotated[dict, Body()]) -> dict:
        changes = {}
        if "sample_cap" in body:
            cap = int(body["sample_cap"])
            if not 100 <= cap <= 200_000:
                raise ValueError("sample_cap must be between 100 and 200000 lines per source")
            changes["sample_cap"] = cap
        if "retrain_days" in body:
            days = float(body["retrain_days"])
            if days <= 0:
                raise ValueError("retrain_days must be positive")
            changes["retrain_days"] = days
        server.settings.update(changes)
        server.control.put("settings", "server", server.settings)
        return server.settings

    @app.get("/api/streams")
    def streams() -> dict:
        out = []
        for s in server.control.all("stream"):
            members = [sid for sid in server.sources if any(x["id"] == s["id"] for x in server.streams_for(sid))]
            out.append(s | {"sources": members, "has_model": s["id"] in server.scorers,
                            "vocab": len(s.get("vocab") or [])})
        return {"streams": sorted(out, key=lambda s: (s["id"] != "all", s["name"]))}

    @app.post("/api/streams")
    def create_stream(body: Annotated[dict, Body()]) -> dict:
        return server.save_stream(body)

    @app.put("/api/streams/{stream_id}")
    def update_stream(stream_id: str, body: Annotated[dict, Body()]) -> dict:
        if server.control.get("stream", stream_id) is None:
            raise KeyError(stream_id)
        return server.save_stream(body, stream_id)

    @app.delete("/api/streams/{stream_id}")
    def delete_stream(stream_id: str) -> dict:
        server.delete_stream(stream_id)
        return {"ok": True}

    @app.post("/api/streams/{stream_id}/train", status_code=202)
    def train_stream(stream_id: str) -> dict:
        if server.control.get("stream", stream_id) is None:
            raise KeyError(stream_id)
        return server.enqueue("retrain", streams=[stream_id])

    @app.get("/api/warnings")
    def warnings() -> dict:
        return {"warnings": server.warnings()}

    @app.post("/api/warnings/{wid}/resolve")
    def resolve(wid: str, body: Annotated[dict | None, Body()] = None) -> dict:
        return server.resolve_warning(wid, str((body or {}).get("note") or "marked resolved"))

    @app.post("/api/warnings/{wid}/same-as")
    def same_as(wid: str, body: Annotated[dict, Body()]) -> dict:
        return server.same_as(wid, str(body["target"]))

    @app.post("/api/warnings/{wid}/recalibrate", status_code=202)
    def recalibrate(wid: str) -> dict:
        return server.recalibrate_warning(wid)

    @app.get("/api/jobs")
    def jobs() -> dict:
        return {"jobs": sorted(server.control.all("job"), key=lambda j: j["created_at"], reverse=True)[:50]}

    # --- demo ---------------------------------------------------------------------------
    @app.get("/demo", include_in_schema=False)
    def demo_redirect():
        return RedirectResponse("/demo/")

    @app.get("/demo/", include_in_schema=False)
    def demo_page():
        if demo is None:
            return HTMLResponse(
                "<!doctype html><meta charset=utf-8><title>logpipe demo</title>"
                "<body style='font:16px system-ui;max-width:40rem;margin:4rem auto;padding:0 1rem'>"
                "<h1>Demo not running</h1><p>The developer demo replays a synthetic IoT fleet through "
                "the whole pipeline. Start the server with <code>logpipe server --demo</code> "
                "(it calibrates and trains a small model first), or run <code>logpipe demo</code> "
                "on its own.</p><p><a href='/'>Back to logpipe</a></p>", 404)
        return FileResponse(DEMO_INDEX)

    if demo is not None:
        demo_get = {
            "stream": lambda q: demo.stream(int(q.get("after", 0))),
            "system": lambda q: demo.system(),
            "events": lambda q: demo.timeline(int(q.get("after", 0))),
            "templates": lambda q: demo.templates(),
            "query": lambda q: demo.query(q["sql"]),
            "metrics": lambda q: demo.metrics(),
            "trace": lambda q: demo.trace(int(q["i"])),
            "eval": lambda q: demo.eval_report(),
        }

        @app.get("/demo/api/{name}", include_in_schema=False)
        def demo_api(name: str, request: Request):
            if name not in demo_get:
                raise KeyError(name)
            try:
                return demo_get[name](dict(request.query_params))
            except Exception as error:  # noqa: BLE001 - same {"error"} shape the demo page reads
                return JSONResponse({"error": f"{type(error).__name__}: {error}"}, 400)

        @app.post("/demo/api/control", include_in_schema=False)
        def demo_control(action: str, value: str = ""):
            try:
                return demo.control(action, value)
            except Exception as error:  # noqa: BLE001
                return JSONResponse({"error": f"{type(error).__name__}: {error}"}, 400)

    # --- the React app --------------------------------------------------------------------
    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith(("api/", "intake/")):
            raise HTTPException(404)
        file = (STATIC / path).resolve()
        if path and file.is_file() and STATIC.resolve() in file.parents:
            return FileResponse(file)
        index = STATIC / "index.html"
        if index.exists():
            return FileResponse(index)
        return HTMLResponse("<!doctype html><meta charset=utf-8><title>logpipe</title><body style="
                            "'font:16px system-ui;margin:4rem'><h1>UI not built</h1><p>Run "
                            "<code>cd web &amp;&amp; npm ci &amp;&amp; npm run build</code>; the API is "
                            "at <a href='/api/docs'>/api/docs</a>.</p>", 503)

    return app
