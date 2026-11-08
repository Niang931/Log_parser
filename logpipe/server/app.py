"""HTTP surface of ``logpipe server``: ``/intake/v1/*`` for agents (API key), ``/api/*`` for the
React UI, the built UI at ``/`` and the developer demo at ``/demo/``."""

import asyncio
import importlib.metadata
import json
import os
import uuid
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
    Response,
    StreamingResponse,
)

from ..parsing.grok import rule_from_template
from ..registry import UNK_TEMPLATE_ID
from . import otlp, querybuild, querybuild_mongo
from .auth import COOKIE, SESSION_SECONDS, Auth
from .core import MIN_DRAFT_LINES, Busy, Server, now_iso
from .metrics import METRICS
from .reads import FACETS

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


def create_app(server: Server, demo: Any = None, info: dict | None = None) -> FastAPI:
    """``info``: how ``logpipe server`` was started (store, queue, LLM backend), shown on Settings."""
    started = now_iso()
    app = FastAPI(title="logpipe", docs_url="/api/docs", openapi_url="/api/openapi.json")
    engine = {"MemoryReads": "sqlite", "MongoReads": "mongodb"}.get(server.reads.__class__.__name__, "clickhouse")
    qb = querybuild_mongo if engine == "mongodb" else querybuild
    auth = Auth(server.control)

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        # `logpipe server` always sets a password before serving; with none set (tests, an
        # embedded app) the API stays open. Agents' intake routes use API keys instead.
        path = request.url.path
        guarded = (path.startswith(("/api/", "/demo/api/")) and not path.startswith("/api/auth/"))
        if guarded and auth.configured and not auth.valid(request.cookies.get(COOKIE)):
            return JSONResponse({"detail": "log in first"}, 401)
        return await call_next(request)

    def client_of(request: Request) -> str:
        return request.client.host if request.client else "?"

    @app.get("/api/auth/status")
    def auth_status(request: Request) -> dict:
        return {"required": auth.configured,
                "authenticated": not auth.configured or auth.valid(request.cookies.get(COOKIE))}

    @app.post("/api/auth/login")
    def login(request: Request, body: Annotated[dict, Body()]) -> Response:
        client = client_of(request)
        if wait := auth.blocked_for(client):
            raise HTTPException(429, f"too many wrong passwords; try again in {int(wait) // 60 + 1} min")
        token = auth.login(client, str(body.get("password") or ""))
        if token is None:
            raise HTTPException(401, "wrong password")
        res = JSONResponse({"ok": True})
        res.set_cookie(COOKIE, token, max_age=SESSION_SECONDS, httponly=True, samesite="strict",
                       secure=request.url.scheme == "https")
        return res

    @app.post("/api/auth/logout")
    def logout() -> Response:
        res = JSONResponse({"ok": True})
        res.delete_cookie(COOKIE)
        return res

    @app.post("/api/auth/password")
    def change_password(request: Request, body: Annotated[dict, Body()]) -> Response:
        if not auth.valid(request.cookies.get(COOKIE)):
            raise HTTPException(401, "log in first")
        client = client_of(request)
        if wait := auth.blocked_for(client):
            raise HTTPException(429, f"too many wrong passwords; try again in {int(wait) // 60 + 1} min")
        if auth.login(client, str(body.get("current") or "")) is None:
            raise HTTPException(403, "the current password is wrong")
        auth.set_password(str(body.get("new") or ""))  # ValueError -> 400 when too short
        res = JSONResponse({"ok": True})  # the old sessions are void now; this one is renewed
        res.set_cookie(COOKIE, auth.login(client, str(body["new"])) or "", max_age=SESSION_SECONDS,
                       httponly=True, samesite="strict", secure=request.url.scheme == "https")
        return res

    def agent_key(x_api_key: Annotated[str | None, Header()] = None,
                  authorization: Annotated[str | None, Header()] = None) -> dict:
        # X-API-Key, or `Authorization: Bearer <key>` as OTel exporters are usually configured
        token = x_api_key or (authorization or "").removeprefix("Bearer ").strip() or None
        key = server.check_key(token)
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

    async def body(request: Request) -> bytes:
        raw = await request.body()
        if request.headers.get("content-encoding") == "gzip":
            try:
                d = zlib.decompressobj(16 + zlib.MAX_WBITS)
                raw = d.decompress(raw, MAX_BODY)
                if d.unconsumed_tail:
                    raise HTTPException(413, "batch too large")
            except zlib.error as error:
                raise HTTPException(400, f"bad gzip body: {error}") from error
        return raw

    @app.post("/v1/logs")
    async def otlp_logs(request: Request, key: Annotated[dict, Depends(agent_key)]) -> Response:
        """OTLP/HTTP logs (OpenTelemetry), JSON or protobuf: what logpipe-agent sends, and what
        the OpenTelemetry Collector's otlphttp exporter or an OTel SDK can send."""
        raw = await body(request)
        ctype = request.headers.get("content-type", "application/json")
        try:
            evs = otlp.events(otlp.decode(raw, ctype))
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        try:
            server.ingest(evs)
        except Busy as error:  # OTLP clients retry 429 with backoff
            raise HTTPException(429, str(error)) from error
        # An empty ExportLogsServiceResponse in the request's encoding.
        proto = ctype.split(";")[0].strip() == otlp.PROTOBUF
        return Response(b"" if proto else b"{}", media_type=otlp.PROTOBUF if proto else "application/json")

    @app.post("/intake/v1/logs", status_code=202)
    async def intake_logs(request: Request, key: Annotated[dict, Depends(agent_key)]) -> dict:
        """logpipe's own format, kept for agents from before OTLP (deprecated)."""
        raw = await body(request)
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
            "review": len(server.batches()),
            "streams": len(server.control.all("stream")), "models": len(server.scorers),
            "settings": server.settings, "jobs": jobs,
            "recent_alerts": server.reads.alerts(5),
        }

    def log_filters(host: str | None = None, service: str | None = None, source: str | None = None,
                    status: str | None = None, template_id: int | None = None,
                    level: str | None = None) -> dict:
        return {k: v for k, v in (("host", host), ("service", service), ("source", source),
                                  ("status", status), ("template_id", template_id), ("level", level))
                if v is not None and v != ""}

    def clamp(minutes: int) -> int:
        return min(max(minutes, 1), 7 * 24 * 60)

    @app.get("/api/logs")
    def logs(filters: Annotated[dict, Depends(log_filters)], q: str = "", minutes: int | None = None, limit: int = 200) -> dict:
        return {"logs": server.reads.recent(filters, min(max(limit, 1), 1000), q,
                                            clamp(minutes) if minutes else None)}

    @app.get("/api/logs/histogram")
    def histogram(filters: Annotated[dict, Depends(log_filters)], q: str = "", minutes: int = 15, by: str = "level") -> dict:
        """Line counts per bucket over the whole range (empty buckets included), by level or
        parse status: the explorer's timeline."""
        return server.reads.histogram(clamp(minutes), by, filters, q)

    @app.get("/api/logs/facets")
    def facets(filters: Annotated[dict, Depends(log_filters)], q: str = "", minutes: int = 15) -> dict:
        """Counts per value of each facet, under every filter except the facet's own, so the
        other values stay visible to switch to."""
        return {f: server.reads.top(f, clamp(minutes), 12, {k: v for k, v in filters.items() if k != f}, q)
                for f in FACETS}

    @app.get("/api/templates")
    def templates() -> dict:
        snap = server.registry.current
        calibrated_at = server.settings.get("calibrated_at") or ""
        vocabs = {sid: s.vocab for sid, s in server.scorers.items()}
        out = []
        links: dict[int, list[int]] = {}
        for t in snap.templates.values():
            if t.alias_of is not None:
                links.setdefault(t.alias_of, []).append(t.template_id)
        for t in sorted(snap.templates.values(), key=lambda t: t.template_id):
            meta = server.tmeta.get(t.template_id, {})
            seen_by = [sid for sid, v in vocabs.items() if t.template_id in v]
            out.append(asdict(t) | {
                "text": " ".join(t.tokens), "count": meta.get("count", 0),
                "rule": t.rule or rule_from_template(t.tokens, t.keys), "rule_edited": bool(t.rule),
                # "you", or the online learner: "pipeline" (only variables differ) or "llm" (it confirmed a rewording)
                "linked_by": (meta.get("linked_by") or "llm") if t.alias_of is not None else None,
                "link_reason": meta.get("link_reason") if t.alias_of is not None else None,
                "canonical": snap.canonical_id(t.template_id), "linked": sorted(links.get(t.template_id, [])),
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
        if engine != "mongodb" and not sql.lower().startswith(("select", "with")):
            raise HTTPException(400, "only SELECT / WITH queries")
        return server.reads.query(sql)

    def query_schema() -> dict:
        snap = server.registry.current
        fields = set(snap.key_vocab) | {k for t in snap.templates.values() for k in t.keys}
        return qb.schema(engine, sorted(fields))

    @app.get("/api/query/schema")
    def get_query_schema() -> dict:
        return query_schema()

    @app.post("/api/query/build")
    def build_query(body: Annotated[dict, Body()]) -> dict:
        sql = qb.build(body, engine)
        return {"sql": sql, "result": server.reads.query(sql)}

    # --- Ask: question -> plan of read-only queries -> approval -> answer (logpipe.ask) ---
    def ask_flag(body: dict) -> bool | None:
        v = body.get("require_approval")
        if v is not None and not isinstance(v, bool):
            raise ValueError("require_approval is true or false")
        return v

    @app.post("/api/ask")
    def ask(body: Annotated[dict, Body()]) -> dict:
        tz = body.get("tz_offset")
        if tz is not None and (not isinstance(tz, int) or not -16 * 60 <= tz <= 16 * 60):
            raise ValueError("tz_offset is minutes east of UTC")
        spec = body.get("spec")
        if spec is not None and not isinstance(spec, dict):
            raise ValueError("spec is an object")
        return server.ask.ask(str(body.get("question", "")), require_approval=ask_flag(body), tz_offset=tz, spec=spec)

    @app.get("/api/ask")
    def asked(limit: int = 20) -> dict:
        return {"plans": server.ask.recent(max(1, min(limit, 100))),
                "require_approval": bool(server.settings.get("ask_require_approval", True))}

    @app.get("/api/ask/{plan_id}")
    def ask_plan(plan_id: str) -> dict:
        return server.ask.get(plan_id)

    @app.post("/api/ask/{plan_id}/approve")
    def approve_plan(plan_id: str, body: Annotated[dict | None, Body()] = None) -> dict:
        ids = (body or {}).get("queries")
        if ids is not None and not (isinstance(ids, list) and all(isinstance(i, str) for i in ids)):
            raise ValueError("queries is a list of query ids")
        return server.ask.approve(plan_id, ids)

    @app.post("/api/ask/{plan_id}/edit")
    def edit_plan(plan_id: str, body: Annotated[dict, Body()]) -> dict:
        if not isinstance(body.get("spec"), dict):
            raise ValueError("send the edited spec as {\"spec\": {...}}")  # noqa: TRY004 (ValueError is the 400 handler)
        return server.ask.edit(plan_id, body["spec"])

    @app.post("/api/ask/{plan_id}/reject")
    def reject_plan(plan_id: str) -> dict:
        return server.ask.reject(plan_id)

    @app.get("/api/ask-shutdown")
    def shutdown_templates() -> dict:
        return server.ask.shutdown_view()

    @app.put("/api/ask-shutdown")
    def edit_shutdown(body: Annotated[dict, Body()]) -> dict:
        return server.ask.edit_shutdown(body)

    @app.get("/api/queries")
    def saved_queries() -> dict:
        return {"queries": sorted(server.control.all("saved_query"), key=lambda q: q["name"].lower())}

    @app.post("/api/queries")
    def save_query(body: Annotated[dict, Body()]) -> dict:
        name, sql = str(body.get("name", "")).strip(), str(body.get("sql", "")).strip()
        if not name or not sql:
            raise ValueError("a saved query needs a name and SQL")
        qid = str(body.get("id") or uuid.uuid4().hex[:10])
        return server.control.put("saved_query", qid, {"id": qid, "name": name[:120], "sql": sql,
                                                       "spec": body.get("spec"), "saved_at": now_iso()})

    @app.delete("/api/queries/{qid}")
    def delete_query(qid: str) -> dict:
        if server.control.get("saved_query", qid) is None:
            raise KeyError(qid)
        server.control.delete("saved_query", qid)
        return {"ok": True}

    @app.get("/api/stats")
    def stats(minutes: int = 60) -> dict:
        minutes = clamp(minutes)
        snap = server.registry.current
        top_t = server.reads.top("template_id", minutes, 10)
        return {
            "hosts": server.reads.top("host", minutes, 10),
            "services": server.reads.top("service", minutes, 10),
            "sources": server.reads.top("source", minutes, 10),
            "templates": [[tid, n, " ".join(snap.templates[tid].tokens) if tid in snap.templates
                           else ("unparsed (pending)" if tid == UNK_TEMPLATE_ID else f"#{tid}")]
                          for tid, n in top_t],
        }

    @app.get("/api/system")
    def system(minutes: int = 60) -> dict:
        """The server's own metrics (see ``metrics.py``) over the last ``minutes``."""
        return {"metrics": {k: {"label": label, "unit": unit, "group": group}
                            for k, (label, unit, group) in METRICS.items()},
                "current": server.metrics.current, "sampled_at": server.metrics.sampled_at,
                "every": server.metrics.every, "cpus": os.cpu_count(),
                **server.metrics.series(clamp(minutes))}

    @app.get("/api/drift")
    def drift() -> dict:
        return {"streams": server.drift.report(server.control.all("stream"))}

    @app.get("/api/system/alerts")
    def system_alerts() -> dict:
        return {"alerts": server.metric_rules.alerts(), "rules": server.metric_rules.rules()}

    @app.post("/api/system/rules")
    def add_rule(body: Annotated[dict, Body()]) -> dict:
        return server.metric_rules.save(body)

    @app.put("/api/system/rules/{rule_id}")
    def edit_rule(rule_id: str, body: Annotated[dict, Body()]) -> dict:
        if server.control.get("metric_rule", rule_id) is None:
            raise KeyError(rule_id)
        return server.metric_rules.save(body, rule_id)

    @app.delete("/api/system/rules/{rule_id}")
    def delete_rule(rule_id: str) -> dict:
        server.metric_rules.delete(rule_id)
        return {"ok": True}

    @app.get("/api/alerts")
    def alerts(limit: int = 200) -> dict:
        return {"alerts": server.reads.alerts(min(max(limit, 1), 1000)),
                "streams": {sid: {"miss_threshold": s.settings.thresholds.miss_ratio,
                                  "version": s.version} for sid, s in server.scorers.items()}}

    @app.get("/api/incidents")
    def incidents(status: str = "") -> dict:
        out = [i | {"windows": len(i["windows"])} for i in server.incidents()]
        return {"incidents": [i for i in out if not status or i["status"] == status]}

    @app.get("/api/incidents/{incident_id}")
    def incident(incident_id: str) -> dict:
        return server.incident(incident_id) | {"windows": None}

    @app.put("/api/incidents/{incident_id}")
    def set_incident(incident_id: str, body: Annotated[dict, Body()]) -> dict:
        return server.set_incident(incident_id, str(body.get("status", "")), str(body.get("note") or ""))

    @app.post("/api/incidents/{incident_id}/explain", status_code=202)
    def explain_incident(incident_id: str, body: Annotated[dict | None, Body()] = None) -> dict:
        return server.explain_incident(incident_id, ask_flag(body or {}))

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
            "min_lines": MIN_DRAFT_LINES,
            "review": next((b for b in server.batches() if b["initial"]), None),
        }

    @app.get("/api/about")
    def about() -> dict:
        try:
            version = importlib.metadata.version("log-parser")
        except importlib.metadata.PackageNotFoundError:
            version = "dev"
        return {"version": version, "engine": engine, "started_at": started, "cpus": os.cpu_count(),
                "password_set": auth.configured, "metrics_days": 30, **(info or {})}

    @app.get("/api/settings")
    def get_settings() -> dict:
        return server.settings

    @app.put("/api/settings")
    def update_settings(body: Annotated[dict, Body()]) -> dict:
        return server.update_settings(body)

    # --- template review -----------------------------------------------------------------
    @app.post("/api/review/generate", status_code=202)
    def generate_drafts(body: Annotated[dict | None, Body()] = None) -> dict:
        return server.generate_drafts((body or {}).get("sample_cap"))

    @app.get("/api/review")
    def review_batches() -> dict:
        return {"batches": server.batches(), "auto_templates": server.auto_templates()}

    @app.get("/api/review/{batch_id}")
    def review_batch(batch_id: str) -> dict:
        batch = server.control.get("batch", batch_id)
        if batch is None:
            raise KeyError(batch_id)
        return batch | {"drafts": server.drafts(batch_id)}

    @app.post("/api/review/{batch_id}/check")
    def check_draft(batch_id: str, body: Annotated[dict, Body()]) -> dict:
        return server.check_draft(batch_id, [str(t) for t in body["tokens"]], [str(k) for k in body["keys"]])

    @app.post("/api/review/{batch_id}/decide-all")
    def decide_all(batch_id: str, body: Annotated[dict, Body()]) -> dict:
        ids = body.get("ids")
        return {"changed": server.decide_all(batch_id, str(body["status"]),
                                             [str(i) for i in ids] if isinstance(ids, list) else None)}

    @app.post("/api/review/{batch_id}/apply", status_code=202)
    def apply_batch(batch_id: str) -> dict:
        return server.apply_batch(batch_id)

    @app.post("/api/review/{batch_id}/discard")
    def discard_batch(batch_id: str) -> dict:
        return server.discard_batch(batch_id)

    @app.put("/api/drafts/{draft_id}")
    def update_draft(draft_id: str, body: Annotated[dict, Body()]) -> dict:
        return server.update_draft(draft_id, body)

    @app.post("/api/drafts/{draft_id}/merge")
    def merge_drafts(draft_id: str, body: Annotated[dict, Body()]) -> dict:
        return server.merge_drafts(draft_id, [str(o) for o in body["others"]])

    @app.post("/api/drafts/{draft_id}/check")
    def check_draft_rule(draft_id: str, body: Annotated[dict, Body()]) -> dict:
        return server.check_draft_rule(draft_id, str(body["rule"]))

    @app.post("/api/templates/{tid}/check")
    def check_template_rule(tid: int, body: Annotated[dict, Body()]) -> dict:
        return server.check_template_rule(tid, str(body["rule"]))

    @app.post("/api/templates/{tid}/link")
    def link_template(tid: int, body: Annotated[dict, Body()]) -> dict:
        return server.link_template(tid, int(body["target"]))

    @app.post("/api/templates/{tid}/unlink")
    def unlink_template(tid: int) -> dict:
        return server.unlink_template(tid)

    @app.post("/api/templates/{tid}/review")
    def review_template(tid: int, body: Annotated[dict, Body()]) -> dict:
        return server.review_template(tid, body)

    @app.get("/api/maintenance")
    def maintenance() -> dict:
        return server.maintenance()

    @app.get("/api/streams")
    def streams() -> dict:
        out = []
        for s in server.control.all("stream"):
            members = [sid for sid in server.sources if any(x["id"] == s["id"] for x in server.streams_for(sid))]
            out.append(s | {"sources": members, "has_model": s["id"] in server.scorers,
                            "vocab": len(s.get("vocab") or [])})
        return {"streams": sorted(out, key=lambda s: s["name"])}

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
        return server.train([stream_id])

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

    @app.get("/api/bench")
    def bench_runs() -> dict:
        return {"runs": sorted(server.control.all("bench"), key=lambda r: r["started_at"], reverse=True),
                "cpus": os.cpu_count()}

    @app.post("/api/bench", status_code=202)
    def bench_start(body: Annotated[dict | None, Body()] = None) -> dict:
        body = body or {}
        store = str(body.get("store") or "memory")
        if store not in ("memory", "clickhouse", "mongodb"):
            raise ValueError("store is memory, clickhouse or mongodb")
        if store != "memory" and engine != store:
            raise ValueError(f"this server has no {store} to benchmark against")
        return server.enqueue("bench", store=store, max_rate=max(0, int(body.get("max_rate") or 0)),
                              step_seconds=min(max(float(body.get("step_seconds") or 10), 2), 120))

    @app.delete("/api/bench/{run_id}")
    def bench_delete(run_id: str) -> dict:
        if server.control.get("bench", run_id) is None:
            raise KeyError(run_id)
        server.control.delete("bench", run_id)
        return {"ok": True}

    @app.get("/api/jobs")
    def jobs() -> dict:
        return {"jobs": sorted(server.control.all("job"), key=lambda j: j["created_at"], reverse=True)[:100]}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        return server.cancel_job(job_id)

    @app.post("/api/jobs/{job_id}/retry", status_code=202)
    def retry_job(job_id: str) -> dict:
        return server.retry_job(job_id)

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
            # hashed asset names change with every build, so they can be cached for good
            return FileResponse(file, headers={"Cache-Control": "public, max-age=31536000, immutable"}
                                if path.startswith("assets/") else None)
        index = STATIC / "index.html"
        if index.exists():  # revalidated every load, so an upgraded server serves its new UI
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
        return HTMLResponse("<!doctype html><meta charset=utf-8><title>logpipe</title><body style="
                            "'font:16px system-ui;margin:4rem'><h1>UI not built</h1><p>Run "
                            "<code>cd web &amp;&amp; npm ci &amp;&amp; npm run build</code>; the API is "
                            "at <a href='/api/docs'>/api/docs</a>.</p>", 503)

    return app
