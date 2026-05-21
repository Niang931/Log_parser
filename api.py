"""
DeepParse v2 — FastAPI Demo Server
===================================
Exposes the full pipeline as a REST + Server-Sent Events API.
Serves the live dashboard at http://localhost:8000

Endpoints
---------
GET  /                          → Dashboard UI
POST /api/parse                 → Upload + parse a file (returns run_id)
GET  /api/stream/{run_id}       → SSE stream of live parse events
GET  /api/runs                  → List all runs with metrics
GET  /api/runs/{run_id}         → Single run detail + metrics
GET  /api/runs/{run_id}/records → Parsed records (paginated)
GET  /api/runs/{run_id}/clusters→ Template cluster analytics
GET  /api/health                → System health + config
DELETE /api/runs/{run_id}       → Remove a run

Rubric: real integration points, deploy model (API/CLI/container),
        security constraints, operational readiness, observable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── pipeline imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from main import (
    FAB_DOMAIN_CONTEXT,
    TELEMETRY,
    Telemetry,
    evaluate_records,
    health_check,
    load_logs,
    normalize,
    parse_to_records,
    sanitize_for_llm,
    set_global_seed,
    synthesize_masks_adaptive,
    validate_records,
    write_cluster_report,
    write_csv,
    write_json,
    write_sqlite,
)
from DeepParse.deepparse import Drain

# ---------------------------------------------------------------------------
log = logging.getLogger("deepparse.api")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

ROOT_DIR   = Path(__file__).parent
OUTPUT_DIR = ROOT_DIR / "artifacts" / "output"
UPLOAD_DIR = ROOT_DIR / "artifacts" / "uploads"
STATIC_DIR = ROOT_DIR / "static"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DeepParse v2",
    description="Silicon-Fab Log Parsing Pipeline — Live Demo API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# In-memory event store for SSE streaming
# ---------------------------------------------------------------------------

_event_queues: dict[str, asyncio.Queue] = {}
_run_cache:    dict[str, dict]          = {}


def _push_event(run_id: str, event_type: str, data: dict) -> None:
    """Push a structured event to the SSE queue for a run."""
    q = _event_queues.get(run_id)
    if q:
        payload = {"type": event_type, "ts": datetime.now(timezone.utc).isoformat(), **data}
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


# ---------------------------------------------------------------------------
# Security — allowed file extensions
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {
    ".log", ".txt", ".csv", ".tsv", ".json",
    ".jsonl", ".xml", ".yaml", ".yml", ".ini",
}
MAX_FILE_SIZE_MB = 50


def _check_file(filename: str, size_bytes: int) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"File type '{ext}' not supported. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    if size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_bytes/1024/1024:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB",
        )


# ---------------------------------------------------------------------------
# Background parse task (runs pipeline, pushes SSE events)
# ---------------------------------------------------------------------------

async def _run_pipeline(
    run_id:     str,
    file_path:  Path,
    llm_provider:       str   = "groq",
    static_masks_path:  str   = "masks_fab_universal.json",
    adaptive_threshold: float = 0.80,
    adaptive_rounds:    int   = 3,
    drain_sim:          float = 0.3,
    drain_depth:        int   = 3,
    max_logs:           int   = 1000,
    seed:               int   = 42,
) -> None:
    """Full pipeline as an async background task with live SSE events."""
    tel = Telemetry()
    set_global_seed(seed)
    t_start = time.monotonic()

    try:
        # ── Stage 1: Load ────────────────────────────────────────────────────
        _push_event(run_id, "stage", {"stage": "load", "message": "Loading file…"})
        await asyncio.sleep(0)

        raw_logs = await asyncio.get_event_loop().run_in_executor(
            None, lambda: load_logs(str(file_path), max_length=256)
        )
        _push_event(run_id, "load_done", {
            "message": f"Loaded {len(raw_logs)} raw lines",
            "raw_count": len(raw_logs),
        })

        # ── Stage 2: Normalise ───────────────────────────────────────────────
        _push_event(run_id, "stage", {"stage": "normalise", "message": "Normalising + sanitising…"})
        await asyncio.sleep(0)

        logs = await asyncio.get_event_loop().run_in_executor(
            None, lambda: normalize(raw_logs, max_length=256, max_logs=max_logs, sanitize=True)
        )
        _push_event(run_id, "normalise_done", {
            "message": f"Kept {len(logs)} unique lines after dedup + injection guard",
            "kept": len(logs),
        })

        # ── Stage 3: Mask synthesis ──────────────────────────────────────────
        _push_event(run_id, "stage", {
            "stage": "masks",
            "message": f"Synthesising masks via {llm_provider}…",
        })
        await asyncio.sleep(0)

        mask_cache = OUTPUT_DIR / f"masks_{file_path.stem}.json"
        masks_dicts = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: synthesize_masks_adaptive(
                logs=logs,
                llm_provider=llm_provider,
                max_length=256,
                mask_cache_path=mask_cache,
                use_cache=False,
                static_masks_path=Path(static_masks_path) if Path(static_masks_path).exists() else None,
                telemetry=tel,
                max_rounds=adaptive_rounds,
                threshold=adaptive_threshold,
            ),
        )
        _push_event(run_id, "masks_done", {
            "message": f"{len(masks_dicts)} masks ready (static + LLM-synthesised)",
            "mask_count": len(masks_dicts),
            "llm_calls": tel.summary()["llm_calls"],
            "latency_s": tel.summary()["total_latency_s"],
        })

        # ── Stage 4: Drain parse (streaming record events) ───────────────────
        _push_event(run_id, "stage", {"stage": "drain", "message": "Running Drain parser…"})
        await asyncio.sleep(0)

        drain = Drain(
            sim_threshold=drain_sim,
            depth=drain_depth,
            auto_tune_threshold=True,
        )
        drain.load_masks(masks_dicts)

        # Parse in small batches so we can stream progress
        all_records = []
        batch_size  = max(1, len(logs) // 20)
        for i in range(0, len(logs), batch_size):
            batch   = logs[i : i + batch_size]
            records = await asyncio.get_event_loop().run_in_executor(
                None, lambda b=batch: parse_to_records(b, drain, run_id)
            )
            all_records.extend(records)

            # Stream a sample of newly parsed records to the UI
            sample = [
                r for r in records if r["parsed"] == 1
            ][:3]
            _push_event(run_id, "records_batch", {
                "progress": min(100, int((i + batch_size) / len(logs) * 100)),
                "parsed_so_far": sum(r["parsed"] for r in all_records),
                "total_so_far": len(all_records),
                "sample": [
                    {
                        "raw":      r["raw"][:120],
                        "template": r["template"][:120],
                        "variables": json.loads(r["variables"])
                            if isinstance(r["variables"], str) else r["variables"],
                        "var_count": r["var_count"],
                    }
                    for r in sample
                ],
            })
            await asyncio.sleep(0.05)  # yield to event loop

        # ── Stage 5: Validate ────────────────────────────────────────────────
        _push_event(run_id, "stage", {"stage": "validate", "message": "Validating output schema…"})
        valid_records, invalid_records = validate_records(all_records)

        # ── Stage 6: Metrics ─────────────────────────────────────────────────
        parsed_count = sum(r["parsed"] for r in valid_records)
        parse_rate   = round(parsed_count / max(len(valid_records), 1), 4)
        eval_metrics = evaluate_records(valid_records)

        run_meta = {
            "run_id":        run_id,
            "created_utc":   datetime.now(timezone.utc).isoformat(),
            "seed":          seed,
            "llm_provider":  llm_provider,
            "mask_count":    len(masks_dicts),
            "log_count":     len(valid_records),
            "parsed_count":  parsed_count,
            "parse_rate":    parse_rate,
            "invalid_count": len(invalid_records),
            "notes":         f"input={file_path.name} fp={drain.template_fingerprint()}",
        }

        # ── Stage 7: Write artifacts ─────────────────────────────────────────
        _push_event(run_id, "stage", {"stage": "write", "message": "Writing artifacts…"})
        write_sqlite(valid_records, OUTPUT_DIR / "parsed_logs.db", run_meta, invalid_records)
        write_csv(valid_records,    OUTPUT_DIR / f"{run_id}.csv")
        write_json(valid_records,   OUTPUT_DIR / f"{run_id}.json")
        write_cluster_report(drain, OUTPUT_DIR, run_id)

        elapsed = round(time.monotonic() - t_start, 2)

        # ── Final summary event ──────────────────────────────────────────────
        clusters = drain.cluster_summary()
        summary  = {
            "run_id":            run_id,
            "file":              file_path.name,
            "elapsed_s":         elapsed,
            "log_count":         len(valid_records),
            "parsed_count":      parsed_count,
            "parse_rate":        parse_rate,
            "unique_templates":  eval_metrics.unique_templates,
            "avg_wildcard_ratio":round(eval_metrics.avg_wildcard_ratio, 3),
            "mask_count":        len(masks_dicts),
            "llm_calls":         tel.summary()["llm_calls"],
            "llm_latency_s":     tel.summary()["total_latency_s"],
            "quarantined":       len(invalid_records),
            "drain_fingerprint": drain.template_fingerprint(),
            "top_clusters": [
                {"template": c["template"][:100], "size": c["size"], "stability": c["stability"]}
                for c in clusters[:10]
            ],
        }
        _run_cache[run_id] = summary
        _push_event(run_id, "done", summary)

    except Exception as exc:
        log.error("Pipeline error for run %s: %s", run_id, exc, exc_info=True)
        _push_event(run_id, "error", {"message": str(exc)})
    finally:
        # Signal the SSE stream to close
        q = _event_queues.get(run_id)
        if q:
            q.put_nowait(None)  # sentinel


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the live demo dashboard."""
    html_path = STATIC_DIR / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>DeepParse v2</h1><p>Place dashboard.html in /static/</p>")


@app.post("/api/parse")
async def parse_file(
    file:               UploadFile  = File(...),
    llm_provider:       str         = Query(default="groq"),
    adaptive_threshold: float       = Query(default=0.80),
    adaptive_rounds:    int         = Query(default=3),
    drain_sim:          float       = Query(default=0.3),
    drain_depth:        int         = Query(default=3),
    max_logs:           int         = Query(default=1000),
    seed:               int         = Query(default=42),
):
    """
    Upload a log file and start parsing.
    Returns immediately with a run_id — stream progress via GET /api/stream/{run_id}
    """
    content = await file.read()
    _check_file(file.filename, len(content))

    # Injection guard on filename itself
    safe_name = sanitize_for_llm(file.filename or "upload.log", max_len=120)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Filename contains unsafe content")

    run_id    = f"run_{Path(safe_name).stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{seed}"
    file_path = UPLOAD_DIR / f"{run_id}{Path(safe_name).suffix}"
    file_path.write_bytes(content)

    # Create SSE queue for this run
    _event_queues[run_id] = asyncio.Queue(maxsize=500)

    # Launch pipeline as background task
    asyncio.create_task(
        _run_pipeline(
            run_id=run_id,
            file_path=file_path,
            llm_provider=llm_provider,
            adaptive_threshold=adaptive_threshold,
            adaptive_rounds=adaptive_rounds,
            drain_sim=drain_sim,
            drain_depth=drain_depth,
            max_logs=max_logs,
            seed=seed,
        )
    )

    return {
        "run_id":   run_id,
        "file":     safe_name,
        "size_kb":  round(len(content) / 1024, 1),
        "stream":   f"/api/stream/{run_id}",
        "results":  f"/api/runs/{run_id}",
    }

@app.post("/api/parse-batch")
async def parse_batch(
    files:              list[UploadFile] = File(...),
    llm_provider:       str              = Query(default="groq"),
    adaptive_threshold: float            = Query(default=0.80),
    adaptive_rounds:    int              = Query(default=3),
    drain_sim:          float            = Query(default=0.3),
    drain_depth:        int              = Query(default=3),
    max_logs:           int              = Query(default=1000),
    seed:               int              = Query(default=42),
):
    """
    Upload multiple log files and start parsing all of them.
    Returns a list of run entries immediately.
    Stream each file's progress via GET /api/stream/{run_id}
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Max 20 files per batch")

    results = []
    for i, file in enumerate(files):
        content = await file.read()
        try:
            _check_file(file.filename, len(content))
        except HTTPException as e:
            results.append({
                "file":    file.filename,
                "error":   e.detail,
                "skipped": True,
            })
            continue

        safe_name = sanitize_for_llm(file.filename or f"upload_{i}.log", max_len=120)
        if not safe_name:
            results.append({"file": file.filename, "error": "Unsafe filename", "skipped": True})
            continue

        # Stagger seeds so runs are distinguishable
        run_seed  = seed + i
        run_id    = (f"run_{Path(safe_name).stem}_"
                     f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_"
                     f"{run_seed}")
        file_path = UPLOAD_DIR / f"{run_id}{Path(safe_name).suffix}"
        file_path.write_bytes(content)

        _event_queues[run_id] = asyncio.Queue(maxsize=500)

        # Small stagger so LLM calls don't all fire simultaneously
        asyncio.create_task(
            _staggered_pipeline(
                delay_s=i * 0.5,
                run_id=run_id,
                file_path=file_path,
                llm_provider=llm_provider,
                adaptive_threshold=adaptive_threshold,
                adaptive_rounds=adaptive_rounds,
                drain_sim=drain_sim,
                drain_depth=drain_depth,
                max_logs=max_logs,
                seed=run_seed,
            )
        )

        results.append({
            "run_id":  run_id,
            "file":    safe_name,
            "size_kb": round(len(content) / 1024, 1),
            "stream":  f"/api/stream/{run_id}",
            "results": f"/api/runs/{run_id}",
            "skipped": False,
        })

    return {
        "batch_size": len(files),
        "queued":     sum(1 for r in results if not r.get("skipped")),
        "skipped":    sum(1 for r in results if r.get("skipped")),
        "runs":       results,
    }


async def _staggered_pipeline(delay_s: float, **kwargs) -> None:
    """Wrapper that adds a small delay before starting a pipeline task."""
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    await _run_pipeline(**kwargs)


@app.get("/api/stream/{run_id}")
async def stream_events(run_id: str):
    """
    Server-Sent Events stream for live parse progress.
    Connect from JS: const es = new EventSource('/api/stream/RUN_ID')
    """
    q = _event_queues.get(run_id)
    if q is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps({'type': 'connected', 'run_id': run_id})}\n\n"
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
                if event is None:
                    yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
                    break
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering":"no",
            "Connection":       "keep-alive",
        },
    )


@app.get("/api/runs")
async def list_runs(limit: int = Query(default=20)):
    """List all runs from SQLite with summary metrics."""
    db = OUTPUT_DIR / "parsed_logs.db"
    if not db.exists():
        return {"runs": []}
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT run_id, created_utc, llm_provider, log_count,
                  parsed_count, parse_rate, mask_count, invalid_count, notes
           FROM run_meta
           ORDER BY created_utc DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    con.close()
    return {"runs": [dict(r) for r in rows]}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Full detail for a single run including cluster summary."""
    # Check cache first
    if run_id in _run_cache:
        return _run_cache[run_id]

    db = OUTPUT_DIR / "parsed_logs.db"
    if not db.exists():
        raise HTTPException(status_code=404, detail="No runs found")
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT * FROM run_meta WHERE run_id = ?", (run_id,)
    ).fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    result = dict(row)

    # Attach cluster report if available
    clust_file = OUTPUT_DIR / f"{run_id}_clusters.json"
    if clust_file.exists():
        clust_data = json.loads(clust_file.read_text(encoding="utf-8"))
        result["top_clusters"]       = clust_data.get("top_clusters", [])[:10]
        result["drain_fingerprint"]  = clust_data.get("template_fingerprint", "")
        result["total_clusters"]     = clust_data.get("total_clusters", 0)

    return result


@app.get("/api/runs/{run_id}/records")
async def get_records(
    run_id: str,
    page:   int = Query(default=1,  ge=1),
    size:   int = Query(default=50, ge=1, le=500),
    parsed_only: bool = Query(default=False),
):
    """Paginated records for a run."""
    db = OUTPUT_DIR / "parsed_logs.db"
    if not db.exists():
        raise HTTPException(status_code=404, detail="No runs found")
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row

    where  = "WHERE run_id = ?" + (" AND parsed = 1" if parsed_only else "")
    total  = con.execute(f"SELECT COUNT(*) FROM parsed_logs {where}", (run_id,)).fetchone()[0]
    rows   = con.execute(
        f"SELECT * FROM parsed_logs {where} ORDER BY id LIMIT ? OFFSET ?",
        (run_id, size, (page - 1) * size),
    ).fetchall()
    con.close()

    records = []
    for r in rows:
        row = dict(r)
        try:
            row["variables"] = json.loads(row["variables"])
        except Exception:
            row["variables"] = {}
        records.append(row)

    return {
        "run_id":  run_id,
        "total":   total,
        "page":    page,
        "size":    size,
        "pages":   (total + size - 1) // size,
        "records": records,
    }


@app.get("/api/runs/{run_id}/clusters")
async def get_clusters(run_id: str):
    """Template cluster analytics for a run."""
    clust_file = OUTPUT_DIR / f"{run_id}_clusters.json"
    if not clust_file.exists():
        raise HTTPException(status_code=404, detail="Cluster report not found")
    return json.loads(clust_file.read_text(encoding="utf-8"))


@app.get("/api/runs/{run_id}/download/{fmt}")
async def download_artifact(run_id: str, fmt: str):
    """Download run artifact: csv or json."""
    if fmt not in ("csv", "json"):
        raise HTTPException(status_code=400, detail="fmt must be csv or json")
    path = OUTPUT_DIR / f"{run_id}.{fmt}"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{fmt} artifact not found")
    return FileResponse(
        path,
        filename=path.name,
        media_type="text/csv" if fmt == "csv" else "application/json",
    )


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    """Remove a run's artifacts."""
    _run_cache.pop(run_id, None)
    _event_queues.pop(run_id, None)
    removed = []
    for pattern in [f"{run_id}.csv", f"{run_id}.json",
                    f"{run_id}_clusters.json", f"{run_id}_drain_state.json",
                    f"{run_id}_telemetry.json"]:
        p = OUTPUT_DIR / pattern
        if p.exists():
            p.unlink()
            removed.append(pattern)
    return {"deleted": removed}


@app.get("/api/health")
async def health():
    """System health, config, and operational readiness check."""
    db          = OUTPUT_DIR / "parsed_logs.db"
    total_runs  = 0
    total_logs  = 0
    avg_rate    = 0.0
    if db.exists():
        con        = sqlite3.connect(db)
        total_runs = con.execute("SELECT COUNT(*) FROM run_meta").fetchone()[0]
        total_logs = con.execute("SELECT COUNT(*) FROM parsed_logs").fetchone()[0]
        avg_row    = con.execute("SELECT AVG(parse_rate) FROM run_meta").fetchone()[0]
        avg_rate   = round(avg_row or 0.0, 3)
        con.close()

    providers = {
        "groq":      bool(os.environ.get("GROQ_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "gemini":    bool(os.environ.get("GEMINI_API_KEY")),
        "openai":    bool(os.environ.get("OPENAI_API_KEY")),
    }

    return {
        "status":          "ok",
        "version":         "2.0.0",
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "providers_ready": providers,
        "active_runs":     len(_event_queues),
        "db_stats": {
            "total_runs":   total_runs,
            "total_logs":   total_logs,
            "avg_parse_rate": avg_rate,
        },
        "storage": {
            "output_dir": str(OUTPUT_DIR),
            "upload_dir": str(UPLOAD_DIR),
        },
    }


@app.get("/api/domain-context")
async def domain_context():
    """Return the grounding context injected into LLM prompts."""
    return {
        "context": FAB_DOMAIN_CONTEXT,
        "description": "Domain schema injected into every LLM mask synthesis prompt (RAG-lite grounding)",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  DeepParse v2 — Live Demo Server")
    print("="*60)
    print("  Dashboard  → http://localhost:8000")
    print("  API docs   → http://localhost:8000/docs")
    print("  Health     → http://localhost:8000/api/health")
    print("="*60 + "\n")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)