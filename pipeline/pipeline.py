"""
pipeline/pipeline.py — Core run() function
Orchestrates all stages: load → normalise → masks → drain → validate → write
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from DeepParse.deepparse import Drain
from DeepParse.deepparse.evaluation.eval_runner import evaluate_records
from pipeline.config import (
    ADAPTIVE_MAX_RETRIES,
    ADAPTIVE_PARSE_THRESHOLD,
    SOURCE_SIM_THRESHOLDS,
)
from pipeline.loaders import load_logs, normalize
from pipeline.masks import synthesize_masks_adaptive
from pipeline.parser import (
    health_check,
    parse_to_records,
    teach_back_to_drain,
    validate_records,
)
from pipeline.telemetry import TELEMETRY, Telemetry, set_global_seed
from pipeline.writers import (
    write_cluster_report,
    write_csv,
    write_json,
    write_sqlite,
)

log = logging.getLogger("deepparse.pipeline")


def run(
    mode: str,
    root_dir: Path,
    *,
    seed: int                    = 42,
    output_dir: Path | None      = None,
    input_path: str | None       = None,
    input_fmt: str               = "auto",
    message_col: str             = "message",
    llm_provider: str            = "groq",
    use_mask_cache: bool         = True,
    max_length: int              = 256,
    max_logs: int                = 1000,
    config: str | None           = None,
    static_masks_path: str | None = None,
    adaptive_threshold: float    = ADAPTIVE_PARSE_THRESHOLD,
    adaptive_rounds: int         = ADAPTIVE_MAX_RETRIES,
    drain_sim_threshold: float   = 0.5,
    drain_depth: int             = 4,
    save_drain_state: bool       = True,
) -> dict:
    """
    Full pipeline execution. Returns a summary dict.
    Stages:
      1. Load      — multi-format file ingestion
      2. Normalise — dedup, length cap, injection guard
      3. Masks     — static + LLM adaptive synthesis
      4. Drain     — prefix-tree clustering with pre-masking
      5. Validate  — JSON-Schema guardrail + quarantine
      6. Write     — SQLite + CSV + JSON + cluster report
    """
    set_global_seed(seed)

    if output_dir is None:
        output_dir = root_dir / "artifacts" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── eval mode ────────────────────────────────────────────────────────────
    if mode == "eval":
        from DeepParse.deepparse.tools.fetch_loghub import download_logs
        from DeepParse.deepparse.evaluation.eval_runner import EvaluationRunner
        out_folder  = root_dir / "artifacts" / "data"
        download_logs(out=out_folder)
        eval_config = config or str(root_dir / "eval.yaml")
        if not Path(eval_config).exists():
            Path(eval_config).write_text(
                "output_dir: artifacts/output\neval_dir: artifacts/eval\n"
                "min_parse_rate: 0.60\nmin_ga: 0.50\npass_on_no_gt: true\n",
                encoding="utf-8",
            )
        EvaluationRunner(Path(eval_config)).run()
        return {}

    if not input_path:
        print("No --input file supplied.")
        return {}

    # ── test / parse mode ────────────────────────────────────────────────────
    t_start   = time.monotonic()
    file_stem = Path(input_path).stem
    run_id    = (f"run_{file_stem}_"
                 f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{seed}")
    TELEMETRY.record("pipeline_start", run_id=run_id, input=input_path)

    # 1 — Load
    raw_logs = load_logs(input_path, fmt=input_fmt,
                         message_col=message_col, max_length=max_length)
    TELEMETRY.record("load", lines=len(raw_logs))

    # 2 — Normalise
    logs = normalize(raw_logs, max_length=max_length,
                     max_logs=max_logs, sanitize=True)

    # 3 — Adaptive mask synthesis
    mask_cache  = output_dir / f"masks_{file_stem}.json"
    masks_dicts = synthesize_masks_adaptive(
        logs=logs,
        llm_provider=llm_provider,
        max_length=max_length,
        mask_cache_path=mask_cache,
        use_cache=use_mask_cache,
        static_masks_path=Path(static_masks_path) if static_masks_path else None,
        telemetry=TELEMETRY,
        max_rounds=adaptive_rounds,
        threshold=adaptive_threshold,
    )
    TELEMETRY.record("masks_ready", count=len(masks_dicts))

    # 4 — Drain parse (per-source sim_threshold, doc section 1.4)
    _ext         = Path(input_path).suffix.lower()
    effective_sim = min(drain_sim_threshold,
                        SOURCE_SIM_THRESHOLDS.get(_ext, drain_sim_threshold))
    log.info("sim_threshold: %s → %.2f", _ext, effective_sim)

    drain = Drain(sim_threshold=effective_sim, depth=drain_depth,
                  auto_tune_threshold=True)
    drain.load_masks(masks_dicts)
    records = parse_to_records(logs, drain, run_id)

    # Loop 2 — teach resolved templates back to Drain cache
    teach_back_to_drain(drain, records, masks_dicts)

    # Register into RAG template cache for next run (doc section 3.1)
    try:
        from DeepParse.deepparse.synth.hf_deepseek_r1 import register_resolved_template
        for r in records:
            if r["parsed"] == 1:
                register_resolved_template(r["template"])
    except Exception:
        pass

    if save_drain_state:
        drain.save_state(output_dir / f"{run_id}_drain_state.json")

    # 5 — Schema validation / quarantine
    valid_records, invalid_records = validate_records(records)

    # 6 — Build run metadata
    parsed_count = sum(r["parsed"] for r in valid_records)
    run_meta = {
        "run_id":        run_id,
        "created_utc":   datetime.now(timezone.utc).isoformat(),
        "seed":          seed,
        "llm_provider":  llm_provider,
        "mask_count":    len(masks_dicts),
        "log_count":     len(valid_records),
        "parsed_count":  parsed_count,
        "parse_rate":    round(parsed_count / max(len(valid_records), 1), 4),
        "invalid_count": len(invalid_records),
        "notes":         f"input={input_path} fp={drain.template_fingerprint()}",
    }

    # Health check
    health_check(valid_records, run_meta)

    # Eval metrics
    eval_metrics = evaluate_records(valid_records)
    TELEMETRY.record("eval_metrics",
                     parse_rate=eval_metrics.parse_rate,
                     unique_templates=eval_metrics.unique_templates)

    # 7 — Write artifacts
    db_file    = write_sqlite(valid_records, output_dir / "parsed_logs.db",
                               run_meta, invalid_records)
    csv_file   = write_csv(valid_records,    output_dir / f"{run_id}.csv")
    js_file    = write_json(valid_records,   output_dir / f"{run_id}.json")
    clust_file = write_cluster_report(drain, output_dir, run_id)
    tel_file   = output_dir / f"{run_id}_telemetry.json"

    # ── Compute summary FIRST before any downstream use ──────────────────────
    elapsed=time.monotonic() - t_start
    tel_summary = TELEMETRY.summary()

    TELEMETRY.record("pipeline_done", elapsed_s=round(elapsed, 3))
    TELEMETRY.write(tel_file)

    # Ship metrics to Loki/Grafana (non-fatal if Loki is down)
    try:
        from pipeline.loki_logger import push_run_metrics
        push_run_metrics(run_meta, eval_metrics, tel_summary)
    except Exception:
        pass

        # Optional PostgreSQL write
    pg_url = os.environ.get("POSTGRES_URL")
    if pg_url:
        try:
            from pipeline.writers import write_postgres
            pg_records = [
                {**r, "variables": r["variables"]
                if isinstance(r["variables"], str)
                else json.dumps(r["variables"])}
                for r in valid_records
            ]
            write_postgres(pg_records, pg_url, run_meta, invalid_records)
        except Exception as exc:
            log.error("PostgreSQL write failed (non-fatal): %s", exc)

    n_fab = len([m for m in masks_dicts if any(
        tok in m.get("mask_with", "")
        for tok in ("LOT", "EQP", "WFR", "CJOB", "PRJOB", "RCP", "EC_ID")
    )])

    print(f"\n{'=' * 62}")
    print(f"  DeepParse v2  |  {run_id}")
    print(f"{'=' * 62}")
    print(f"  Logs processed     : {len(valid_records):>6,}  ({len(invalid_records)} quarantined)")
    print(f"  Templates found    : {eval_metrics.unique_templates:>6,}")
    print(f"  Parse rate         : {run_meta['parse_rate'] * 100:>6.1f}%")
    print(f"  Avg wildcard ratio : {eval_metrics.avg_wildcard_ratio:>6.3f}")
    print(f"  Masks used         : {len(masks_dicts):>6,}  ({n_fab} fab-specific)")
    print(f"  Drain clusters     : {len(drain.get_clusters()):>6,}  fp={drain.template_fingerprint()}")
    print(f"  LLM calls          : {tel_summary['llm_calls']:>6,}")
    print(f"  Elapsed            : {elapsed:>6.2f}s")
    print(f"\n  Artifacts → '{output_dir}/'")
    print(f"    [SQLite   ] {db_file.name}")
    print(f"    [CSV      ] {csv_file.name}")
    print(f"    [JSON     ] {js_file.name}")
    print(f"    [Clusters ] {clust_file.name}")
    print(f"    [Telemetry] {tel_file.name}")

    return {
        "run_id": run_id,
        "parse_rate": run_meta["parse_rate"],
        "templates": eval_metrics.unique_templates,
        "masks": len(masks_dicts),
        "elapsed_s": elapsed,
        "output_dir": str(output_dir),
    }