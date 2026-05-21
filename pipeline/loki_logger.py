"""
pipeline/loki_logger.py — Ships structured telemetry to Loki
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from datetime import datetime, timezone

log = logging.getLogger("deepparse.loki")

LOKI_URL = "http://localhost:3100/loki/api/v1/push"


def _ns_timestamp() -> str:
    """Current time as nanosecond Unix timestamp string (Loki requirement)."""
    return str(int(time.time() * 1e9))


def push_to_loki(
    message: str,
    labels: dict[str, str],
    level: str = "info",
) -> None:
    """
    Push a single log line to Loki.
    Labels become filterable dimensions in Grafana.
    """
    payload = {
        "streams": [
            {
                "stream": {
                    "app":   "deepparse",
                    "level": level,
                    **labels,
                },
                "values": [[_ns_timestamp(), message]],
            }
        ]
    }
    try:
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            LOKI_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2):
            pass
    except Exception as exc:
        # Never crash the pipeline if Loki is unavailable
        log.debug("Loki push failed (non-fatal): %s", exc)


def push_run_metrics(run_meta: dict, eval_metrics: object, tel_summary: dict) -> None:
    """Push end-of-run metrics to Loki as a structured JSON line."""
    message = json.dumps({
        "event":            "run_complete",
        "run_id":           run_meta.get("run_id", ""),
        "parse_rate":       run_meta.get("parse_rate", 0),
        "log_count":        run_meta.get("log_count", 0),
        "parsed_count":     run_meta.get("parsed_count", 0),
        "mask_count":       run_meta.get("mask_count", 0),
        "llm_provider":     run_meta.get("llm_provider", ""),
        "llm_calls":        tel_summary.get("llm_calls", 0),
        "llm_latency_s":    tel_summary.get("total_latency_s", 0),
        "unique_templates": getattr(eval_metrics, "unique_templates", 0),
        "avg_wildcard":     getattr(eval_metrics, "avg_wildcard_ratio", 0),
        "quarantined":      run_meta.get("invalid_count", 0),
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    })
    push_to_loki(
        message=message,
        labels={
            "run_id":       run_meta.get("run_id", "")[-20:],
            "llm_provider": run_meta.get("llm_provider", "unknown"),
        },
        level="info",
    )


def push_parse_rate_probe(round_n: int, rate: float, threshold: float) -> None:
    """Push adaptive loop probe events — shows feedback loop in Grafana."""
    level   = "info" if rate >= threshold else "warn"
    message = json.dumps({
        "event":     "parse_rate_probe",
        "round":     round_n,
        "rate":      round(rate, 4),
        "threshold": threshold,
        "passed":    rate >= threshold,
    })
    push_to_loki(message=message, labels={"event": "probe"}, level=level)


def push_llm_call(provider: str, latency_s: float, mask_count: int) -> None:
    """Push LLM call metrics for cost/latency dashboard."""
    message = json.dumps({
        "event":      "llm_call",
        "provider":   provider,
        "latency_s":  latency_s,
        "mask_count": mask_count,
    })
    push_to_loki(message=message, labels={"event": "llm"}, level="info")


def push_injection_blocked(snippet: str) -> None:
    """Push security alert when injection is detected."""
    message = json.dumps({
        "event":   "injection_blocked",
        "snippet": snippet[:80],
    })
    push_to_loki(message=message, labels={"event": "security"}, level="warn")


def push_schema_drift(column: str, action: str, table: str) -> None:
    """Push schema drift alert when new columns are detected."""
    message = json.dumps({
        "event":  "schema_drift",
        "column": column,
        "action": action,
        "table":  table,
    })
    push_to_loki(message=message, labels={"event": "schema_drift"}, level="warn")