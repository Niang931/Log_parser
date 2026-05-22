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
    return str(int(time.time() * 1e9))


def push_to_loki(message: str, event_type: str, level: str = "info") -> None:
    payload = {
        "streams": [{
            "stream": {
                "app":        "deepparse",
                "level":      level,
                "event_type": event_type,
            },
            "values": [[_ns_timestamp(), message]],
        }]
    }
    try:
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            LOKI_URL, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2):
            pass
    except Exception as exc:
        log.debug("Loki push failed (non-fatal): %s", exc)


def push_run_metrics(run_meta: dict, eval_metrics: object, tel_summary: dict) -> None:
    message = json.dumps({
        "event":            "run_complete",
        "run_id":           run_meta.get("run_id", ""),
        "parse_rate":       round(run_meta.get("parse_rate", 0), 4),
        "log_count":        run_meta.get("log_count", 0),
        "parsed_count":     run_meta.get("parsed_count", 0),
        "mask_count":       run_meta.get("mask_count", 0),
        "llm_provider":     run_meta.get("llm_provider", ""),
        "llm_calls":        tel_summary.get("llm_calls", 0),
        "llm_latency_s":    tel_summary.get("total_latency_s", 0),
        "unique_templates": getattr(eval_metrics, "unique_templates", 0),
        "avg_wildcard":     round(getattr(eval_metrics, "avg_wildcard_ratio", 0), 4),
        "quarantined":      run_meta.get("invalid_count", 0),
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    })
    push_to_loki(message=message, event_type="run_complete", level="info")


def push_parse_rate_probe(round_n: int, rate: float, threshold: float) -> None:
    message = json.dumps({
        "event":     "parse_rate_probe",
        "round":     round_n,
        "rate":      round(rate, 4),
        "threshold": threshold,
        "passed":    rate >= threshold,
    })
    push_to_loki(message=message, event_type="probe",
                 level="info" if rate >= threshold else "warn")


def push_llm_call(provider: str, latency_s: float, mask_count: int) -> None:
    message = json.dumps({
        "event":      "llm_call",
        "provider":   provider,
        "latency_s":  round(latency_s, 3),
        "mask_count": mask_count,
    })
    push_to_loki(message=message, event_type="llm_call", level="info")


def push_injection_blocked(snippet: str) -> None:
    message = json.dumps({
        "event":   "injection_blocked",
        "snippet": snippet[:80],
    })
    push_to_loki(message=message, event_type="security", level="warn")


def push_schema_drift(column: str, action: str, table: str) -> None:
    message = json.dumps({
        "event":  "schema_drift",
        "column": column,
        "action": action,
        "table":  table,
    })
    push_to_loki(message=message, event_type="schema_drift", level="warn")
