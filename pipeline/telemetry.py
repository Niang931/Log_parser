"""
pipeline/telemetry.py — Structured in-process telemetry collector
"""
from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("deepparse.telemetry")


class Telemetry:
    """Lightweight event collector — latency, token cost, parse metrics."""

    def __init__(self) -> None:
        self._events: list[dict] = []

    def record(self, event: str, **kw) -> None:
        self._events.append({
            "event": event,
            "ts": datetime.now(timezone.utc).isoformat(),
            **kw,
        })

    def summary(self) -> dict:
        llm_calls     = [e for e in self._events if e["event"] == "llm_call"]
        total_latency = sum(e.get("latency_s", 0) for e in llm_calls)
        total_tokens  = sum(e.get("tokens_out", 0) for e in llm_calls)
        return {
            "llm_calls":       len(llm_calls),
            "total_latency_s": round(total_latency, 3),
            "est_tokens_out":  total_tokens,
            "events":          len(self._events),
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"summary": self.summary(), "events": self._events},
                      f, indent=2)


# Global singleton used throughout the pipeline
TELEMETRY = Telemetry()


def set_global_seed(seed: int) -> None:
    """Set all RNG seeds for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    log.info("Global seed → %d", seed)