"""Is a stream's anomaly model going stale?

Training keeps a histogram of how much of each held-out (normal) window the model missed. Live,
every scored window's miss ratio goes into a rolling histogram per stream; the population
stability index (PSI) between the two says how far today's traffic has moved from what the
model learned. Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 retrain.
"""

from __future__ import annotations

import math
import threading
from collections import deque

import numpy as np

from ..settings import get_settings

BINS = np.linspace(0.0, 1.0, 11)  # miss ratio deciles
EPS = 1e-4


def histogram(values) -> list[float]:
    counts, _ = np.histogram(np.clip(np.asarray(values, dtype=float), 0, 1), bins=BINS)
    total = counts.sum()
    return [round(float(c) / total, 5) for c in counts] if total else [0.0] * (len(BINS) - 1)


def psi(expected: list[float], actual: list[float]) -> float:
    return float(sum((a - e) * math.log((a + EPS) / (e + EPS)) for e, a in zip(expected, actual)))


class Drift:
    def __init__(self):
        self._lock = threading.Lock()
        self._live: dict[str, deque[tuple[float, bool]]] = {}  # stream -> (miss ratio, alerted)

    def observe(self, stream_id: str, scored) -> None:
        """``scored``: the detector's ScoredWindows since the last call."""
        with self._lock:
            live = self._live.setdefault(stream_id, deque(maxlen=get_settings().server.drift_keep))
            live.extend((w.score.miss_ratio, bool(w.reasons)) for w in scored)

    def reset(self, stream_id: str) -> None:
        """A new model: its baseline is new, so is the comparison."""
        with self._lock:
            self._live.pop(stream_id, None)

    def report(self, streams: list[dict]) -> list[dict]:
        out = []
        for s in streams:
            with self._lock:
                live = list(self._live.get(s["id"], ()))
            baseline = s.get("miss_baseline")
            row = {
                "id": s["id"],
                "name": s.get("name", s["id"]),
                "live_windows": len(live),
                "has_baseline": bool(baseline),
                "trained_at": s.get("trained_at"),
                "psi": None,
                "alert_pct": None,
                "live": None,
                "baseline": baseline,
            }
            if live:
                row["alert_pct"] = round(100 * sum(a for _, a in live) / len(live), 2)
                row["live"] = histogram([m for m, _ in live])
                if baseline and len(live) >= get_settings().server.drift_min_live:
                    row["psi"] = round(psi(baseline, row["live"]), 4)
            out.append(row)
        return out
