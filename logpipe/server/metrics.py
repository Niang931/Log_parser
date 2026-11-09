"""The server's own metrics, instead of Prometheus and Grafana.

The hot path bumps counters; every ``every`` seconds :meth:`Metrics.sample` turns them into
one value per metric (rates, latency percentiles, process and host load, gauges the server
passes in) and writes the sample to a :class:`MemorySeries` (memory store) or to ClickHouse's
``server_metrics`` table (kept 30 days). The System page charts :meth:`Metrics.series`.
"""
from __future__ import annotations

import os
import random
import threading
import time
from collections import deque

import numpy as np

from ..settings import get_settings

# name -> (label, unit, group); the UI draws one chart per metric, grouped.
METRICS: dict[str, tuple[str, str, str]] = {
    "lines_per_s": ("Lines processed", "lines/s", "Throughput"),
    "parse_p50_us": ("Parse latency p50", "µs", "Throughput"),
    "parse_p95_us": ("Parse latency p95", "µs", "Throughput"),
    "parse_p99_us": ("Parse latency p99", "µs", "Throughput"),
    "unknown_pct": ("Lines no template matched", "%", "Throughput"),
    "errors_per_s": ("Lines that failed to process", "lines/s", "Throughput"),
    "inbox": ("Intake buffer", "events", "Learning loop"),
    "pending_rows": ("Rows waiting to be learned (PENDING)", "rows", "Learning loop"),
    "queue_lag": ("Unresolved queue lag", "messages", "Learning loop"),
    "llm_calls_per_min": ("LLM calls", "calls/min", "LLM"),
    "llm_errors_per_min": ("LLM errors", "errors/min", "LLM"),
    "llm_latency_s": ("LLM call duration (mean)", "s", "LLM"),
    "alerts_per_min": ("Anomaly alerts", "alerts/min", "Detector"),
    "cpu_pct": ("Server CPU (100% = one core)", "%", "Process"),
    "rss_mb": ("Server memory (RSS)", "MB", "Process"),
    "threads": ("Server threads", "threads", "Process"),
    "host_cpu_pct": ("Host CPU (all cores)", "%", "Host"),
    "host_mem_pct": ("Host memory used", "%", "Host"),
    "score_psi_max": ("Anomaly-score drift, worst stream (PSI)", "PSI", "Drift"),
    "alert_window_pct": ("Windows alerted, worst stream", "%", "Drift"),
    "new_templates_per_hour": ("New templates learned", "templates/h", "Drift"),
    "store_mb": ("Log store on disk", "MB", "Storage"),
    "disk_used_pct": ("Store disk used", "%", "Storage"),
}


def _host_cpu() -> tuple[float, float] | None:
    """(busy, total) jiffies from /proc/stat; None off Linux."""
    try:
        with open("/proc/stat") as f:
            vals = [float(v) for v in f.readline().split()[1:]]
    except (OSError, ValueError):
        return None
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return sum(vals) - idle, sum(vals)


def _host_mem_pct() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            info = {line.split(":")[0]: float(line.split()[1]) for line in f}
        return round(100 * (1 - info["MemAvailable"] / info["MemTotal"]), 1)
    except (OSError, KeyError, ValueError, ZeroDivisionError):
        return None


def _rss_mb() -> float:
    try:
        with open("/proc/self/statm") as f:
            return round(int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20, 1)
    except (OSError, ValueError, IndexError):
        import resource  # peak, not current, but better than nothing off Linux

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


class MemorySeries:
    """Samples in process memory (the memory store's history; 24 h at 10 s)."""

    def __init__(self, keep: int = 8640):
        self.rows: deque[tuple[float, dict]] = deque(maxlen=keep)
        self._lock = threading.Lock()

    def write(self, ts: float, values: dict[str, float]) -> None:
        with self._lock:
            self.rows.append((ts, values))

    def query(self, minutes: int, step: int) -> dict[str, list[list[float]]]:
        since = time.time() - minutes * 60
        buckets: dict[str, dict[float, list[float]]] = {}
        with self._lock:
            rows = [r for r in self.rows if r[0] >= since]
        for ts, values in rows:
            b = ts // step * step
            for name, v in values.items():
                buckets.setdefault(name, {}).setdefault(b, []).append(v)
        return {name: [[b, round(sum(vs) / len(vs), 3)] for b, vs in sorted(by.items())]
                for name, by in buckets.items()}


class ClickHouseSeries:
    """``server_metrics``: one row per metric per sample, dropped after 30 days (TTL)."""

    def __init__(self, client):
        self.client = client
        self.client.command(
            "CREATE TABLE IF NOT EXISTS server_metrics (ts DateTime64(3, 'UTC'), "
            "name LowCardinality(String), value Float64) ENGINE = MergeTree ORDER BY (name, ts) "
            "TTL toDateTime(ts) + INTERVAL 30 DAY")

    def write(self, ts: float, values: dict[str, float]) -> None:
        from datetime import UTC, datetime

        when = datetime.fromtimestamp(ts, UTC)
        self.client.insert("server_metrics", [[when, k, float(v)] for k, v in values.items()],
                           column_names=["ts", "name", "value"])

    def query(self, minutes: int, step: int) -> dict[str, list[list[float]]]:
        rows = self.client.query(
            "SELECT name, toUnixTimestamp(toStartOfInterval(ts, toIntervalSecond({step:UInt32}))) AS b, "
            "round(avg(value), 3) FROM server_metrics WHERE ts >= now() - toIntervalMinute({m:UInt32}) "
            "GROUP BY name, b ORDER BY name, b", parameters={"step": step, "m": minutes}).result_rows
        out: dict[str, list[list[float]]] = {}
        for name, b, v in rows:
            out.setdefault(name, []).append([float(b), v])
        return out


class MongoSeries:
    """``server_metrics``: a time-series collection (metric name = metaField), dropped after 30 days."""

    TTL_SECONDS = 30 * 86_400

    def __init__(self, db):
        if "server_metrics" not in db.list_collection_names():
            db.create_collection("server_metrics", expireAfterSeconds=self.TTL_SECONDS,
                                 timeseries={"timeField": "ts", "metaField": "name", "granularity": "seconds"})
        self.col = db["server_metrics"]

    def write(self, ts: float, values: dict[str, float]) -> None:
        from datetime import UTC, datetime

        when = datetime.fromtimestamp(ts, UTC)
        self.col.insert_many([{"ts": when, "name": k, "value": float(v)} for k, v in values.items()])

    def query(self, minutes: int, step: int) -> dict[str, list[list[float]]]:
        from datetime import UTC, datetime, timedelta

        ms = {"$toLong": "$ts"}
        rows = self.col.aggregate([
            {"$match": {"ts": {"$gte": datetime.now(UTC) - timedelta(minutes=minutes)}}},
            {"$group": {"_id": {"name": "$name", "b": {"$subtract": [ms, {"$mod": [ms, step * 1000]}]}},
                        "v": {"$avg": "$value"}}},
            {"$sort": {"_id.name": 1, "_id.b": 1}}])
        out: dict[str, list[list[float]]] = {}
        for r in rows:
            out.setdefault(r["_id"]["name"], []).append([r["_id"]["b"] / 1000, round(r["v"], 3)])
        return out


class Metrics:
    def __init__(self, sink: MemorySeries | ClickHouseSeries | MongoSeries | None = None, every: float = 10):
        self.sink = sink or MemorySeries()
        self.every = every
        self._lock = threading.Lock()
        self._rng = random.Random(0)
        self._counts = {"lines": 0, "unknown": 0, "errors": 0, "alerts": 0,
                        "llm_calls": 0, "llm_errors": 0}
        self._llm_seconds = 0.0
        self._lat: list[int] = []
        self._seen = 0  # latencies offered this sample (for the reservoir)
        self._cap = get_settings().server.metrics_latency_cap
        self._at = time.monotonic()
        self._cpu = sum(os.times()[:2])
        self._host = _host_cpu()
        self.current: dict[str, float] = {}
        self.sampled_at: float | None = None

    def parsed(self, ns: int, unknown: bool, alerts: int = 0) -> None:
        with self._lock:
            self._counts["lines"] += 1
            self._counts["unknown"] += unknown
            self._counts["alerts"] += alerts
            self._seen += 1
            if len(self._lat) < self._cap:
                self._lat.append(ns)
            elif (j := self._rng.randrange(self._seen)) < self._cap:
                self._lat[j] = ns

    def count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counts[name] += n

    def llm(self, seconds: float, ok: bool) -> None:
        with self._lock:
            self._counts["llm_calls"] += 1
            self._counts["llm_errors"] += not ok
            self._llm_seconds += seconds

    def sample(self, gauges: dict[str, float | None] | None = None) -> dict[str, float]:
        """Rates since the last sample, plus ``gauges``; written to the sink and kept as
        :attr:`current`."""
        now, cpu, host = time.monotonic(), sum(os.times()[:2]), _host_cpu()
        with self._lock:
            counts, lat, llm_s = dict(self._counts), self._lat, self._llm_seconds
            self._counts = dict.fromkeys(counts, 0)
            self._lat, self._seen, self._llm_seconds = [], 0, 0.0
            dt, self._at = max(now - self._at, 1e-6), now
            dcpu, self._cpu = cpu - self._cpu, cpu
            prev_host, self._host = self._host, host
        v: dict[str, float | None] = {
            "lines_per_s": counts["lines"] / dt,
            "errors_per_s": counts["errors"] / dt,
            "unknown_pct": 100 * counts["unknown"] / counts["lines"] if counts["lines"] else None,
            "alerts_per_min": 60 * counts["alerts"] / dt,
            "llm_calls_per_min": 60 * counts["llm_calls"] / dt,
            "llm_errors_per_min": 60 * counts["llm_errors"] / dt,
            "llm_latency_s": llm_s / counts["llm_calls"] if counts["llm_calls"] else None,
            "cpu_pct": 100 * dcpu / dt,
            "rss_mb": _rss_mb(),
            "threads": threading.active_count(),
            "host_mem_pct": _host_mem_pct(),
        }
        if lat:
            us = np.asarray(lat) / 1000
            for q in (50, 95, 99):
                v[f"parse_p{q}_us"] = float(np.percentile(us, q))
        if host and prev_host and host[1] > prev_host[1]:
            v["host_cpu_pct"] = 100 * (host[0] - prev_host[0]) / (host[1] - prev_host[1])
        v |= gauges or {}
        values = {k: round(float(x), 3) for k, x in v.items() if x is not None}
        self.current, self.sampled_at = values, time.time()
        self.sink.write(self.sampled_at, values)
        return values

    def series(self, minutes: int) -> dict:
        step = max(int(self.every), minutes * 60 // 240)  # ~240 points per chart
        return {"step": step, "series": self.sink.query(minutes, step)}


class CountedAdapter:
    """Wraps an LLM adapter so every call it answers (or fails) is counted and timed."""

    def __init__(self, inner, metrics: Metrics):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_metrics", metrics)

    def __setattr__(self, name: str, value) -> None:  # e.g. the worker's key_vocab
        setattr(self._inner, name, value)

    def __getattr__(self, name: str):
        attr = getattr(self._inner, name)
        if not callable(attr) or name.startswith("_"):
            return attr

        def counted(*args, **kwargs):
            start, ok = time.monotonic(), False
            try:
                result = attr(*args, **kwargs)
                ok = True
                return result
            finally:
                self._metrics.llm(time.monotonic() - start, ok)

        return counted


# (metric, op, threshold, for_minutes, enabled, label): starting points, editable on the System page
DEFAULT_RULES = (
    ("parse_p99_us", ">", 5000, 5, True, "Parse p99 above 5 ms"),
    ("inbox", ">", 25_000, 2, True, "Intake buffer half full (falling behind)"),
    ("pending_rows", ">", 100_000, 15, True, "Large backlog of lines waiting to be learned"),
    ("queue_lag", ">", 10_000, 10, True, "Unresolved queue lagging"),
    ("llm_errors_per_min", ">", 5, 15, True, "LLM keeps failing"),
    ("unknown_pct", ">", 20, 10, True, "Many lines match no template"),
    ("disk_used_pct", ">", 80, 5, True, "Store disk above 80%"),
    ("cpu_pct", ">", 90, 10, True, "Server CPU saturated"),
    ("lines_per_s", "<", 0.1, 15, False, "No logs arriving"),
    ("score_psi_max", ">", 0.25, 30, True, "Anomaly model drifting: retrain it"),
    ("alert_window_pct", ">", 5, 30, True, "Unusually many windows alerted"),
    ("new_templates_per_hour", ">", 50, 30, True, "Log formats changing fast: consider recalibrating"),
)
OPS = {">": lambda v, t: v > t, "<": lambda v, t: v < t}


class MetricRules:
    """Threshold rules on the sampled metrics; a breach that lasts ``for_minutes`` opens a
    ``metric_alert`` (firing) that resolves once the metric is back."""

    def __init__(self, control):
        self.control = control
        self._since: dict[str, float] = {}  # rule id -> when its breach began
        # Each default once (newer versions add theirs), so deleted defaults stay deleted.
        seeded = set((control.get("meta", "metric_rules_seeded") or {}).get("ids", ()))
        for metric, op, threshold, minutes, enabled, label in DEFAULT_RULES:
            if (rid := f"default-{metric}") not in seeded:
                self.save({"metric": metric, "op": op, "threshold": threshold, "for_minutes": minutes,
                           "enabled": enabled, "label": label}, rule_id=rid)
                seeded.add(rid)
        control.put("meta", "metric_rules_seeded", {"ids": sorted(seeded)})

    def rules(self) -> list[dict]:
        return sorted(self.control.all("metric_rule"), key=lambda r: r["id"])

    def save(self, body: dict, rule_id: str | None = None) -> dict:
        old = self.control.get("metric_rule", rule_id) if rule_id else None
        rule = {**(old or {}), **body}
        if rule.get("metric") not in METRICS:
            raise ValueError(f"unknown metric {rule.get('metric')!r}")
        if rule.get("op") not in OPS:
            raise ValueError("op is > or <")
        rule["threshold"] = float(rule["threshold"])
        rule["for_minutes"] = min(max(float(rule.get("for_minutes", 5)), 0), 24 * 60)
        rule["enabled"] = bool(rule.get("enabled", True))
        rule["label"] = str(rule.get("label") or f"{METRICS[rule['metric']][0]} {rule['op']} {rule['threshold']:g}")[:120]
        rule["id"] = rule_id or os.urandom(6).hex()
        self._since.pop(rule["id"], None)
        return self.control.put("metric_rule", rule["id"], rule)

    def delete(self, rule_id: str) -> None:
        if self.control.get("metric_rule", rule_id) is None:
            raise KeyError(rule_id)
        self.control.delete("metric_rule", rule_id)
        self._resolve(rule_id, time.time(), "rule deleted")

    def alerts(self, limit: int = 100) -> list[dict]:
        docs = self.control.all("metric_alert")
        return sorted(docs, key=lambda a: (a["status"] != "firing", -a["started_at"]))[:limit]

    def _firing(self, rule_id: str) -> dict | None:
        return self.control.get("metric_alert", f"{rule_id}:firing")

    def _resolve(self, rule_id: str, now: float, why: str) -> None:
        if (doc := self._firing(rule_id)) is not None:
            self.control.delete("metric_alert", doc["id"])
            done_id = f"{rule_id}:{int(doc['started_at'])}"
            self.control.put("metric_alert", done_id, {**doc, "id": done_id, "status": "resolved",
                                                       "resolved_at": now, "resolution": why})
            old = sorted((a for a in self.control.all("metric_alert") if a["status"] == "resolved"),
                         key=lambda a: a["resolved_at"])
            for a in old[:-200]:  # keep the last 200 resolved
                self.control.delete("metric_alert", a["id"])

    def evaluate(self, values: dict[str, float], now: float | None = None) -> list[dict]:
        """After each sample: returns alerts that started firing."""
        now = time.time() if now is None else now
        fired = []
        for rule in self.rules():
            rid, v = rule["id"], values.get(rule["metric"])
            breach = rule["enabled"] and v is not None and OPS[rule["op"]](v, rule["threshold"])
            if not breach:
                self._since.pop(rid, None)
                self._resolve(rid, now, "back to normal" if rule["enabled"] else "rule disabled")
                continue
            since = self._since.setdefault(rid, now)
            doc = self._firing(rid)
            if doc is not None:
                self.control.update("metric_alert", doc["id"], value=v, peak=max(doc["peak"], v)
                                    if rule["op"] == ">" else min(doc["peak"], v), last_at=now)
            elif now - since >= rule["for_minutes"] * 60:
                doc = {"id": f"{rid}:firing", "rule": rid, "label": rule["label"], "metric": rule["metric"],
                       "op": rule["op"], "threshold": rule["threshold"], "value": v, "peak": v,
                       "started_at": since, "last_at": now, "status": "firing"}
                self.control.put("metric_alert", doc["id"], doc)
                fired.append(doc)
        return fired
