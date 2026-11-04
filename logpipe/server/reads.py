"""What the UI and the learning jobs read from the log store, for both backends."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from ..storage.base import LogRow
from ..storage.memory import MemoryStore

FILTERS = ("host", "service", "source", "status", "template_id")
TOP_FIELDS = {"host": "device_id", "service": "service", "source": "source",
              "template_id": "template_id"}
MAX_QUERY_ROWS = 500


def row_json(r: LogRow) -> dict:
    return {"log_id": str(r.log_id), "timestamp": r.timestamp.isoformat(), "host": r.device_id,
            "service": r.service, "source": r.source, "tags": list(r.tags), "status": str(r.status),
            "template_id": r.template_id, "payload": r.payload, "message": r.raw_message}


class MemoryReads:
    def __init__(self, store: MemoryStore):
        self.store = store

    def _rows(self) -> list[LogRow]:
        with self.store._lock:
            return list(self.store.rows.values())

    def recent(self, filters: dict, limit: int, text: str = "") -> list[dict]:
        out = []
        for r in reversed(self._rows()):
            j = row_json(r)
            if all(str(j[k]) == str(v) for k, v in filters.items()) and (
                    not text or text.lower() in r.raw_message.lower()):
                out.append(j)
                if len(out) >= limit:
                    break
        return out

    def per_minute(self, minutes: int) -> list[dict]:
        since = datetime.now(UTC) - timedelta(minutes=minutes)
        buckets: dict[str, Counter] = {}
        for r in self._rows():
            if r.timestamp >= since:
                buckets.setdefault(r.timestamp.strftime("%Y-%m-%dT%H:%M"), Counter())[str(r.status)] += 1
        return [{"minute": m, "FINAL": c["FINAL"], "PENDING": c["PENDING"]}
                for m, c in sorted(buckets.items())]

    def top(self, field: str, minutes: int, limit: int) -> list[tuple]:
        since = datetime.now(UTC) - timedelta(minutes=minutes)
        attr = TOP_FIELDS[field]
        counts = Counter(getattr(r, attr) for r in self._rows() if r.timestamp >= since)
        return counts.most_common(limit)

    def source_lines(self, host: str, service: str, source: str, limit: int) -> list[str]:
        rows = [r.raw_message for r in self._rows()
                if (r.device_id, r.service, r.source) == (host, service, source)]
        return rows[-limit:]

    def source_rows(self, host: str, service: str, source: str, limit: int) -> list[tuple]:
        rows = [(r.timestamp, r.template_id) for r in self._rows()
                if (r.device_id, r.service, r.source) == (host, service, source)]
        return sorted(rows)[-limit:]

    def alerts(self, limit: int) -> list[dict]:
        with self.store._lock:
            alerts = list(self.store.alerts)
        return [alert_json(asdict(a)) for a in reversed(alerts[-limit:])]

    def query(self, sql: str) -> dict:
        rows = self._rows()
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE logs (log_id TEXT, timestamp TEXT, device_id TEXT, service TEXT, "
                   "source TEXT, template_id INTEGER, status TEXT, payload TEXT, raw_message TEXT)")
        db.executemany("INSERT INTO logs VALUES (?,?,?,?,?,?,?,?,?)", [
            (str(r.log_id), r.timestamp.isoformat(), r.device_id, r.service, r.source,
             r.template_id, str(r.status), json.dumps(r.payload), r.raw_message) for r in rows])
        db.execute("PRAGMA query_only = ON")
        try:
            cur = db.execute(sql)
        except sqlite3.Error as error:
            raise ValueError(str(error)) from error
        return {"columns": [d[0] for d in cur.description or ()],
                "rows": cur.fetchmany(MAX_QUERY_ROWS), "engine": "sqlite", "table": "logs"}


def alert_json(a: dict) -> dict:
    return {"alert_id": str(a["alert_id"]), "key": a["device_id"],
            "window_start": _iso(a["window_start"]), "window_end": _iso(a["window_end"]),
            "template_ids": list(a["template_ids"]), "miss_ratio": a["miss_ratio"],
            "novelty_ratio": a["novelty_ratio"], "reasons": list(a["reasons"]),
            "model_version": a["model_version"]}


def _iso(t) -> str:
    return t.isoformat() if isinstance(t, datetime) else str(t)


class ClickHouseReads:
    def __init__(self, store):
        self.store = store
        self.client = store.client

    def _q(self, sql: str, **params):
        self.store.flush()
        return self.client.query(sql, parameters=params).result_rows

    def recent(self, filters: dict, limit: int, text: str = "") -> list[dict]:
        where: list[str] = ["1"]
        params: dict[str, object] = {"limit": limit}
        cols = {"host": "device_id", "service": "service", "source": "source",
                "status": "toString(log_status)", "template_id": "toString(template_id)"}
        for k, v in filters.items():
            where.append(f"{cols[k]} = {{{k}:String}}")
            params[k] = str(v)
        if text:
            where.append("positionCaseInsensitive(raw_message, {text:String}) > 0")
            params["text"] = text
        rows = self._q(
            "SELECT log_id, timestamp, device_id, service, source, tags, toString(log_status), "
            "template_id, toJSONString(payload), raw_message FROM iot_logs FINAL "
            f"WHERE {' AND '.join(where)} ORDER BY timestamp DESC LIMIT {{limit:UInt32}}", **params)
        return [{"log_id": str(i), "timestamp": _iso(ts), "host": h, "service": s, "source": src,
                 "tags": list(tags), "status": st, "template_id": tid, "payload": json.loads(p or "{}"),
                 "message": m} for i, ts, h, s, src, tags, st, tid, p, m in rows]

    def per_minute(self, minutes: int) -> list[dict]:
        rows = self._q(
            "SELECT formatDateTime(toStartOfMinute(timestamp), '%Y-%m-%dT%H:%i') AS m, "
            "toString(log_status), count() FROM iot_logs FINAL "
            "WHERE timestamp >= now() - INTERVAL {minutes:UInt32} MINUTE GROUP BY m, log_status ORDER BY m",
            minutes=minutes)
        buckets: dict[str, dict] = {}
        for m, status, n in rows:
            buckets.setdefault(m, {"minute": m, "FINAL": 0, "PENDING": 0})[status] = n
        return list(buckets.values())

    def top(self, field: str, minutes: int, limit: int) -> list[tuple]:
        col = TOP_FIELDS[field]
        return [tuple(r) for r in self._q(
            f"SELECT {col}, count() AS n FROM iot_logs FINAL "
            "WHERE timestamp >= now() - INTERVAL {minutes:UInt32} MINUTE "
            "GROUP BY 1 ORDER BY n DESC LIMIT {limit:UInt32}", minutes=minutes, limit=limit)]

    def source_lines(self, host: str, service: str, source: str, limit: int) -> list[str]:
        return [m for (m,) in self._q(
            "SELECT raw_message FROM iot_logs FINAL WHERE device_id = {h:String} AND "
            "service = {s:String} AND source = {src:String} ORDER BY timestamp DESC "
            "LIMIT {limit:UInt32}", h=host, s=service, src=source, limit=limit)][::-1]

    def source_rows(self, host: str, service: str, source: str, limit: int) -> list[tuple]:
        return [tuple(r) for r in self._q(
            "SELECT timestamp, template_id FROM iot_logs FINAL WHERE device_id = {h:String} AND "
            "service = {s:String} AND source = {src:String} ORDER BY timestamp DESC "
            "LIMIT {limit:UInt32}", h=host, s=service, src=source, limit=limit)][::-1]

    def alerts(self, limit: int) -> list[dict]:
        cols = ["alert_id", "device_id", "window_start", "window_end", "template_ids",
                "miss_ratio", "novelty_ratio", "reasons", "model_version"]
        rows = self._q(f"SELECT {', '.join(cols)} FROM anomaly_alerts ORDER BY window_end DESC "
                       "LIMIT {limit:UInt32}", limit=limit)
        return [alert_json(dict(zip(cols, r))) for r in rows]

    def query(self, sql: str) -> dict:
        self.store.flush()
        try:
            result = self.client.query(sql, settings={"readonly": 2, "max_result_rows": MAX_QUERY_ROWS,
                                                      "result_overflow_mode": "break"})
        except Exception as error:  # the server's message, shown under the editor
            raise ValueError(str(error).strip().splitlines()[0]) from error
        return {"columns": list(result.column_names), "rows": result.result_rows[:MAX_QUERY_ROWS],
                "engine": "clickhouse", "table": "iot_logs"}


def reads_for(store):
    return MemoryReads(store) if isinstance(store, MemoryStore) else ClickHouseReads(store)
