"""What the UI and the learning jobs read from the log store, for both backends."""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from ..storage.base import LogRow
from ..storage.memory import MemoryStore

FILTERS = ("host", "service", "source", "status", "template_id", "level")
FACETS = ("level", "service", "host", "source", "status")
MAX_QUERY_ROWS = 500
LEVELS = ("error", "warn", "info", "debug")
HEAD = 120  # explicit level words are looked for in the line's first characters only

# (level, regex, head only). Written in the regex subset Python and ClickHouse (RE2) share, so
# the explorer's facet counts (SQL) and its rows (Python) always agree. First match wins: a
# declared level beats heuristics, the way Datadog's status remapper works.
LEVEL_RULES = (
    ("error", r"\b(FATAL|CRIT|CRITICAL|ERROR|ERR|SEVERE|EMERG|PANIC)\b", True),
    ("warn", r"\b(WARN|WARNING)\b", True),
    ("debug", r"\b(DEBUG|TRACE)\b", True),
    ("info", r"\b(INFO|NOTICE)\b", True),
    ("error", r"HTTP/[0-9.]+\" 5[0-9][0-9] |(?i:\b(exception|traceback|panic)\b)", False),
    ("warn", r"HTTP/[0-9.]+\" 4[0-9][0-9] ", False),
)
_RULES = [(lvl, re.compile(rx), head) for lvl, rx, head in LEVEL_RULES]


def level(message: str, tags=()) -> str:
    """A severity the sender stated (OTLP severity -> ``severity:<level>`` tag) wins; else read
    from the text."""
    for tag in tags:
        if tag.startswith("severity:") and tag[9:] in LEVELS:
            return tag[9:]
    for lvl, rx, head in _RULES:
        if rx.search(message[:HEAD] if head else message):
            return lvl
    return "info"


def _sql_str(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


LEVEL_SQL = "multiIf(" + "".join(f"has(tags, 'severity:{lvl}'), '{lvl}', " for lvl in LEVELS) + ", ".join(
    f"match({f'substring(raw_message, 1, {HEAD})' if head else 'raw_message'}, {_sql_str(rx)}), '{lvl}'"
    for lvl, rx, head in LEVEL_RULES) + ", 'info')"


def step_for(minutes: int) -> int:
    """Bucket seconds giving 40-90 bars over the range, on round intervals."""
    for step in (10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800):
        if minutes * 60 / step <= 90:
            return step
    return 21600


def zero_fill(minutes: int, step: int, counts: dict[int, Counter], keys: tuple[str, ...]) -> list[dict]:
    """Every bucket in the range, empty ones included, so bars keep their width and gaps show."""
    end = int(datetime.now(UTC).timestamp()) // step * step
    start = end - (minutes * 60 // step) * step
    return [{"t": t * 1000, **{k: counts.get(t, Counter())[k] for k in keys}}
            for t in range(start + step, end + step, step)]


def row_json(r: LogRow) -> dict:
    return {"log_id": str(r.log_id), "timestamp": r.timestamp.isoformat(), "host": r.device_id,
            "service": r.service, "source": r.source, "tags": list(r.tags), "status": str(r.status),
            "template_id": r.template_id, "payload": r.payload, "message": r.raw_message,
            "level": level(r.raw_message, r.tags)}


class MemoryReads:
    def __init__(self, store: MemoryStore):
        self.store = store

    def _rows(self) -> list[LogRow]:
        with self.store._lock:
            return list(self.store.rows.values())

    def _match(self, filters: dict, text: str, minutes: int | None) -> list[dict]:
        """Rows as JSON, oldest first, within ``minutes`` and matching every filter."""
        since = datetime.now(UTC) - timedelta(minutes=minutes) if minutes else None
        text = text.lower()
        out = []
        for r in self._rows():
            if (since and r.timestamp < since) or (text and text not in r.raw_message.lower()):
                continue
            j = row_json(r)
            if all(str(j[k]) == str(v) for k, v in filters.items()):
                out.append(j)
        return out

    def recent(self, filters: dict, limit: int, text: str = "", minutes: int | None = None) -> list[dict]:
        return sorted(self._match(filters, text, minutes), key=lambda j: j["timestamp"], reverse=True)[:limit]

    def between(self, start: datetime, end: datetime, limit: int, host: str | None = None,
                service: str | None = None) -> list[dict]:
        """Rows logged from ``start`` to ``end`` (inclusive), oldest first."""
        rows = sorted((r for r in self._rows() if start <= r.timestamp <= end
                       and host in (None, r.device_id) and service in (None, r.service)),
                      key=lambda r: r.timestamp)
        return [row_json(r) for r in rows[:limit]]

    def histogram(self, minutes: int, by: str, filters: dict | None = None, text: str = "") -> dict:
        step, key = step_for(minutes), "level" if by == "level" else "status"
        counts: dict[int, Counter] = {}
        for j in self._match(filters or {}, text, minutes):
            t = int(datetime.fromisoformat(j["timestamp"]).timestamp()) // step * step
            counts.setdefault(t, Counter())[j[key]] += 1
        return {"step": step, "points": zero_fill(minutes, step, counts, _keys(by))}

    def top(self, field: str, minutes: int, limit: int, filters: dict | None = None,
            text: str = "") -> list[tuple]:
        col = {"host": "host"}.get(field, field)
        return Counter(j[col] for j in self._match(filters or {}, text, minutes)).most_common(limit)

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
                   "source TEXT, template_id INTEGER, status TEXT, payload TEXT, raw_message TEXT, "
                   "level TEXT)")
        db.executemany("INSERT INTO logs VALUES (?,?,?,?,?,?,?,?,?,?)", [
            (str(r.log_id), r.timestamp.astimezone(UTC).isoformat(), r.device_id, r.service, r.source,
             r.template_id, str(r.status), json.dumps(r.payload), r.raw_message, level(r.raw_message, r.tags))
            for r in rows])
        db.execute("PRAGMA query_only = ON")
        try:
            cur = db.execute(sql)
        except sqlite3.Error as error:
            raise ValueError(str(error)) from error
        return {"columns": [d[0] for d in cur.description or ()],
                "rows": cur.fetchmany(MAX_QUERY_ROWS), "engine": "sqlite", "table": "logs"}


def _keys(by: str) -> tuple[str, ...]:
    return LEVELS if by == "level" else ("FINAL", "PENDING")


def alert_json(a: dict) -> dict:
    return {"alert_id": str(a["alert_id"]), "key": a["device_id"],
            "window_start": _iso(a["window_start"]), "window_end": _iso(a["window_end"]),
            "template_ids": list(a["template_ids"]), "miss_ratio": a["miss_ratio"],
            "novelty_ratio": a["novelty_ratio"], "reasons": list(a["reasons"]),
            "model_version": a["model_version"]}


def _iso(t) -> str:
    return t.isoformat() if isinstance(t, datetime) else str(t)


CH_COLS = {"host": "device_id", "service": "service", "source": "source",
           "status": "toString(log_status)", "template_id": "toString(template_id)", "level": LEVEL_SQL}


class ClickHouseReads:
    def __init__(self, store):
        self.store = store
        self.client = store.client

    def _q(self, sql: str, **params):
        self.store.flush()
        return self.client.query(sql, parameters=params).result_rows

    def _where(self, filters: dict, text: str, minutes: int | None) -> tuple[str, dict]:
        where, params = ["1"], {}
        for k, v in filters.items():
            where.append(f"{CH_COLS[k]} = {{f_{k}:String}}")
            params[f"f_{k}"] = str(v)
        if text:
            where.append("positionCaseInsensitive(raw_message, {text:String}) > 0")
            params["text"] = text
        if minutes:
            where.append("timestamp >= now() - INTERVAL {minutes:UInt32} MINUTE")
            params["minutes"] = minutes
        return " AND ".join(where), params

    def recent(self, filters: dict, limit: int, text: str = "", minutes: int | None = None) -> list[dict]:
        where, params = self._where(filters, text, minutes)
        rows = self._q(
            "SELECT log_id, timestamp, device_id, service, source, tags, toString(log_status), "
            f"template_id, toJSONString(payload), raw_message FROM iot_logs FINAL "
            f"WHERE {where} ORDER BY timestamp DESC LIMIT {{limit:UInt32}}", limit=limit, **params)
        return [{"log_id": str(i), "timestamp": _iso(ts), "host": h, "service": s, "source": src,
                 "tags": list(tags), "status": st, "template_id": tid, "payload": json.loads(p or "{}"),
                 "message": m, "level": level(m, tags)} for i, ts, h, s, src, tags, st, tid, p, m in rows]

    def between(self, start: datetime, end: datetime, limit: int, host: str | None = None,
                service: str | None = None) -> list[dict]:
        where = ["timestamp >= parseDateTime64BestEffort({s:String}, 3)",
                 "timestamp <= parseDateTime64BestEffort({e:String}, 3)"]
        params: dict = {"s": start.astimezone(UTC).isoformat(), "e": end.astimezone(UTC).isoformat(), "limit": limit}
        if host is not None:
            where.append("device_id = {h:String}")
            params["h"] = host
        if service is not None:
            where.append("service = {sv:String}")
            params["sv"] = service
        rows = self._q(
            "SELECT log_id, timestamp, device_id, service, source, tags, toString(log_status), "
            f"template_id, toJSONString(payload), raw_message FROM iot_logs FINAL "
            f"WHERE {' AND '.join(where)} ORDER BY timestamp LIMIT {{limit:UInt32}}", **params)
        return [{"log_id": str(i), "timestamp": _iso(ts), "host": h, "service": sv, "source": src,
                 "tags": list(tags), "status": st, "template_id": tid, "payload": json.loads(p or "{}"),
                 "message": m, "level": level(m, tags)} for i, ts, h, sv, src, tags, st, tid, p, m in rows]

    def histogram(self, minutes: int, by: str, filters: dict | None = None, text: str = "") -> dict:
        step = step_for(minutes)
        where, params = self._where(filters or {}, text, minutes)
        key = LEVEL_SQL if by == "level" else "toString(log_status)"
        counts: dict[int, Counter] = {}
        for t, k, n in self._q(
                f"SELECT intDiv(toUnixTimestamp(timestamp), {step}) * {step} AS t, {key} AS k, count() "
                f"FROM iot_logs FINAL WHERE {where} GROUP BY t, k", **params):
            counts.setdefault(int(t), Counter())[k] = n
        return {"step": step, "points": zero_fill(minutes, step, counts, _keys(by))}

    def top(self, field: str, minutes: int, limit: int, filters: dict | None = None,
            text: str = "") -> list[tuple]:
        col = {"template_id": "template_id"}.get(field, CH_COLS[field])
        where, params = self._where(filters or {}, text, minutes)
        return [tuple(r) for r in self._q(
            f"SELECT {col} AS v, count() AS n FROM iot_logs FINAL WHERE {where} "
            "GROUP BY v ORDER BY n DESC LIMIT {limit:UInt32}", limit=limit, **params)]

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
