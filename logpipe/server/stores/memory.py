"""The memory store: UI reads in Python over its rows, and queries on a read-only SQLite copy of
them (``logs``), rebuilt only after new inserts."""
from __future__ import annotations

import json
import sqlite3
import threading
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import ClassVar

from ...settings import get_settings
from ...storage.base import LogRow
from ...storage.memory import MemoryStore
from .base import StoreReads, alert_json, histogram_keys, level, row_json, step_for, zero_fill
from .sql import SqlCompiler, guard_sql


class SqliteSql(SqlCompiler):
    engine, table, source = "sqlite", "logs", "logs"
    columns: ClassVar[dict[str, str]] = {"host": "device_id", "service": "service", "source": "source", "level": "level",
               "status": "status", "template_id": "template_id", "message": "raw_message"}

    def quote(self, value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    def extracted(self, key: str, numeric: bool) -> str:
        raw = f"json_extract(payload, '$.{key}')"
        return f"CAST({raw} AS REAL)" if numeric else raw

    def time_col(self) -> str:
        return "julianday(timestamp)"  # ISO text with offsets: compare as instants

    def instant(self, t: datetime) -> str:
        return f"julianday('{t.astimezone(UTC):%Y-%m-%dT%H:%M:%S.%f}+00:00')"

    def since_minutes(self, minutes: int) -> str:
        return f"timestamp >= strftime('%Y-%m-%dT%H:%M:%S', 'now', '-{minutes} minutes')"

    def contains(self, col: str, value: str) -> str:
        return f"instr(lower({col}), lower({self.quote(value)})) > 0"

    def unix_seconds(self) -> str:
        return "CAST(strftime('%s', timestamp) AS INTEGER)"

    def time_bucket(self, step: int) -> str:
        return f"datetime(({self.unix_seconds()} / {step}) * {step}, 'unixepoch')"

    def aggregate(self, fn: str, col: str) -> str:
        if fn == "p95":
            raise ValueError("p95 needs ClickHouse; the memory store's SQLite has no percentile")
        return f"count(DISTINCT {col})" if fn == "count_distinct" else f"{fn}({col})"

    def template_counts(self, filters: list[dict], start: datetime, end: datetime) -> str:
        return ("SELECT template_id, service, n, timestamp AS first, last, raw_message AS example, log_id,\n"
                "  device_id AS host, level\nFROM (SELECT *, ROW_NUMBER() OVER w AS rn, COUNT(*) OVER p AS n,\n"
                "    MAX(timestamp) OVER p AS last\n"
                f"  FROM logs WHERE {self._where(filters, start, end)}\n"
                "  WINDOW p AS (PARTITION BY template_id, service),\n"
                "    w AS (PARTITION BY template_id, service ORDER BY julianday(timestamp)))\n"
                f"WHERE rn = 1\nORDER BY julianday(first)\nLIMIT {get_settings().ask.agg_cap}")

    def template_sample(self, template_id: int, keys: list[str]) -> str:
        cols = "".join(f",\n  json_extract(payload, '$.\"{k}\"') AS \"{k}\"" for k in keys)
        return (f"SELECT timestamp, device_id AS host, service{cols}\nFROM logs\n"
                f"WHERE template_id = {int(template_id)}\nORDER BY timestamp DESC\nLIMIT 50")


class MemoryReads(StoreReads):
    def __init__(self, store: MemoryStore):
        self.store = store
        self.compiler = SqliteSql()
        self._db: tuple[int, sqlite3.Connection] | None = None
        self._db_lock = threading.Lock()

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
        return {"step": step, "points": zero_fill(minutes, step, counts, histogram_keys(by))}

    def top(self, field: str, minutes: int, limit: int, filters: dict | None = None, text: str = "") -> list[tuple]:
        return Counter(j[field] for j in self._match(filters or {}, text, minutes)).most_common(limit)

    def source_lines(self, host: str, service: str, source: str, limit: int) -> list[str]:
        return [r.raw_message for r in self._rows() if (r.device_id, r.service, r.source) == (host, service, source)][-limit:]

    def source_rows(self, host: str, service: str, source: str, limit: int) -> list[tuple]:
        rows = [(r.timestamp, r.template_id) for r in self._rows()
                if (r.device_id, r.service, r.source) == (host, service, source)]
        return sorted(rows, key=lambda r: r[0])[-limit:]  # stable: same-millisecond lines keep arrival order

    def alerts(self, limit: int) -> list[dict]:
        with self.store._lock:
            alerts = list(self.store.alerts)
        return [alert_json(asdict(a)) for a in reversed(alerts[-limit:])]

    def _sqlite(self) -> sqlite3.Connection:
        if self._db is None or self._db[0] != self.store.inserts:
            rows, inserts = self._rows(), self.store.inserts
            db = sqlite3.connect(":memory:", check_same_thread=False)
            db.execute("CREATE TABLE logs (log_id TEXT, timestamp TEXT, device_id TEXT, service TEXT, "
                       "source TEXT, template_id INTEGER, status TEXT, payload TEXT, raw_message TEXT, level TEXT)")
            db.executemany("INSERT INTO logs VALUES (?,?,?,?,?,?,?,?,?,?)", [
                (str(r.log_id), r.timestamp.astimezone(UTC).isoformat(), r.device_id, r.service, r.source,
                 r.template_id, str(r.status), json.dumps(r.payload), r.raw_message, level(r.raw_message, r.tags))
                for r in rows])
            db.commit()
            db.execute("PRAGMA query_only = ON")
            self._db = (inserts, db)
        return self._db[1]

    def query(self, text: str, max_rows: int | None = None) -> dict:
        sql = guard_sql(text)
        with self._db_lock:
            try:
                cur = self._sqlite().execute(sql)
                rows = cur.fetchmany(max_rows or get_settings().server.max_query_rows)
                return {"columns": [d[0] for d in cur.description or ()], "rows": rows, "engine": "sqlite", "table": "logs"}
            except sqlite3.Error as error:
                raise ValueError(str(error)) from error
