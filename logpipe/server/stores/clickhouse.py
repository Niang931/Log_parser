"""ClickHouse: reads on ``iot_logs`` (``FINAL``, since rows are upserted by version) and its SQL
dialect. Free-form queries run with ``readonly=2``, or as the read-only user when one is set."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from typing import ClassVar

from ...settings import get_settings
from .base import (
    HEAD,
    LEVEL_RULES,
    LEVELS,
    StoreReads,
    alert_json,
    histogram_keys,
    iso,
    level,
    step_for,
    zero_fill,
)
from .sql import SqlCompiler, guard_sql


def _sql_str(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


# level() as a ClickHouse expression: a stated severity tag, then LEVEL_RULES in order.
LEVEL_SQL = (
    "multiIf("
    + "".join(f"has(tags, 'severity:{lvl}'), '{lvl}', " for lvl in LEVELS)
    + ", ".join(
        f"match({f'substring(raw_message, 1, {HEAD})' if head else 'raw_message'}, {_sql_str(rx)}), '{lvl}'"
        for lvl, rx, head in LEVEL_RULES
    )
    + ", 'info')"
)


class ClickHouseSql(SqlCompiler):
    engine, table, source = "clickhouse", "iot_logs", "iot_logs FINAL"
    columns: ClassVar[dict[str, str]] = {
        "host": "device_id",
        "service": "service",
        "source": "source",
        "level": LEVEL_SQL,
        "status": "toString(log_status)",
        "template_id": "template_id",
        "message": "raw_message",
    }
    about_table: ClassVar[dict[str, str]] = {"level": "level (computed)"}
    log_id, count = "toString(log_id)", "count()"

    def quote(self, value: str) -> str:
        return _sql_str(value)

    def extracted(self, key: str, numeric: bool) -> str:
        raw = f"JSONExtractString(toJSONString(payload), {self.quote(key)})"
        return f"toFloat64OrNull({raw})" if numeric else raw

    def payload_json(self) -> str:
        return "toJSONString(payload)"

    def time_col(self) -> str:
        return "timestamp"

    def instant(self, t: datetime) -> str:
        return f"toDateTime64('{t.astimezone(UTC):%Y-%m-%d %H:%M:%S.%f}', 3, 'UTC')"

    def since_minutes(self, minutes: int) -> str:
        return f"timestamp >= now() - INTERVAL {minutes} MINUTE"

    def contains(self, col: str, value: str) -> str:
        return f"positionCaseInsensitive({col}, {self.quote(value)}) > 0"

    def unix_seconds(self) -> str:
        return "toUnixTimestamp(timestamp)"

    def bucket_index(self, origin: int, step: int) -> str:
        return f"intDiv({self.unix_seconds()} - {origin}, {step})"

    def time_bucket(self, step: int) -> str:
        return f"toStartOfInterval(timestamp, INTERVAL {step} SECOND)"

    def aggregate(self, fn: str, col: str) -> str:
        return {"count_distinct": f"uniqExact({col})", "p95": f"quantile(0.95)({col})"}.get(
            fn, f"{fn}({col})"
        )

    def template_counts(self, filters: list[dict], start: datetime, end: datetime) -> str:
        return (
            "SELECT template_id, service, count() AS n, min(timestamp) AS first, max(timestamp) AS last,\n"
            "  argMin(raw_message, timestamp) AS example, argMin(toString(log_id), timestamp) AS log_id,\n"
            f"  argMin(device_id, timestamp) AS host, argMin({LEVEL_SQL}, timestamp) AS level\n"
            f"FROM iot_logs FINAL\nWHERE {self._where(filters, start, end)}\n"
            f"GROUP BY template_id, service\nORDER BY first\nLIMIT {get_settings().ask.agg_cap}"
        )

    def template_sample(self, template_id: int, keys: list[str]) -> str:
        cols = "".join(f",\n  payload.`{k}` AS `{k}`" for k in keys)
        return (
            f"SELECT timestamp, device_id AS host, service{cols}\nFROM iot_logs FINAL\n"
            f"WHERE template_id = {int(template_id)}\nORDER BY timestamp DESC\nLIMIT 50"
        )


# The UI's own filters (Log Explorer): parameterised, not built from text.
_FILTER_COLS = {
    "host": "device_id",
    "service": "service",
    "source": "source",
    "status": "toString(log_status)",
    "template_id": "toString(template_id)",
    "level": LEVEL_SQL,
}
_ROW_COLS = (
    "SELECT log_id, timestamp, device_id, service, source, tags, toString(log_status), "
    "template_id, toJSONString(payload), raw_message FROM iot_logs FINAL "
)


def _row(i, ts, h, s, src, tags, st, tid, p, m) -> dict:
    return {
        "log_id": str(i),
        "timestamp": iso(ts),
        "host": h,
        "service": s,
        "source": src,
        "tags": list(tags),
        "status": st,
        "template_id": tid,
        "payload": json.loads(p or "{}"),
        "message": m,
        "level": level(m, tags),
    }


class ClickHouseReads(StoreReads):
    def __init__(self, store):
        self.store = store
        self.client = store.client
        self.compiler = ClickHouseSql()
        self.query_client = self.client
        cfg = get_settings().clickhouse
        # docker/clickhouse-init.sh creates it with SELECT on the log tables only
        if cfg.readonly_user:
            import clickhouse_connect

            self.query_client = clickhouse_connect.get_client(
                host=cfg.host,
                port=cfg.port,
                username=cfg.readonly_user,
                password=cfg.readonly_password,
                database=store.client.database,
            )

    def _q(self, sql: str, **params):
        self.store.flush()
        return self.client.query(sql, parameters=params).result_rows

    def _where(self, filters: dict, text: str, minutes: int | None) -> tuple[str, dict]:
        where, params = ["1"], {}
        for k, v in filters.items():
            where.append(f"{_FILTER_COLS[k]} = {{f_{k}:String}}")
            params[f"f_{k}"] = str(v)
        if text:
            where.append("positionCaseInsensitive(raw_message, {text:String}) > 0")
            params["text"] = text
        if minutes:
            where.append("timestamp >= now() - INTERVAL {minutes:UInt32} MINUTE")
            params["minutes"] = minutes
        return " AND ".join(where), params

    def recent(
        self, filters: dict, limit: int, text: str = "", minutes: int | None = None
    ) -> list[dict]:
        where, params = self._where(filters, text, minutes)
        return [
            _row(*r)
            for r in self._q(
                f"{_ROW_COLS}WHERE {where} ORDER BY timestamp DESC LIMIT {{limit:UInt32}}",
                limit=limit,
                **params,
            )
        ]

    def between(
        self,
        start: datetime,
        end: datetime,
        limit: int,
        host: str | None = None,
        service: str | None = None,
    ) -> list[dict]:
        where = [
            "timestamp >= parseDateTime64BestEffort({s:String}, 3)",
            "timestamp <= parseDateTime64BestEffort({e:String}, 3)",
        ]
        params: dict = {
            "s": start.astimezone(UTC).isoformat(),
            "e": end.astimezone(UTC).isoformat(),
            "limit": limit,
        }
        if host is not None:
            where.append("device_id = {h:String}")
            params["h"] = host
        if service is not None:
            where.append("service = {sv:String}")
            params["sv"] = service
        return [
            _row(*r)
            for r in self._q(
                f"{_ROW_COLS}WHERE {' AND '.join(where)} ORDER BY timestamp LIMIT {{limit:UInt32}}",
                **params,
            )
        ]

    def histogram(self, minutes: int, by: str, filters: dict | None = None, text: str = "") -> dict:
        step = step_for(minutes)
        where, params = self._where(filters or {}, text, minutes)
        key = LEVEL_SQL if by == "level" else "toString(log_status)"
        counts: dict[int, Counter] = {}
        for t, k, n in self._q(
            f"SELECT intDiv(toUnixTimestamp(timestamp), {step}) * {step} AS t, {key} AS k, count() "
            f"FROM iot_logs FINAL WHERE {where} GROUP BY t, k",
            **params,
        ):
            counts.setdefault(int(t), Counter())[k] = n
        return {"step": step, "points": zero_fill(minutes, step, counts, histogram_keys(by))}

    def top(
        self, field: str, minutes: int, limit: int, filters: dict | None = None, text: str = ""
    ) -> list[tuple]:
        col = "template_id" if field == "template_id" else _FILTER_COLS[field]
        where, params = self._where(filters or {}, text, minutes)
        return [
            tuple(r)
            for r in self._q(
                f"SELECT {col} AS v, count() AS n FROM iot_logs FINAL WHERE {where} "
                "GROUP BY v ORDER BY n DESC LIMIT {limit:UInt32}",
                limit=limit,
                **params,
            )
        ]

    def source_lines(self, host: str, service: str, source: str, limit: int) -> list[str]:
        return [
            m
            for (m,) in self._q(
                "SELECT raw_message FROM iot_logs FINAL WHERE device_id = {h:String} AND "
                "service = {s:String} AND source = {src:String} ORDER BY timestamp DESC "
                "LIMIT {limit:UInt32}",
                h=host,
                s=service,
                src=source,
                limit=limit,
            )
        ][::-1]

    def source_rows(self, host: str, service: str, source: str, limit: int) -> list[tuple]:
        return [
            tuple(r)
            for r in self._q(
                "SELECT timestamp, template_id FROM iot_logs FINAL WHERE device_id = {h:String} AND "
                "service = {s:String} AND source = {src:String} ORDER BY timestamp DESC "
                "LIMIT {limit:UInt32}",
                h=host,
                s=service,
                src=source,
                limit=limit,
            )
        ][::-1]

    def alerts(self, limit: int) -> list[dict]:
        cols = [
            "alert_id",
            "device_id",
            "window_start",
            "window_end",
            "template_ids",
            "miss_ratio",
            "novelty_ratio",
            "reasons",
            "model_version",
        ]
        rows = self._q(
            f"SELECT {', '.join(cols)} FROM anomaly_alerts ORDER BY window_end DESC "
            "LIMIT {limit:UInt32}",
            limit=limit,
        )
        return [alert_json(dict(zip(cols, r))) for r in rows]

    def query(self, text: str, max_rows: int | None = None) -> dict:
        sql = guard_sql(text)
        max_rows = max_rows or get_settings().server.max_query_rows
        self.store.flush()
        try:
            settings = {
                "max_result_rows": max_rows,
                "result_overflow_mode": "break",
                "max_execution_time": 30,
            }
            # the read-only user's profile pins readonly itself
            if self.query_client is self.client:
                settings["readonly"] = 2
            result = self.query_client.query(sql, settings=settings)
        except Exception as error:  # the server's message, shown under the editor
            raise ValueError(str(error).strip().splitlines()[0]) from error
        return {
            "columns": list(result.column_names),
            "rows": result.result_rows[:max_rows],
            "engine": "clickhouse",
            "table": "iot_logs",
        }
