"""ClickHouse store (``STORE=clickhouse``); tables in ``schema.sql``.

Rows are buffered and written in batches. While ClickHouse is down they stay in
the buffer; a full buffer blocks the caller and ``flush`` waits until all is written.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import fields
from importlib.resources import files

import clickhouse_connect

from ..parsing.frozen_parser import Status
from .base import AnomalyAlert, LogRow

LOGGER = logging.getLogger(__name__)

LOG_COLUMNS = [f.name for f in fields(LogRow)]
ALERT_COLUMNS = [f.name for f in fields(AnomalyAlert)]


def connect(database: str | None = None):
    return clickhouse_connect.get_client(
        host=os.environ.get("CLICKHOUSE_HOST", "127.0.0.1"),
        port=int(os.environ.get("CLICKHOUSE_PORT", "8123")),
        username=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
        database=database or os.environ.get("CLICKHOUSE_DB", "default"),
        tz_mode="aware",  # LogRow timestamps are aware UTC datetimes
    )


def create_schema(client, **settings: int) -> None:
    text = files(__package__).joinpath("schema.sql").read_text()
    text = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("--")
    )
    for statement in text.format(**{k: int(v) for k, v in settings.items()}).split(";"):
        if statement.strip():
            client.command(statement)


class ClickHouseStore:
    def __init__(
        self,
        client,
        *,
        pending_max_age_hours: int = 24,
        batch_size: int = 10_000,
        max_buffer: int = 500_000,
        flush_every: float = 1.0,
        max_backoff: float = 30.0,
    ):
        self.client = client
        self.pending_max_age_hours = int(pending_max_age_hours)
        self.batch_size = batch_size
        self.max_buffer = max_buffer
        self.flush_every = flush_every
        self.max_backoff = max_backoff
        self._buffers: dict[str, list[list]] = {"logs": [], "alerts": []}
        self._last_flush = time.monotonic()
        self._down = False

    @classmethod
    def from_env(cls, database: str | None = None, **kwargs) -> ClickHouseStore:
        client = connect(database)
        pending_max_age_hours = int(os.environ.get("PENDING_MAX_AGE_HOURS", "24"))
        create_schema(
            client,
            raw_ttl_days=int(os.environ.get("RAW_TTL_DAYS", "7")),
            log_ttl_days=int(os.environ.get("LOG_TTL_DAYS", "30")),
            alert_ttl_days=int(os.environ.get("ALERT_TTL_DAYS", "90")),
            pending_max_age_hours=pending_max_age_hours,
        )
        return cls(client, pending_max_age_hours=pending_max_age_hours, **kwargs)

    def insert_logs(self, rows: Sequence[LogRow]) -> None:
        self._add("logs", [[getattr(r, c) for c in LOG_COLUMNS] for r in rows])

    def insert_alerts(self, alerts: Sequence[AnomalyAlert]) -> None:
        self._add("alerts", [[getattr(a, c) for c in ALERT_COLUMNS] for a in alerts])

    def fetch_pending(self, limit: int = 10_000) -> list[LogRow]:
        # Older rows are dead-lettered, not re-queued. Inner query has no FINAL
        # so the skip indexes work (docs/clickhouse.md).
        result = self.client.query(
            f"SELECT {', '.join(LOG_COLUMNS)} FROM logs FINAL "
            "WHERE status = 'PENDING' AND (device_id, timestamp, log_id) IN ("
            "  SELECT device_id, timestamp, log_id FROM logs"
            "  WHERE status = 'PENDING' AND inserted_at >= now() - INTERVAL {age:UInt32} HOUR"
            ") ORDER BY timestamp LIMIT {limit:UInt32}",
            parameters={"age": self.pending_max_age_hours, "limit": limit},
        )
        rows = []
        for values in result.result_rows:
            row = dict(zip(LOG_COLUMNS, values, strict=True))
            row["status"] = Status(row["status"])
            rows.append(LogRow(**row))
        return rows

    def count_by_status(self) -> dict[str, int]:
        result = self.client.query(
            "SELECT status, count() FROM logs FINAL GROUP BY status"
        )
        return {status: count for status, count in result.result_rows}

    def flush(self) -> None:
        self._send(block=True)

    def close(self) -> None:
        self.flush()
        self.client.close()

    def _add(self, table: str, rows: list[list]) -> None:
        if not rows:
            return
        self._buffers[table].extend(rows)
        buffered = sum(len(b) for b in self._buffers.values())
        if buffered >= self.max_buffer:
            self._send(block=True)
        elif (
            buffered >= self.batch_size
            or time.monotonic() - self._last_flush >= self.flush_every
        ):
            self._send(block=False)

    def _send(self, *, block: bool) -> None:
        backoff = 0.5
        for table, columns in (("logs", LOG_COLUMNS), ("alerts", ALERT_COLUMNS)):
            while self._buffers[table]:
                try:
                    self.client.insert(
                        table, self._buffers[table], column_names=columns
                    )
                except Exception as exc:  # rows stay buffered
                    if not self._down:
                        LOGGER.warning(
                            "ClickHouse insert failed, buffering rows: %s", exc
                        )
                        self._down = True
                    if not block:
                        return
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self.max_backoff)
                    continue
                if self._down:
                    LOGGER.warning("ClickHouse insert recovered")
                    self._down = False
                self._buffers[table] = []
        self._last_flush = time.monotonic()
