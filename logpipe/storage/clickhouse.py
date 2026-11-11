"""ClickHouse backend: batched inserts into the versioned ``iot_logs`` table.

Rows are buffered and flushed when the batch fills (or on :meth:`flush`).
A failed insert keeps the batch and retries with backoff; past
``max_buffer`` rows the writer raises :class:`BackpressureError` so the
caller slows down instead of silently dropping logs.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Sequence
from datetime import UTC
from importlib import resources

import clickhouse_connect

from ..parsing.frozen_parser import Status
from ..settings import get_settings
from .base import AnomalyAlert, LogRow

LOG_COLUMNS = [
    "log_id",
    "timestamp",
    "device_id",
    "template_id",
    "log_status",
    "registry_version",
    "version",
    "payload",
    "raw_message",
    "service",
    "source",
    "tags",
]
ALERT_COLUMNS = [
    "alert_id",
    "device_id",
    "window_start",
    "window_end",
    "template_ids",
    "miss_ratio",
    "novelty_ratio",
    "distance",
    "reasons",
    "had_unknowns",
    "registry_version",
    "model_version",
]


class BackpressureError(RuntimeError):
    """The write buffer is full because ClickHouse keeps rejecting inserts."""


def _schema_statements() -> list[str]:
    sql = resources.files("logpipe.storage").joinpath("schema.sql").read_text(encoding="utf-8")
    body = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    return [stmt.strip() for stmt in body.split(";") if stmt.strip()]


class ClickHouseStore:
    def __init__(
        self,
        client,
        *,
        batch_size: int = 5_000,
        max_buffer: int = 500_000,
        retries: int = 5,
        backoff: float = 0.5,
    ):
        self.client = client
        self.batch_size = batch_size
        self.max_buffer = max_buffer
        self.retries = retries
        self.backoff = backoff
        self._buffer: list[LogRow] = []
        self._lock = threading.Lock()
        for statement in _schema_statements():
            self.client.command(statement)

    @classmethod
    def from_env(cls, database: str | None = None, **kwargs) -> ClickHouseStore:
        """``CLICKHOUSE_*`` settings; ``database`` (created if missing) overrides ``CLICKHOUSE_DB``."""
        cfg = get_settings().clickhouse
        client = clickhouse_connect.get_client(
            host=cfg.host,
            port=cfg.port,
            username=cfg.user,
            password=cfg.password,
            # One client serves the ingest, job, housekeeping and request threads; ClickHouse
            # refuses concurrent queries inside one session, and nothing here needs a session.
            autogenerate_session_id=False,
        )
        db = database or cfg.db
        client.command(f"CREATE DATABASE IF NOT EXISTS `{db}`")
        client.database = db
        return cls(client, **kwargs)

    def insert_logs(self, rows: Sequence[LogRow]) -> None:
        with self._lock:
            if len(self._buffer) + len(rows) > self.max_buffer:
                raise BackpressureError(
                    f"{len(self._buffer)} rows buffered; ClickHouse is not accepting writes"
                )
            self._buffer.extend(rows)
            if len(self._buffer) >= self.batch_size:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        data = [
            [
                r.log_id,
                r.timestamp,
                r.device_id,
                r.template_id,
                str(r.status),
                r.registry_version,
                r.version,
                json.dumps(r.payload),
                r.raw_message,
                r.service,
                r.source,
                list(r.tags),
            ]
            for r in self._buffer
        ]
        self._insert("iot_logs", data, LOG_COLUMNS)
        self._buffer.clear()

    def _insert(self, table: str, data: list[list], columns: list[str]) -> None:
        for attempt in range(self.retries):
            try:
                self.client.insert(table, data, column_names=columns)
                return
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(self.backoff * 2**attempt)

    def insert_alerts(self, alerts: Sequence[AnomalyAlert]) -> None:
        if not alerts:
            return
        data = [
            [
                a.alert_id,
                a.device_id,
                a.window_start,
                a.window_end,
                list(a.template_ids),
                a.miss_ratio,
                a.novelty_ratio,
                a.distance,
                list(a.reasons),
                int(a.had_unknowns),
                a.registry_version,
                a.model_version,
            ]
            for a in alerts
        ]
        self._insert("anomaly_alerts", data, ALERT_COLUMNS)

    def fetch_pending(self, limit: int = 10_000, template_id: int | None = None) -> list[LogRow]:
        """PENDING rows, or with ``template_id`` that template's rows (to re-extract them)."""
        self.flush()
        where = "log_status = 'PENDING'" if template_id is None else "template_id = {tid:UInt32}"
        result = self.client.query(
            "SELECT log_id, timestamp, device_id, template_id, registry_version, version, "
            f"raw_message, service, source, tags FROM iot_logs FINAL WHERE {where} "
            "ORDER BY timestamp LIMIT {limit:UInt32}",
            parameters={"limit": limit, "tid": template_id or 0},
        )
        return [
            LogRow(
                log_id=log_id if isinstance(log_id, uuid.UUID) else uuid.UUID(str(log_id)),
                timestamp=ts if ts.tzinfo else ts.replace(tzinfo=UTC),
                device_id=device_id,
                template_id=template_id,
                status=Status.PENDING if template_id is None else Status.FINAL,
                registry_version=registry_version,
                payload={},
                raw_message=raw,
                version=version,
                service=service,
                source=source,
                tags=tuple(tags),
            )
            for log_id, ts, device_id, template_id, registry_version, version, raw, service, source, tags in result.result_rows
        ]

    def count_by_status(self) -> dict[str, int]:
        self.flush()
        result = self.client.query(
            "SELECT toString(log_status), count() FROM iot_logs FINAL GROUP BY log_status"
        )
        return {status: count for status, count in result.result_rows}

    def close(self) -> None:
        self.flush()
        self.client.close()
