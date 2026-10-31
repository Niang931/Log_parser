"""ClickHouse-only behaviour: schema, TTLs and the dead-letter view (``pytest -m integration``).

The behaviour shared with MemoryStore is tested in ``test_memory_storage.py``.
TTLs use the defaults of ``ClickHouseStore.from_env`` (raw 7 d, logs 30 d,
PENDING max age 24 h + 1 h grace).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from logpipe.parsing.frozen_parser import ParseResult, Status
from logpipe.storage.base import LogRow
from logpipe.storage.clickhouse import LOG_COLUMNS, create_schema

pytestmark = pytest.mark.integration

NOW = datetime.now(UTC).replace(microsecond=0)


def _row(status: Status, *, age: timedelta = timedelta(0), line: str = "x") -> LogRow:
    result = ParseResult(1, status, {}, registry_version=3)
    return LogRow.from_result(result, device_id="d1", timestamp=NOW - age, raw_message=line)


def _insert_aged(store, rows: list[LogRow], inserted_age: timedelta) -> None:
    """Insert as if ingested ``inserted_age`` ago (the store itself always uses now())."""
    data = [[getattr(r, c) for c in LOG_COLUMNS] + [NOW - inserted_age] for r in rows]
    store.client.insert("logs", data, column_names=[*LOG_COLUMNS, "inserted_at"])


def _refresh_dead_letter(store) -> list:
    store.client.command("SYSTEM REFRESH VIEW logs_dead_letter_mv")
    store.client.command("SYSTEM WAIT VIEW logs_dead_letter_mv")
    return store.client.query("SELECT log_id, reason FROM logs_dead_letter").result_rows


def test_schema_can_be_created_twice(clickhouse_store):
    create_schema(clickhouse_store.client, raw_ttl_days=7, log_ttl_days=30,
                  alert_ttl_days=90, pending_max_age_hours=24)


def test_ttl_drops_old_rows_then_old_raw_text(clickhouse_store):
    expired = _row(Status.FINAL, age=timedelta(days=40))
    raw_expired = _row(Status.FINAL, age=timedelta(days=10))
    fresh = _row(Status.FINAL)
    clickhouse_store.insert_logs([expired, raw_expired, fresh])
    clickhouse_store.flush()
    clickhouse_store.client.command("OPTIMIZE TABLE logs FINAL")
    raw = dict(clickhouse_store.client.query("SELECT log_id, raw_message FROM logs").result_rows)
    assert raw == {raw_expired.log_id: "", fresh.log_id: "x"}


def test_old_pending_rows_go_to_dead_letter_once(clickhouse_store):
    old, young = _row(Status.PENDING, line="old"), _row(Status.PENDING, line="young")
    _insert_aged(clickhouse_store, [old], timedelta(hours=24, minutes=30))
    _insert_aged(clickhouse_store, [young], timedelta(0))

    assert _refresh_dead_letter(clickhouse_store) == [(old.log_id, "pending_max_age")]
    assert _refresh_dead_letter(clickhouse_store) == [(old.log_id, "pending_max_age")]
    # rehydrate must not re-queue a dead-lettered row
    assert clickhouse_store.fetch_pending() == [young]


def test_pending_rows_are_deleted_after_the_grace_period(clickhouse_store):
    pending, final = _row(Status.PENDING), _row(Status.FINAL)
    _insert_aged(clickhouse_store, [pending, final], timedelta(hours=26))
    clickhouse_store.client.command("OPTIMIZE TABLE logs FINAL")
    assert clickhouse_store.count_by_status() == {"FINAL": 1}


def test_rows_stay_buffered_while_clickhouse_rejects_them(clickhouse_store, monkeypatch):
    calls = []

    def down(*args, **kwargs):
        calls.append(1)
        raise ConnectionError("down")

    real_insert = clickhouse_store.client.insert
    monkeypatch.setattr(clickhouse_store.client, "insert", down)
    clickhouse_store.batch_size = 1
    clickhouse_store.insert_logs([_row(Status.PENDING)])  # auto-flush fails, must not raise
    assert calls and clickhouse_store.count_by_status() == {}

    monkeypatch.setattr(clickhouse_store.client, "insert", real_insert)
    clickhouse_store.flush()
    assert clickhouse_store.count_by_status() == {"PENDING": 1}
