"""Needs a ClickHouse server: ``make docker-up`` (or set CLICKHOUSE_* env vars)."""
import os
from datetime import UTC, datetime, timedelta

import pytest

from logpipe.parsing.frozen_parser import ParseResult, Status
from logpipe.storage.base import AnomalyAlert, LogRow

pytestmark = pytest.mark.integration
clickhouse_connect = pytest.importorskip("clickhouse_connect")


@pytest.fixture
def store():
    from logpipe.storage.clickhouse import ClickHouseStore

    try:
        client = clickhouse_connect.get_client(
            host=os.environ.get("CLICKHOUSE_HOST", "127.0.0.1"),
            port=int(os.environ.get("CLICKHOUSE_PORT", "8123")),
            username=os.environ.get("CLICKHOUSE_USER", "logpipe"),
            password=os.environ.get("CLICKHOUSE_PASSWORD", "change-me"),
        )
    except Exception as error:  # noqa: BLE001 - any connection failure means "skip"
        pytest.skip(f"ClickHouse not reachable: {error}")
    db = "logpipe_test"
    client.command(f"DROP DATABASE IF EXISTS {db}")
    client.command(f"CREATE DATABASE {db}")
    client.database = db
    yield ClickHouseStore(client, batch_size=2)
    client.command(f"DROP DATABASE {db}")


TS = datetime(2026, 4, 13, 12, tzinfo=UTC)


def _pending(i: int, ts=TS) -> LogRow:
    return LogRow.from_result(ParseResult(1, Status.PENDING, {}, 1), device_id="dev-1",
                              timestamp=ts, raw_message=f"unknown line {i}")


def test_backfill_supersedes_pending_rows(store):
    rows = [_pending(i) for i in range(3)]  # same device, same millisecond
    store.insert_logs(rows)
    assert store.count_by_status() == {"PENDING": 3}
    pending = store.fetch_pending()
    assert {r.log_id for r in pending} == {r.log_id for r in rows}

    resolved = rows[0].resolved(ParseResult(42, Status.FINAL, {"temp": "22.5C"}, 2))
    store.insert_logs([resolved])
    assert store.count_by_status() == {"FINAL": 1, "PENDING": 2}
    payload = store.client.query(
        "SELECT payload.temp, template_id, version FROM iot_logs FINAL WHERE log_id = {id:UUID}",
        parameters={"id": rows[0].log_id},
    ).result_rows
    assert payload == [("22.5C", 42, 2)]


def test_alerts_round_trip(store):
    alert = AnomalyAlert("dev-1", TS, TS + timedelta(seconds=30), (2, 3, 1), 0.6, 0.33, 1.5,
                         ("sequence", "novelty"), True, 3, "logbert-test")
    store.insert_alerts([alert])
    rows = store.client.query("SELECT device_id, reasons, had_unknowns FROM anomaly_alerts").result_rows
    assert rows == [("dev-1", ["sequence", "novelty"], 1)]
