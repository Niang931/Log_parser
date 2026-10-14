import uuid
from datetime import UTC, datetime

from logpipe.parsing.frozen_parser import ParseResult, Status
from logpipe.storage.base import LogRow
from logpipe.storage.memory import MemoryStore

TS = datetime(2026, 4, 13, tzinfo=UTC)


def _pending(line="x") -> LogRow:
    result = ParseResult(1, Status.PENDING, {}, registry_version=3)
    return LogRow.from_result(result, device_id="d1", timestamp=TS, raw_message=line)


def test_resolved_row_keeps_identity_and_sort_key_with_higher_version():
    row = _pending()
    final = row.resolved(ParseResult(7, Status.FINAL, {"k": "v"}, registry_version=4))
    assert (final.log_id, final.device_id, final.timestamp) == (row.log_id, row.device_id, row.timestamp)
    assert final.version > row.version and final.status == Status.FINAL


def test_memory_store_keeps_latest_version_like_replacing_merge_tree():
    store = MemoryStore()
    row = _pending()
    store.insert_logs([row])
    assert store.count_by_status() == {"PENDING": 1}
    assert store.fetch_pending() == [row]
    store.insert_logs([row.resolved(ParseResult(7, Status.FINAL, {}, registry_version=4))])
    store.insert_logs([row])  # a late duplicate of the old version must not win
    assert store.count_by_status() == {"FINAL": 1}
    assert store.fetch_pending() == []


def test_rows_in_the_same_millisecond_are_all_kept():
    store = MemoryStore()
    rows = [_pending(f"line {i}") for i in range(5)]
    store.insert_logs(rows)
    assert len({r.log_id for r in rows}) == 5
    assert store.count_by_status() == {"PENDING": 5}
    assert all(isinstance(r.log_id, uuid.UUID) for r in rows)
