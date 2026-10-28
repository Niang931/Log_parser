"""Unresolved queue: wire format, the in-process queue, and the worker's at-least-once loop."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from logpipe.enrich.llm import OfflineAdapter
from logpipe.enrich.worker import EnrichmentWorker
from logpipe.parsing.frozen_parser import ParseResult, ParserHandle, Status
from logpipe.registry import SchemaRegistry
from logpipe.storage.base import LogRow
from logpipe.storage.memory import MemoryStore
from logpipe.unresolved import MemoryQueue, open_queue
from logpipe.unresolved.codec import decode_row, encode_row, partition_key, shape_key

TS = datetime(2026, 4, 13, 10, 30, 0, 123000, tzinfo=UTC)


def _pending(line: str, device: str = "thermostat-TH-1") -> LogRow:
    result = ParseResult(0, Status.PENDING, {}, registry_version=0)
    return LogRow.from_result(result, device_id=device, timestamp=TS, raw_message=line)


# -- codec ---------------------------------------------------------------------


def test_row_round_trips_through_json():
    row = _pending('OTA update failed code=0x1F msg="disk full" retry=3')
    assert decode_row(encode_row(row)) == row


def test_resolved_row_keeps_payload_and_version():
    row = _pending("x").resolved(ParseResult(7, Status.FINAL, {"k": "v"}, registry_version=4))
    back = decode_row(encode_row(row))
    assert back.status == Status.FINAL and back.payload == {"k": "v"} and back.version == 4
    assert isinstance(back.log_id, uuid.UUID)


def test_shape_key_ignores_numbers_but_not_words():
    a = shape_key("Watchdog reset reason=0x1F uptime=8123s")
    assert a == shape_key("Watchdog reset reason=0x2A uptime=99s")
    assert a != shape_key("Watchdog started uptime=8123s")


def test_partition_key_modes():
    row = _pending("sensor read value=42", device="dev-7")
    assert partition_key(row, "device") == b"dev-7"
    assert partition_key(row, "shape") == shape_key(row.raw_message)
    with pytest.raises(ValueError):
        partition_key(row, "random")


# -- memory queue --------------------------------------------------------------


def test_memory_queue_batches_and_redelivers_until_committed():
    queue = MemoryQueue()
    rows = [_pending(f"line {i}") for i in range(3)]
    for row in rows:
        queue.publish(row)
    first = queue.consume(2)
    assert [m.row for m in first] == rows[:2] and len(queue) == 1
    # Not committed: the same batch comes back (what a crashed worker would see).
    assert queue.consume(2) == first
    queue.commit()
    assert [m.row for m in queue.consume(10)] == rows[2:]
    queue.commit()
    assert queue.consume(10) == []
    stats = queue.stats()
    assert (stats.published, stats.consumed, stats.committed, stats.lag) == (3, 3, 3, 0)


def test_memory_queue_drops_oldest_when_full_and_counts_it():
    queue = MemoryQueue(max_size=2)
    for i in range(3):
        queue.publish(_pending(f"line {i}"))
    assert [m.row.raw_message for m in queue.consume(10)] == ["line 1", "line 2"]
    assert queue.stats().dropped == 1


def test_memory_queue_dead_letter_keeps_reason():
    queue = MemoryQueue()
    queue.publish(_pending("bad"), attempts=2)
    (message,) = queue.consume(1)
    assert message.attempts == 2
    queue.dead_letter(message, "rejected: all wildcards")
    assert queue.dead == [(message, "rejected: all wildcards")]
    assert queue.stats().dead_lettered == 1


def test_open_queue_kinds():
    assert isinstance(open_queue("memory", max_size=5), MemoryQueue)
    with pytest.raises(ValueError):
        open_queue("rabbitmq")


# -- worker on the queue -------------------------------------------------------


class FailingStore(MemoryStore):
    """Backfill that fails once, like a storage outage between consume and commit."""

    def __init__(self):
        super().__init__()
        self.fail_next = False

    def insert_logs(self, rows):
        if self.fail_next and any(r.status == Status.FINAL for r in rows):
            self.fail_next = False
            raise ConnectionError("store down")
        super().insert_logs(rows)


NEW = [
    f"2026-04-13 10:30:{i:02d}.000 [ERROR] OTA update failed code=0x{i:02X} retry={i}"
    for i in range(1, 6)
]


def _worker(snapshot, store=None, queue=None, **options):
    registry = SchemaRegistry(snapshot)
    handle = ParserHandle(snapshot)
    store = store or MemoryStore()
    worker = EnrichmentWorker(registry, handle, store, OfflineAdapter(), queue=queue, **options)
    return worker, store


def test_worker_learns_from_the_queue_and_commits_after_backfill(snapshot):
    queue = MemoryQueue()
    worker, store = _worker(snapshot, queue=queue)
    for line in NEW:
        row = _pending(line)
        store.insert_logs([row])
        worker.submit(row)
    report = worker.run_cycle()
    assert report.buffered == len(NEW) and report.resolved == len(NEW)
    assert report.published and report.still_pending == 0 and report.dead_lettered == 0
    assert store.count_by_status() == {"FINAL": len(NEW)}
    stats = queue.stats()
    assert (stats.consumed, stats.committed, stats.lag) == (len(NEW), len(NEW), 0)


def test_backfill_failure_leaves_the_batch_uncommitted_for_redelivery(snapshot):
    queue = MemoryQueue()
    store = FailingStore()
    worker, _ = _worker(snapshot, store=store, queue=queue)
    for line in NEW:
        worker.submit(_pending(line))
    store.fail_next = True
    with pytest.raises(ConnectionError):
        worker.run_cycle()
    assert queue.stats().committed == 0
    # The registry already has the template; the redelivered batch resolves at once.
    report = worker.run_cycle()
    assert report.buffered == len(NEW) and report.resolved == len(NEW)
    assert queue.stats().committed == len(NEW)


def test_rows_that_keep_failing_validation_are_dead_lettered(snapshot):
    queue = MemoryQueue()
    worker, _ = _worker(snapshot, queue=queue, max_attempts=2)
    # Every token differs, so the generalised template is all wildcards: rejected each cycle.
    for i in range(3):
        worker.submit(_pending(f"{i * 7919} {i * 104729} {i * 1299709}"))
    first = worker.run_cycle()
    assert first.rejected and first.still_pending == 3 and first.dead_lettered == 0
    # Re-queued with the attempt counted, in the same order.
    assert [m.attempts for m in queue.consume(10)] == [1, 1, 1]
    queue.commit()
    for i in range(3):
        queue.publish(_pending(f"{i * 7919} {i * 104729} {i * 1299709}"), attempts=1)
    second = worker.run_cycle()
    assert second.dead_lettered == 3 and second.still_pending == 0
    assert len(queue.dead) == 3 and queue.dead[0][1].startswith("rejected")
    assert queue.consume(10) == []


def test_deferred_clusters_do_not_count_as_an_attempt(snapshot):
    queue = MemoryQueue()
    worker, _ = _worker(snapshot, queue=queue, min_cluster_size=2, max_attempts=1)
    worker.submit(_pending(NEW[0]))
    report = worker.run_cycle()
    assert report.deferred == 1 and report.still_pending == 1 and report.dead_lettered == 0
    (message,) = queue.consume(1)
    assert message.attempts == 0


def test_rehydrate_requeues_pending_rows(snapshot):
    worker, store = _worker(snapshot)
    store.insert_logs([_pending(NEW[0]), _pending(NEW[1])])
    assert worker.rehydrate() == 2
    assert worker.run_cycle().resolved == 2
