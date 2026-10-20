"""Round trip through a real Kafka broker (``pytest -m integration``).

Locally ``make test-integration`` starts the compose broker first; in CI the
integration job provides a KRaft service container.  Topics are created per
test run (the broker has ``auto.create.topics.enable``), so runs do not see
each other's messages.
"""

from __future__ import annotations

import os
import socket
import uuid
from datetime import UTC, datetime

import pytest

from logpipe.enrich.llm import OfflineAdapter
from logpipe.enrich.worker import EnrichmentWorker
from logpipe.parsing.frozen_parser import ParseResult, ParserHandle, Status
from logpipe.registry import SchemaRegistry
from logpipe.storage.base import LogRow
from logpipe.storage.memory import MemoryStore
from logpipe.unresolved.kafka import KafkaQueue

pytestmark = pytest.mark.integration

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "127.0.0.1:9092")
TS = datetime(2026, 4, 13, 10, 30, tzinfo=UTC)
NEW = [
    f"2026-04-13 10:30:{i:02d}.000 [ERROR] OTA update failed code=0x{i:02X} retry={i}"
    for i in range(1, 6)
]


def _pending(line: str) -> LogRow:
    return LogRow.from_result(
        ParseResult(0, Status.PENDING, {}, registry_version=0),
        device_id="thermostat-TH-1",
        timestamp=TS,
        raw_message=line,
    )


@pytest.fixture(scope="module")
def broker() -> str:
    host, _, port = BOOTSTRAP.rpartition(":")
    try:
        with socket.create_connection((host or "127.0.0.1", int(port)), timeout=3):
            pass
    except OSError as exc:  # pragma: no cover - env dependent
        pytest.skip(f"Kafka not reachable at {BOOTSTRAP}: {exc}")
    return BOOTSTRAP


@pytest.fixture
def topic() -> str:
    return f"ci.unresolved.{uuid.uuid4().hex[:8]}"


def _wait_for(fn, attempts: int = 20):
    """Consumer group assignment takes a few polls on a fresh topic."""
    result = fn()
    while not result and attempts > 0:
        attempts -= 1
        result = fn()
    return result


def test_publish_consume_commit_round_trip(broker, topic):
    producer = KafkaQueue(broker, topic, "g-" + topic, client_id="ci-fast-path")
    rows = [_pending(line) for line in NEW]
    for row in rows:
        producer.publish(row, attempts=1)
    producer.flush()
    assert producer.stats().published == len(rows) and producer.stats().spilled == 0

    worker = KafkaQueue(broker, topic, "g-" + topic, client_id="ci-worker")
    batch = _wait_for(lambda: worker.consume(100, timeout=2.0))
    while len(batch) < len(rows):
        more = worker.consume(100, timeout=2.0)
        if not more:
            break
        batch.extend(more)
    assert sorted(m.row.raw_message for m in batch) == sorted(NEW)
    assert {m.attempts for m in batch} == {1}
    worker.commit()
    assert worker.stats().committed == len(rows)
    assert worker.lag() == 0

    # A new consumer in the same group starts after the committed offsets.
    again = KafkaQueue(broker, topic, "g-" + topic, client_id="ci-worker-2")
    worker.close()
    assert _wait_for(lambda: again.consume(100, timeout=2.0), attempts=3) == []
    again.close()
    producer.close()


def test_worker_cycle_on_kafka_backfills_and_dead_letters(broker, topic, snapshot):
    fast_path = KafkaQueue(broker, topic, "g-" + topic, client_id="ci-fast-path")
    store = MemoryStore()
    rows = [_pending(line) for line in NEW] + [_pending("98765 43210 13579")]
    store.insert_logs(rows)
    for row in rows:
        fast_path.publish(row)
    fast_path.flush()

    queue = KafkaQueue(broker, topic, "g-" + topic, client_id="ci-worker")
    worker = EnrichmentWorker(
        SchemaRegistry(snapshot),
        ParserHandle(snapshot),
        store,
        OfflineAdapter(),
        queue=queue,
        max_attempts=1,
        consume_timeout=3.0,
    )
    report = _wait_for(lambda: worker.run_cycle(), attempts=10)
    total_resolved, total_dead = report.resolved, report.dead_lettered
    while total_resolved + total_dead < len(rows):
        extra = worker.run_cycle()
        if not extra.buffered:
            break
        total_resolved += extra.resolved
        total_dead += extra.dead_lettered
    assert total_resolved == len(NEW)
    assert total_dead == 1  # the all-numbers line fails validation and goes to the DLQ
    assert store.count_by_status() == {"FINAL": len(NEW), "PENDING": 1}

    dlq = KafkaQueue(broker, queue.dlq_topic, "dlq-" + topic, client_id="ci-dlq")
    dead = _wait_for(lambda: dlq.consume(10, timeout=2.0))
    assert [m.row.raw_message for m in dead] == ["98765 43210 13579"]
    dlq.close()
    queue.close()
    fast_path.close()
