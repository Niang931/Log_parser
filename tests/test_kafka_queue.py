"""KafkaQueue logic against fake clients: spill on broker failure, headers, dead letters.

The real broker is covered by ``tests/test_kafka_integration.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from logpipe.parsing.frozen_parser import ParseResult, Status
from logpipe.storage.base import LogRow
from logpipe.unresolved.codec import decode_row
from logpipe.unresolved.kafka import KafkaQueue

TS = datetime(2026, 4, 13, tzinfo=UTC)


def _pending(line: str) -> LogRow:
    return LogRow.from_result(
        ParseResult(0, Status.PENDING, {}, registry_version=1),
        device_id="d1",
        timestamp=TS,
        raw_message=line,
    )


class FakeRecord:
    def __init__(self, topic, key, value, headers):
        self._topic, self._key, self._value, self._headers = topic, key, value, headers

    def topic(self):
        return self._topic

    def key(self):
        return self._key

    def value(self):
        return self._value

    def headers(self):
        return self._headers

    def error(self):
        return None


class FakeProducer:
    """Delivers on ``poll``; ``fail`` makes every delivery report an error."""

    def __init__(self, conf):
        self.conf = conf
        self.fail = False
        self.full = False
        self.sent: dict[str, list[FakeRecord]] = {}
        self._callbacks = []

    def produce(self, topic, value, key=None, headers=None, on_delivery=None):
        if self.full:
            raise BufferError("Local: Queue full")
        record = FakeRecord(topic, key, value, headers)
        self._callbacks.append((on_delivery, record))

    def poll(self, timeout=0):
        callbacks, self._callbacks = self._callbacks, []
        for callback, record in callbacks:
            if self.fail:
                callback("Local: Message timed out", record)
            else:
                self.sent.setdefault(record.topic(), []).append(record)
                callback(None, record)
        return len(callbacks)

    def flush(self, timeout=None):
        self.poll()
        return len(self._callbacks)

    def __len__(self):
        return len(self._callbacks)


class FakeConsumer:
    def __init__(self, conf, records=()):
        self.conf = conf
        self.records = list(records)
        self.subscribed = None
        self.commits = 0
        self.closed = False

    def subscribe(self, topics):
        self.subscribed = topics

    def consume(self, num_messages=1, timeout=1.0):
        batch, self.records = self.records[:num_messages], self.records[num_messages:]
        return batch

    def commit(self, asynchronous=True):
        self.commits += 1

    def assignment(self):
        return []

    def close(self):
        self.closed = True


def _queue(records=(), **options) -> tuple[KafkaQueue, FakeProducer, FakeConsumer]:
    clients: dict = {}

    def make_producer(conf):
        clients["producer"] = FakeProducer(conf)
        return clients["producer"]

    def make_consumer(conf):
        clients["consumer"] = FakeConsumer(conf, records)
        return clients["consumer"]

    queue = KafkaQueue(
        "broker:9092",
        "t",
        "g",
        producer_factory=make_producer,
        consumer_factory=make_consumer,
        **options,
    )
    queue.producer, queue.consumer  # noqa: B018 - create both fakes
    return queue, clients["producer"], clients["consumer"]


def test_publish_sets_key_and_attempts_header_and_never_blocks():
    queue, producer, _ = _queue()
    row = _pending("Watchdog reset reason=0x1F")
    queue.publish(row, attempts=2)
    queue.flush()
    (record,) = producer.sent["t"]
    assert decode_row(record.value()) == row
    assert record.headers() == [("attempts", b"2")]
    assert record.key() and producer.conf["enable.idempotence"] is True
    assert queue.stats().published == 1 and queue.stats().spilled == 0


def test_delivery_failure_spills_locally_alerts_once_and_drains_when_the_broker_is_back():
    alerts: list[str] = []
    queue, producer, _ = _queue(on_alert=alerts.append, retry_after=0.0)
    producer.fail = True
    rows = [_pending(f"line {i}") for i in range(3)]
    for row in rows:
        queue.publish(row)
    assert queue.spilled_now == 3 and len(alerts) == 1 and "buffering locally" in alerts[0]
    assert queue.stats().spilled == 3 and "t" not in producer.sent

    producer.fail = False
    queue.publish(_pending("line 3"))  # drains the spill on the next publish
    queue.flush()
    delivered = {decode_row(r.value()).raw_message for r in producer.sent["t"]}
    assert delivered == {"line 0", "line 1", "line 2", "line 3"}
    assert queue.spilled_now == 0 and queue.stats().spilled == 0 and queue.stats().dropped == 0


def test_full_send_buffer_spills_and_a_full_spill_drops_the_oldest():
    queue, producer, _ = _queue(spill_size=2, retry_after=60.0)
    producer.full = True
    for i in range(3):
        queue.publish(_pending(f"line {i}"))
    stats = queue.stats()
    assert queue.spilled_now == 2 and stats.spilled == 2 and stats.dropped == 1


def test_consume_reads_attempts_and_commit_is_explicit():
    row = _pending("x")
    from logpipe.unresolved.codec import encode_row

    records = [FakeRecord("t", b"k", encode_row(row), [("attempts", b"1")])]
    queue, _, consumer = _queue(records)
    (message,) = queue.consume(10)
    assert message.row == row and message.attempts == 1
    assert consumer.subscribed == ["t"] and consumer.conf["enable.auto.commit"] is False
    assert consumer.commits == 0
    queue.commit()
    assert consumer.commits == 1 and queue.stats().committed == 1
    queue.commit()  # nothing new: no second round trip
    assert consumer.commits == 1


def test_undecodable_records_go_to_the_dead_letter_topic():
    records = [FakeRecord("t", b"k", b"not json", None)]
    queue, producer, _ = _queue(records)
    assert queue.consume(10) == []
    queue.flush()
    (dead,) = producer.sent["t.dlq"]
    assert dead.value() == b"not json"
    assert dict(dead.headers())["reason"].startswith(b"undecodable")


def test_dead_letter_carries_reason_and_attempts():
    queue, producer, _ = _queue()
    from logpipe.unresolved.base import Message

    queue.dead_letter(Message(_pending("bad"), attempts=3), "rejected: all wildcards")
    queue.flush()
    (dead,) = producer.sent["t.dlq"]
    assert dict(dead.headers()) == {"attempts": b"3", "reason": b"rejected: all wildcards"}
    assert queue.stats().dead_lettered == 1


def test_from_env_reads_kafka_settings():
    env = {
        "KAFKA_BOOTSTRAP": "k:9092",
        "KAFKA_TOPIC": "u",
        "KAFKA_GROUP": "w",
        "KAFKA_KEY_BY": "device",
    }
    queue = KafkaQueue.from_env(env)
    assert (queue.bootstrap, queue.topic, queue.group, queue.dlq_topic, queue.key_by) == (
        "k:9092",
        "u",
        "w",
        "u.dlq",
        "device",
    )
