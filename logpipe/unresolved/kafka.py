"""Durable unresolved queue on a Kafka topic (confluent-kafka).

Fast path (producer)
    ``publish`` hands the row to librdkafka's own send buffer and returns at once; the broker round trip happens in the background.
    When the broker can't take messages (buffer full / delivery timed out) the row is parked in a local *spill* buffer, an alert is raised once, and the
    spill is drained when deliveries succeed again. The row is already stored as PENDING, so even a spill overflow (``dropped``): it can be rehydrated from the store.

Learning workers (consumer)
    Workers form a consumer group on the topic.  Offsets are committed only after the backfill write succeeded (``commit`` is called by the worker at
    the end of a cycle), which gives at-least-once delivery.  Rows that fail ``MAX_ATTEMPTS`` -> ``<topic>.dlq`` with the reason in a header.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from collections.abc import Callable, Mapping

from ..storage.base import LogRow
from .base import Message, QueueStats
from .codec import decode_row, encode_row, partition_key

LOGGER = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP = "127.0.0.1:9092"
DEFAULT_TOPIC = "logpipe.unresolved"
DEFAULT_GROUP = "logpipe-workers"
ATTEMPTS_HEADER = "attempts"
REASON_HEADER = "reason"


def _attempts(headers: list[tuple[str, bytes | None]] | None) -> int:
    for key, value in headers or []:
        if key == ATTEMPTS_HEADER and value:
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


class KafkaQueue:
    def __init__(
        self,
        bootstrap: str = DEFAULT_BOOTSTRAP,
        topic: str = DEFAULT_TOPIC,
        group: str = DEFAULT_GROUP,
        *,
        dlq_topic: str | None = None,
        key_by: str = "shape",
        client_id: str = "logpipe",
        spill_size: int = 100_000,
        retry_after: float = 10.0,
        delivery_timeout_ms: int = 30_000,
        on_alert: Callable[[str], None] | None = None,
        producer_factory: Callable[[dict], object] | None = None,
        consumer_factory: Callable[[dict], object] | None = None,
    ):
        self.bootstrap = bootstrap
        self.topic = topic
        self.group = group
        self.dlq_topic = dlq_topic or f"{topic}.dlq"
        self.key_by = key_by
        self.client_id = client_id
        self.spill_size = spill_size
        self.retry_after = retry_after
        self.delivery_timeout_ms = delivery_timeout_ms
        self.on_alert = on_alert
        self._producer_factory = producer_factory
        self._consumer_factory = consumer_factory
        self._producer = None
        self._consumer = None
        self._parked: deque[tuple[str, bytes, bytes, list[tuple[str, bytes]]]] = deque()
        self._broker_down = False
        self._retry_at = 0.0
        self._uncommitted = 0
        self._stats = QueueStats(kind="kafka")

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ, **overrides) -> KafkaQueue:
        """``KAFKA_BOOTSTRAP``, ``KAFKA_TOPIC``, ``KAFKA_GROUP``, ``KAFKA_DLQ_TOPIC``, ``KAFKA_KEY_BY``."""
        options = dict(
            bootstrap=env.get("KAFKA_BOOTSTRAP", DEFAULT_BOOTSTRAP),
            topic=env.get("KAFKA_TOPIC", DEFAULT_TOPIC),
            group=env.get("KAFKA_GROUP", DEFAULT_GROUP),
            dlq_topic=env.get("KAFKA_DLQ_TOPIC") or None,
            key_by=env.get("KAFKA_KEY_BY", "shape"),
        )
        options.update(overrides)

        return cls(**options)

    # __clients_____________________
    @property
    def producer(self):
        if self._producer is None:
            conf = {
                "bootstrap.servers": self.bootstrap,
                "client.id": f"{self.client_id}-producer",
                "acks": "all",
                "enable.idempotence": True,
                "linger.ms": 5,
                "message.timeout.ms": self.delivery_timeout_ms,
                "socket.keepalive.enable": True,
            }
            if self._producer_factory is not None:
                self._producer = self._producer_factory(conf)
            else:
                from confluent_kafka import Producer

                self._producer = Producer(conf, logger=LOGGER)
        return self._producer

    @property
    def consumer(self):
        if self._consumer is None:
            conf = {
                "bootstrap.servers": self.bootstrap,
                "group.id": self.group,
                "client.id": f"{self.client_id}-consumer",
                "enable.auto.commit": False,
                "auto.offset.reset": "earliest",
                "allow.auto.create.topics": True,
                "socket.keepalive.enable": True,
            }
            if self._consumer_factory is not None:
                self._consumer = self._consumer_factory(conf)
            else:
                from confluent_kafka import Consumer

                self._consumer = Consumer(conf, logger=LOGGER)
            self._consumer.subscribe([self.topic])
        return self._consumer

    # __producer_____________________
    def publish(self, row: LogRow, *, attempts: int = 0) -> None:
        headers = [(ATTEMPTS_HEADER, str(attempts).encode())]
        self._produce(self.topic, partition_key(row, self.key_by), encode_row(row), headers)
        self._stats.published += 1

        self._drain_spill()
        self.producer.poll(0)  # serve delivery callbacks; never blocks

    def _produce(self, topic: str, key: bytes, value: bytes, headers: list[tuple[str, bytes]]) -> bool:
        try:
            self.producer.produce(topic, value, key=key, headers=headers, on_delivery=self._delivered)
            return True
        
        except BufferError:  # Buffer full: the broker is slow or gone
            self._spill(topic, key, value, headers, "local send buffer full")
        except Exception as error:
            self._spill(topic, key, value, headers, str(error))
        return False

    def _delivered(self, error, message) -> None:
        if error is not None:
            headers = [(k, v) for k, v in (message.headers() or []) if v is not None]
            self._spill(message.topic(), message.key(), message.value(), headers, str(error))
            return
        
        if self._broker_down:
            self._broker_down = False
            LOGGER.info("kafka: deliveries to %s succeed again", self.bootstrap)

    def _spill(self, topic, key, value, headers, reason: str) -> None:
        if len(self._parked) >= self.spill_size:
            self._parked.popleft()
            self._stats.dropped += 1

        self._parked.append((topic, key, value, headers))
        self._retry_at = time.monotonic() + self.retry_after
        
        if not self._broker_down:
            self._broker_down = True
            self._alert(f"Kafka: Can't deliver to {self.bootstrap} ({reason}); buffering locally")

    def _alert(self, text: str) -> None:
        LOGGER.error(text)
        if self.on_alert is not None:
            self.on_alert(text)

    def _drain_spill(self, limit: int = 1_000) -> int:
        if not self._parked or time.monotonic() < self._retry_at:
            return 0
        
        sent = 0
        while self._parked and sent < limit:
            topic, key, value, headers = self._parked.popleft()
            if not self._produce(topic, key, value, headers):
                break  # _produce re-spilled it and pushed the retry time out
            sent += 1
        return sent

    @property
    def spilled_now(self) -> int:
        return len(self._parked)

    def dead_letter(self, message: Message, reason: str) -> None:
        headers = [
            (ATTEMPTS_HEADER, str(message.attempts).encode()),
            (REASON_HEADER, reason.encode("utf-8", errors="replace")),
        ]
        self._produce(
            self.dlq_topic,
            partition_key(message.row, self.key_by),
            encode_row(message.row),
            headers,
        )
        self._stats.dead_lettered += 1
        self.producer.poll(0)

    def flush(self, timeout: float = 10.0) -> None:
        if self._producer is None:
            return
        
        deadline = time.monotonic() + timeout
        self._retry_at = 0.0
        self._drain_spill(limit=self.spill_size)
        while True:
            remaining = self._producer.flush(max(0.0, deadline - time.monotonic()))
            if remaining == 0 and not self._parked:
                return
            if time.monotonic() >= deadline:
                LOGGER.warning("Kafka: %d messages still undelivered after %.0fs (%d spilled)", remaining, timeout, len(self._parked))
                return
            
            self._retry_at = 0.0
            self._drain_spill(limit=self.spill_size)

    # __consumers_____________________
    def consume(self, max_messages: int, timeout: float = 1.0) -> list[Message]:
        raw = self.consumer.consume(num_messages=max(1, max_messages), timeout=timeout)
        batch: list[Message] = []
        for record in raw:
            if record.error():
                # A topic nobody has produced to yet is normal at startup, not a fault.
                if "UNKNOWN_TOPIC_OR_PART" in str(record.error()):
                    LOGGER.debug("Kafka: topic %s not created yet", self.topic)
                else:
                    LOGGER.warning("Kafka: consumer error %s", record.error())
                continue

            attempts = _attempts(record.headers())
            try:
                row = decode_row(record.value())
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                self._produce(
                    self.dlq_topic,
                    record.key() or b"",
                    record.value() or b"",
                    [
                        (ATTEMPTS_HEADER, str(attempts).encode()),
                        (REASON_HEADER, f"undecodable: {error}".encode()),
                    ],
                )
                self._stats.dead_lettered += 1
                self._uncommitted += 1
                continue
            batch.append(Message(row, attempts))
        
        self._uncommitted += len(batch)
        self._stats.consumed += len(batch)
        return batch

    def commit(self) -> None:
        if self._consumer is None or not self._uncommitted:
            return
        self._consumer.commit(asynchronous=False)
        self._stats.committed += self._uncommitted
        self._uncommitted = 0

    def lag(self) -> int | None:
        """Messages in the assigned partitions consumer has not reached yet."""
        if self._consumer is None:
            return None
        try:
            total = 0
            for partition in self._consumer.assignment():
                low, high = self._consumer.get_watermark_offsets(partition, timeout=2.0)
                (position,) = self._consumer.position([partition])
                current = position.offset if position.offset >= 0 else low
                total += max(0, high - current)
            return total
        except Exception as error:  # noqa: BLE001 - metrics must never break the loop
            LOGGER.debug("kafka: lag unavailable: %s", error)
            return None

    # __common_____________________
    def stats(self) -> QueueStats:
        snapshot = QueueStats(**self._stats.to_json())
        snapshot.spilled = len(self._parked)
        snapshot.lag = self.lag()
        return snapshot

    def close(self) -> None:
        self.flush()
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None
