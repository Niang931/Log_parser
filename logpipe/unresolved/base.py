"""The unresolved queue: PENDING rows on their way to the learning worker.

Two code share this interface: `class MemoryQueue (memory.py)` and `class KafkaQueue` (kafka.py).

Delivery is *at least once*: the worker consumes a batch, backfills what it resolved, re-publishes or dead-letters the rest, and only then commits.
A crash in between redelivers the batch, which is safe because backfill is idempotent (``log_id`` + version in the ReplacingMergeTree).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

from ..storage.base import LogRow

DEFAULT_MAX_ATTEMPTS = 3


@dataclass(frozen=True, slots=True)
class Message:
    row: LogRow
    # Failed attemps to learn a template for this row (-> dead-letters when reaches MAX_ATTEMPTS)
    attempts: int = 0


@dataclass
class QueueStats:
    kind: str
    published: int = 0
    consumed: int = 0
    committed: int = 0
    dead_lettered: int = 0
    spilled: int = 0            # Rows in local now, the broker couldn't take them (Kafka only).
    dropped: int = 0            # Rows lost to a full local buffer, still PENDING in the store.
    lag: int | None = None      # Messages consumer not processed (None = unknown).

    def to_json(self) -> dict:
        return asdict(self)


class UnresolvedQueue(Protocol):
    def publish(self, row: LogRow, *, attempts: int = 0) -> None:
        """Enqueue without blocking the caller (Parser)."""
        ...

    def consume(self, max_messages: int, timeout: float = 1.0) -> list[Message]:
        """Next batch for worker, same batch again if not committed."""
        ...

    def commit(self) -> None:
        """Acknowledge the last consumed batch."""
        ...

    def dead_letter(self, message: Message, reason: str) -> None:
        ...

    def flush(self, timeout: float = 10.0) -> None:
        """Block until everything published so far is accepted (Parser)"""
        ...

    def stats(self) -> QueueStats:
        ...

    def close(self) -> None:
        ...
