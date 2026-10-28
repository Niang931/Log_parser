"""In-process unresolved queue: the single-process default and the test double."""

from __future__ import annotations

import threading
from collections import deque

from ..storage.base import LogRow
from .base import Message, QueueStats


class MemoryQueue:
    def __init__(self, max_size: int = 200_000):
        self.max_size = max_size
        self.dead: list[tuple[Message, str]] = []
        self._pending: deque[Message] = deque()
        self._in_flight: list[Message] = []
        self._stats = QueueStats(kind="memory")
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._pending)

    def publish(self, row: LogRow, *, attempts: int = 0) -> None:
        with self._lock:
            if len(self._pending) >= self.max_size:
                # Remove oldest row if full; `rehydrate` can bring back (already stored as PENDING).
                self._pending.popleft()
                self._stats.dropped += 1

            self._pending.append(Message(row, attempts))
            self._stats.published += 1

    def consume(self, max_messages: int, timeout: float = 1.0) -> list[Message]:
        with self._lock:
            if self._in_flight:  
                # redelivery: the previous batch was never committed
                return list(self._in_flight)
            
            batch = [self._pending.popleft() for _ in range(min(max_messages, len(self._pending)))]
            self._in_flight = batch
            self._stats.consumed += len(batch)
            return list(batch)

    def commit(self) -> None:
        with self._lock:
            self._stats.committed += len(self._in_flight)
            self._in_flight = []

    def dead_letter(self, message: Message, reason: str) -> None:
        with self._lock:
            self.dead.append((message, reason))
            self._stats.dead_lettered += 1

    def flush(self, timeout: float = 10.0) -> None:
        pass

    def stats(self) -> QueueStats:
        with self._lock:
            self._stats.lag = len(self._pending) + len(self._in_flight)
            return QueueStats(**self._stats.to_json())

    def close(self) -> None:
        pass
