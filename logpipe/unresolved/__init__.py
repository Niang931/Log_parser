"""Unresolved queue: PENDING rows on their way from the fast path to the learning worker."""

from .base import DEFAULT_MAX_ATTEMPTS, Message, QueueStats, UnresolvedQueue
from .memory import MemoryQueue

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "Message",
    "MemoryQueue",
    "QueueStats",
    "UnresolvedQueue",
    "open_queue",
]


def open_queue(kind: str, *, max_size: int = 200_000, **kafka_options) -> UnresolvedQueue:
    """``memory`` (one process) or ``kafka`` (settings from ``KAFKA_*`` in the environment)."""
    if kind == "memory":
        return MemoryQueue(max_size)
    if kind == "kafka":
        from .kafka import KafkaQueue

        return KafkaQueue.from_env(**kafka_options)
    raise ValueError(f"unknown queue {kind!r}; expected 'memory' or 'kafka'")
