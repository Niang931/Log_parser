"""In-process store with ReplacingMergeTree semantics, for tests and offline runs."""
from __future__ import annotations

import threading
import uuid
from collections import Counter
from collections.abc import Sequence

from ..parsing.frozen_parser import Status
from .base import AnomalyAlert, LogRow


class MemoryStore:
    def __init__(self) -> None:
        self.rows: dict[uuid.UUID, LogRow] = {}
        self.alerts: list[AnomalyAlert] = []
        self.inserts = 0
        self._lock = threading.Lock()

    def insert_logs(self, rows: Sequence[LogRow]) -> None:
        with self._lock:
            for row in rows:
                self.inserts += 1
                current = self.rows.get(row.log_id)
                # Same tie-break as ReplacingMergeTree: highest version, then last insert.
                if current is None or row.version >= current.version:
                    self.rows[row.log_id] = row

    def insert_alerts(self, alerts: Sequence[AnomalyAlert]) -> None:
        with self._lock:
            self.alerts.extend(alerts)

    def fetch_pending(self, limit: int = 10_000) -> list[LogRow]:
        with self._lock:
            pending = [r for r in self.rows.values() if r.status == Status.PENDING]
        return sorted(pending, key=lambda r: r.timestamp)[:limit]

    def count_by_status(self) -> dict[str, int]:
        with self._lock:
            return dict(Counter(str(r.status) for r in self.rows.values()))

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
