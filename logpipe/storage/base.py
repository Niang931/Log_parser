"""Rows, alerts and the storage interface shared by every backend."""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Protocol

from ..parsing.frozen_parser import ParseResult, Status


@dataclass(frozen=True, slots=True)
class LogRow:
    log_id: uuid.UUID
    timestamp: datetime
    device_id: str
    template_id: int
    status: Status
    registry_version: int
    payload: dict[str, str]
    raw_message: str
    # ReplacingMergeTree version: the registry version that produced this
    # row, so a backfill made with a newer registry always wins.
    version: int = 0

    @classmethod
    def from_result(
        cls, result: ParseResult, *, device_id: str, timestamp: datetime, raw_message: str
    ) -> LogRow:
        return cls(
            log_id=uuid.uuid4(),
            timestamp=timestamp,
            device_id=device_id,
            template_id=int(result.template_id),
            status=result.status,
            registry_version=result.registry_version,
            payload=result.payload,
            raw_message=raw_message,
            version=result.registry_version,
        )

    def resolved(self, result: ParseResult) -> LogRow:
        """The FINAL version of this row, keeping its identity and sort key."""
        return replace(
            self,
            template_id=int(result.template_id),
            status=result.status,
            registry_version=result.registry_version,
            payload=result.payload,
            version=result.registry_version,
        )


@dataclass(frozen=True, slots=True)
class AnomalyAlert:
    device_id: str
    window_start: datetime
    window_end: datetime
    template_ids: tuple[int, ...]
    miss_ratio: float
    novelty_ratio: float
    distance: float | None
    reasons: tuple[str, ...]
    had_unknowns: bool
    registry_version: int
    model_version: str
    alert_id: uuid.UUID = field(default_factory=uuid.uuid4)


class LogStore(Protocol):
    def insert_logs(self, rows: Sequence[LogRow]) -> None: ...

    def insert_alerts(self, alerts: Sequence[AnomalyAlert]) -> None: ...

    def fetch_pending(self, limit: int = 10_000) -> list[LogRow]:
        """Latest-version rows still PENDING, used to rebuild the retry buffer on restart."""
        ...

    def count_by_status(self) -> dict[str, int]: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...
