"""Streaming anomaly detection: per-device reorder buffer, sliding windows, batched LogBERT.

Device clocks drift and MQTT reorders across reconnects, so events wait in
a small per-device heap until they are older than the newest event minus
``lateness``.  Released events feed a sliding window per device.  Every
``stride`` events a window is queued, and queued windows are scored in
batches across devices: one forward pass per ``batch_size`` windows, not
one per event.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from ..registry import UniqueTemplateID
from ..storage.base import AnomalyAlert
from .scorer import LogBERTScorer, WindowScore


def sliding_windows(seq: Sequence, size: int, stride: int) -> list[list]:
    if len(seq) <= size:
        return [list(seq)] if seq else []
    windows = [list(seq[i : i + size]) for i in range(0, len(seq) - size + 1, stride)]
    if (len(seq) - size) % stride:
        windows.append(list(seq[-size:]))
    return windows


@dataclass
class DetectorConfig:
    window: int = 20
    stride: int = 5
    min_len: int = 5
    batch_size: int = 64
    lateness: timedelta = timedelta(seconds=2)
    # TODO: increasing these 2 might cause problems for security intensive systems so I need to reconsider it
    # minimum number of distinct devices that must report a template
    fleet_min_devices: int = 2
    # minimum number of minutes a single device a single device must record the template
    fleet_min_minutes: int = 10
    record_windows: bool = False


@dataclass(frozen=True)
class ScoredWindow:
    device_id: str
    template_ids: tuple[int, ...]
    start: datetime
    end: datetime
    reasons: tuple[str, ...]
    score: WindowScore


@dataclass
class _Window:
    device_id: str
    template_ids: tuple[int, ...]
    start: datetime
    end: datetime
    pending_lines: tuple[str | None, ...]
    times: tuple[datetime, ...]


class StreamingDetector:
    def __init__(
        self,
        scorer: LogBERTScorer,
        config: DetectorConfig | None = None,
        registry_version: Callable[[], int] = lambda: 0,
        resolver: Callable[[str], int] | None = None,
    ):
        """``resolver`` re-parses a line that was PENDING at ingest with the latest
        registry when its window is scored, so templates the enrichment loop has
        learned by then are scored (and checked for fleet spread) under their id."""
        self.scorer = scorer
        self.config = config or DetectorConfig()
        self.registry_version = registry_version
        self.resolver = resolver
        self._heaps: dict[str, list[tuple[datetime, int, int, str | None]]] = {}
        self._newest: dict[str, datetime] = {}
        self._windows: dict[str, deque[tuple[int, datetime, str | None]]] = {}
        self._since_emit: dict[str, int] = {}
        self._queue: list[_Window] = []
        self._seq = itertools.count()
        # TODO: this seemed rather confusing, should consider making a new class just for it
        self._spread: dict[int, dict[str, set[int]]] = {}
        self.windows_scored = 0
        self.scored: list[ScoredWindow] = []

    def observe(
        self, device_id: str, template_id: int, timestamp: datetime, line: str | None = None
    ) -> list[AnomalyAlert]:
        template_id = int(template_id)
        self._note_device(template_id, device_id, timestamp)
        pending = line if template_id == UniqueTemplateID.UNK_TEMPLATE_ID else None

        # Push to per-device min-heap
        heap = self._heaps.setdefault(device_id, [])
        heapq.heappush(heap, (timestamp, next(self._seq), template_id, pending))

        # track the newest timestamp seen for this device
        newest = max(self._newest.get(device_id, timestamp), timestamp)
        self._newest[device_id] = newest

        # release events that are older than (newest - lateness)
        while heap and heap[0][0] <= newest - self.config.lateness:
            ts, _, tid, pending_line = heapq.heappop(heap)
            self._release(device_id, tid, ts, pending_line)

        # score the queue if its size is enough to batch size
        if len(self._queue) >= self.config.batch_size:
            return self._score_queue()

        return []

    def _note_device(self, template_id: int, device_id: str, timestamp: datetime) -> None:
        """Handle adding metadata for device log occurence. _spread structure would look like:
        {
            template_id: {
                device_id: set(),
            }
        }
        the set() is recording for unique minutes when a log template is recorded
        Example: If a log template repeated for 10 hours, then high chances are it
        would be occuring at different minutes (0-60) and we use that to determine
        its occurence frequency instead.
        """
        if template_id in self.scorer.vocab:
            return

        minutes = self._spread.setdefault(template_id, {}).setdefault(device_id, set())
        if len(minutes) < self.config.fleet_min_minutes:
            # convert raw timestamp to specific minute
            minutes.add(int(timestamp.timestamp() // 60))

    def _release(self, device_id: str, template_id: int, ts: datetime, line: str | None) -> None:
        # the deque (double-ended-queue) automatically drop the older logs whie keeping the latest only
        window = self._windows.setdefault(device_id, deque(maxlen=self.config.window))
        window.append((template_id, ts, line))

        # wait for stride new events to arrive before freezing
        # the current window and pushing to the queue via _enqueue
        # count is the number of templates pushed into a device buffer
        count = self._since_emit.get(device_id, 0) + 1
        if len(window) == self.config.window and count >= self.config.stride:
            self._enqueue(device_id, window)
            count = 0
        self._since_emit[device_id] = count

    def _enqueue(self, device_id: str, window: deque[tuple[int, datetime, str | None]]) -> None:
        templates, timestamps, lines = zip(*window)

        self._queue.append(
            _Window(
                device_id=device_id,
                template_ids=templates,
                start=timestamps[0],
                end=timestamps[-1],
                pending_lines=lines,
                times=timestamps,
            )
        )

    def _resolve(self, window: _Window) -> _Window:
        if not any(window.pending_lines):
            return window
        ids = []
        for tid, line, ts in zip(window.template_ids, window.pending_lines, window.times):
            if line is not None:
                assert self.resolver is not None
                tid = int(self.resolver(line))
                # The event's own time: overlapping windows must not inflate its spread.
                self._note_device(tid, window.device_id, ts)
            ids.append(tid)

        # NOTE: this create a new object, retain all attrs and overwrite the specified ones
        # this avoid having to rewrite a bunch of boilerplate code
        return replace(window, template_ids=tuple(ids))

    def familiar(self) -> frozenset[int]:
        """Templates unknown to the model but already seen across the fleet."""
        n, k = self.config.fleet_min_devices, self.config.fleet_min_minutes

        if n <= 0:
            return frozenset()

        # return frozen set of known templates (has registered id)
        # and have occured across multiple (n) devices and been recorded
        # at different more than (k) unique minutes.
        return frozenset(
            tid
            for tid, devices in self._spread.items()
            if tid != UniqueTemplateID.UNK_TEMPLATE_ID
            and sum(1 for minutes in devices.values() if len(minutes) >= k) >= n
        )

    def flush(self) -> list[AnomalyAlert]:
        """Release every buffered event and score the trailing partial windows."""
        for device_id, heap in self._heaps.items():
            while heap:
                ts, _, tid, line = heapq.heappop(heap)
                self._release(device_id, tid, ts, line)

        for device_id, window in self._windows.items():
            if len(window) >= self.config.min_len and self._since_emit.get(device_id, 0):
                self._enqueue(device_id, window)
                self._since_emit[device_id] = 0

        return self._score_queue()

    def _score_queue(self) -> list[AnomalyAlert]:
        # NOTE: drain and reset pattern, create snapshot, remove pointer and leave gc handle
        queue, self._queue = self._queue, []

        if not queue:
            return []

        # fetches only the most recent batches
        if self.resolver is not None:
            queue = [self._resolve(w) for w in queue]

        scores = self.scorer.score([w.template_ids for w in queue], self.familiar())
        self.windows_scored += len(queue)

        alerts = []
        for window, score in zip(queue, scores):
            reasons = self.scorer.reasons(score)

            if self.config.record_windows:
                self.scored.append(
                    ScoredWindow(
                        device_id=window.device_id,
                        template_ids=window.template_ids,
                        start=window.start,
                        end=window.end,
                        reasons=reasons,
                        score=score,
                    )
                )

            if not reasons:
                continue

            alerts.append(
                AnomalyAlert(
                    device_id=window.device_id,
                    window_start=window.start,
                    window_end=window.end,
                    template_ids=window.template_ids,
                    miss_ratio=score.miss_ratio,
                    novelty_ratio=score.novelty_ratio,
                    distance=score.distance,
                    reasons=reasons,
                    had_unknowns=any(
                        t == UniqueTemplateID.UNK_TEMPLATE_ID or t not in self.scorer.vocab
                        for t in window.template_ids
                    ),
                    registry_version=self.registry_version(),
                    model_version=self.scorer.version,
                )
            )
        return alerts
