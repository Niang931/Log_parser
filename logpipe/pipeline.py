"""The ingest loop: parse → store → detect, with periodic enrichment cycles.

PENDING rows go to `class UnresolvedQueue`.  With the in-process queue the enrichment worker runs inside this loop every ``enrich_every`` lines;
with Kafka this process can be a pure fast path (``worker=None``) and ``logpipe worker`` processes consume the topic.
"""
from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .anomaly.detector import DetectorConfig, StreamingDetector
from .anomaly.scorer import Scorer
from .enrich.llm import LLMAdapter
from .enrich.worker import CycleReport, EnrichmentWorker
from .parsing.frozen_parser import ParseResult, ParserHandle, Status
from .progress import progress
from .registry import SchemaRegistry
from .sources.file_replay import LogEvent
from .sources.timestamps import event_time
from .storage.base import AnomalyAlert, LogRow, LogStore
from .unresolved import QueueStats, UnresolvedQueue


ALERT_TAIL = 1_000
CYCLE_TAIL = 200


@dataclass
class PipelineStats:
    lines: int = 0
    final: int = 0
    pending: int = 0
    # Only the newest alerts/cycles are kept (a 24 h soak would grow them without bound);
    # the totals live in the counters below and feed summary().
    alerts: deque[AnomalyAlert] = field(default_factory=lambda: deque(maxlen=ALERT_TAIL))
    cycles: deque[CycleReport] = field(default_factory=lambda: deque(maxlen=CYCLE_TAIL))
    alert_count: int = 0
    cycle_count: int = 0
    resolved: int = 0
    dead_lettered: int = 0
    templates_learned: int = 0
    aliases: int = 0
    adapter_calls: int = 0
    # Reservoir sample of per-line parse latency, bounded for long runs.
    parse_ns: list[int] = field(default_factory=list)
    seconds: float = 0.0
    queue: QueueStats | None = None
    _rng: random.Random = field(default_factory=lambda: random.Random(0), repr=False)

    def record_latency(self, ns: int, cap: int = 500_000) -> None:
        if len(self.parse_ns) < cap:
            self.parse_ns.append(ns)
        elif (j := self._rng.randrange(self.lines + 1)) < cap:
            self.parse_ns[j] = ns

    def add_cycle(self, report: CycleReport) -> None:
        self.cycles.append(report)
        self.cycle_count += 1
        self.resolved += report.resolved
        self.dead_lettered += report.dead_lettered
        self.templates_learned += len(report.published)
        self.aliases += len(report.aliased)
        self.adapter_calls += report.llm_calls

    def add_alerts(self, alerts: list[AnomalyAlert]) -> None:
        self.alerts.extend(alerts)
        self.alert_count += len(alerts)

    def latency_us(self) -> dict[str, float]:
        if not self.parse_ns:
            return {}
        us = np.asarray(self.parse_ns) / 1000
        return {f"p{q}": round(float(np.percentile(us, q)), 2) for q in (50, 90, 99)} | {
            "max": round(float(us.max()), 2)
        }

    def summary(self) -> dict:
        return {
            "lines": self.lines,
            "final_at_ingest": self.final,
            "pending_at_ingest": self.pending,
            "resolved_by_enrichment": self.resolved,
            "dead_lettered": self.dead_lettered,
            "templates_learned": self.templates_learned,
            "aliases": self.aliases,
            "adapter_calls": self.adapter_calls,
            "alerts": self.alert_count,
            "parse_latency_us": self.latency_us(),
            "lines_per_second": round(self.lines / self.seconds, 1) if self.seconds else None,
            "unresolved_queue": self.queue.to_json() if self.queue else None,
        }


class Pipeline:
    def __init__(
        self,
        handle: ParserHandle,
        store: LogStore,
        *,
        worker: EnrichmentWorker | None = None,
        detector: StreamingDetector | None = None,
        enrich_every: int = 5_000,
        queue: UnresolvedQueue | None = None,
    ):
        self.handle = handle
        self.store = store
        self.worker = worker
        self.detector = detector
        self.enrich_every = enrich_every
        # PENDING rows go here. Defaults to the worker's own queue; a queue without a worker is the publish-only fast path (workers run elsewhere).
        self.queue: UnresolvedQueue | None = (queue if queue is not None else (worker.queue if worker is not None else None))
        self.stats = PipelineStats()
        self._event_time: datetime | None = None

    @classmethod
    def assemble(
        cls,
        registry: SchemaRegistry,
        store: LogStore,
        *,
        adapter: LLMAdapter | None = None,
        scorer: Scorer | None = None,
        detector_config: DetectorConfig | None = None,
        registry_path: Path | None = None,
        enrich_every: int = 5_000,
        queue: UnresolvedQueue | None = None,
    ) -> Pipeline:
        """Wire parser, enrichment and detector around one live registry.

        No ``adapter`` means no enrichment loop in this process: PENDING rows are only published to ``queue`` for external workers. No ``scorer`` means no detector.
        With adapter and scorer, the detector re-resolves PENDING lines against the latest registry when their window is scored.
        """
        handle = ParserHandle(registry.current)
        worker = (EnrichmentWorker(registry, handle, store, adapter, registry_path=registry_path, queue=queue)
            if adapter is not None
            else None
        )
        detector = None
        if scorer is not None:
            detector = StreamingDetector(
                scorer, detector_config,
                registry_version=lambda: handle.version,
                resolver=(lambda line: handle.parse(line).template_id) if worker else None,
            )
        return cls(handle, store, worker=worker, detector=detector, enrich_every=enrich_every, queue=queue)

    def process(self, event: LogEvent) -> ParseResult:
        start = time.perf_counter_ns()
        result = self.handle.parse(event.line)
        self.stats.record_latency(time.perf_counter_ns() - start)

        timestamp = event_time(event.line, result.payload, fallback=datetime.now(UTC))
        self._event_time = timestamp
        row = LogRow.from_result(
            result, device_id=event.device_id, timestamp=timestamp, raw_message=event.line
        )
        self.store.insert_logs([row])
        self.stats.lines += 1
        if result.status == Status.FINAL:
            self.stats.final += 1
        else:
            self.stats.pending += 1
            if self.queue is not None:
                self.queue.publish(row)

        if self.detector is not None:
            self._alerts(self.detector.observe(
                event.device_id, result.template_id, timestamp, event.line
            ))
        if self.worker is not None and self.stats.lines % self.enrich_every == 0:
            self.enrich()
        return result

    def enrich(self) -> CycleReport | None:
        if self.worker is None:
            return None
        self.store.flush()
        report = self.worker.run_cycle(now=self._event_time)
        if report.buffered == 0:
            return None
        self.stats.add_cycle(report)
        return report

    def _alerts(self, alerts: list[AnomalyAlert]) -> None:
        if alerts:
            self.store.insert_alerts(alerts)
            self.stats.add_alerts(alerts)

    def run(self, events: Iterable[LogEvent], *, total: int | None = None) -> PipelineStats:
        started = time.perf_counter()
        for event in progress(events, "ingest", total=total, unit="line"):
            self.process(event)
        self.finish()
        self.stats.seconds = time.perf_counter() - started
        return self.stats

    def finish(self) -> None:
        self.store.flush()
        if self.queue is not None:
            self.queue.flush()
        self.enrich()
        if self.detector is not None:
            self._alerts(self.detector.flush())
        self.store.flush()
        if self.queue is not None:
            self.stats.queue = self.queue.stats()
