"""The ingest loop: parse → store → detect, with periodic enrichment cycles.

PENDING rows go to `class UnresolvedQueue`.  With the in-process queue the enrichment worker runs inside this loop every ``enrich_every`` lines;
with Kafka this process can be a pure fast path (``worker=None``) and ``logpipe worker`` processes consume the topic.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .anomaly.detector import DetectorConfig, StreamingDetector
from .anomaly.scorer import Scorer
from .enrich.llm import LLMAdapter
from .enrich.worker import CycleReport, EnrichmentWorker
from .observability import Datadog
from .parsing.frozen_parser import ParseResult, ParserHandle, Status
from .progress import progress
from .registry import SchemaRegistry
from .sources.file_replay import LogEvent
from .sources.timestamps import event_time
from .storage.base import AnomalyAlert, LogRow, LogStore
from .unresolved import QueueStats, UnresolvedQueue


@dataclass
class PipelineStats:
    lines: int = 0
    final: int = 0
    pending: int = 0
    alerts: list[AnomalyAlert] = field(default_factory=list)
    cycles: list[CycleReport] = field(default_factory=list)
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

    def latency_us(self) -> dict[str, float]:
        if not self.parse_ns:
            return {}
        us = np.asarray(self.parse_ns) / 1000
        return {f"p{q}": round(float(np.percentile(us, q)), 2) for q in (50, 90, 99)} | {
            "max": round(float(us.max()), 2)
        }

    def summary(self) -> dict:
        resolved = sum(c.resolved for c in self.cycles)
        return {
            "lines": self.lines,
            "final_at_ingest": self.final,
            "pending_at_ingest": self.pending,
            "resolved_by_enrichment": resolved,
            "dead_lettered": sum(c.dead_lettered for c in self.cycles),
            "templates_learned": sum(len(c.published) for c in self.cycles),
            "aliases": sum(len(c.aliased) for c in self.cycles),
            "adapter_calls": sum(c.llm_calls for c in self.cycles),
            "alerts": len(self.alerts),
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
        datadog: Datadog | None = None,
    ):
        self.handle = handle
        self.datadog = datadog or Datadog()
        self.store = store
        self.worker = worker
        self.detector = detector
        self.enrich_every = enrich_every
        # Unmatched lines this says to hold stay PENDING in the store instead of going to the
        # online learner (the server holds a new source's lines until its templates are reviewed).
        self.hold: Callable[[LogEvent], bool] | None = None
        # PENDING rows go here. Defaults to the worker's own queue; a queue without a worker is the publish-only fast path (workers run elsewhere).
        self.queue: UnresolvedQueue | None = (
            queue if queue is not None else (worker.queue if worker is not None else None)
        )
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
        publish_lock: AbstractContextManager | None = None,
        datadog: Datadog | None = None,
    ) -> Pipeline:
        """Wire parser, enrichment and detector around one live registry.

        No ``adapter`` means no enrichment loop in this process: PENDING rows are only published to ``queue`` for external workers. No ``scorer`` means no detector.
        With adapter and scorer, the detector re-resolves PENDING lines against the latest registry when their window is scored.
        """
        handle = ParserHandle(registry.current)
        worker = (
            EnrichmentWorker(
                registry,
                handle,
                store,
                adapter,
                registry_path=registry_path,
                queue=queue,
                publish_lock=publish_lock,
            )
            if adapter is not None
            else None
        )
        detector = None
        if scorer is not None:
            detector = StreamingDetector(
                scorer,
                detector_config,
                registry_version=lambda: handle.version,
                resolver=(lambda line: handle.parse(line).template_id) if worker else None,
            )
        return cls(
            handle,
            store,
            worker=worker,
            detector=detector,
            enrich_every=enrich_every,
            queue=queue,
            datadog=datadog,
        )

    def process(self, event: LogEvent) -> ParseResult:
        dd = self.datadog
        start = time.perf_counter_ns()
        with dd.stage("parse"):
            result = self.handle.parse(event.line)
        self.stats.record_latency(time.perf_counter_ns() - start)

        # Agents send their local offset: a zone-less time in the line is that host's wall clock.
        local = event.time.tzinfo if event.time is not None and event.time.tzinfo else UTC
        timestamp = event_time(
            event.line, result.payload, fallback=event.time or datetime.now(UTC), local=local
        ).astimezone(UTC)
        self._event_time = timestamp
        row = LogRow.from_result(
            result,
            device_id=event.device_id,
            timestamp=timestamp,
            raw_message=event.line,
            service=event.service,
            source=event.source,
            tags=event.tags,
        )
        with dd.stage("store"):
            self.store.insert_logs([row])
        if dd.logging_on():
            entry = self.handle.snapshot.templates.get(result.template_id)
            dd.log(
                message=event.line,
                service=event.service or dd.service,
                device_id=event.device_id,
                template_id=result.template_id,
                template=" ".join(entry.tokens) if entry else None,
                parse_status=str(result.status),
            )
        self.stats.lines += 1
        if result.status == Status.FINAL:
            self.stats.final += 1
            dd.incr("lines", 1, "status:final")
        else:
            self.stats.pending += 1
            dd.incr("lines", 1, "status:pending")
            if self.queue is not None and not (self.hold and self.hold(event)):
                self.queue.publish(row)

        if self.detector is not None:
            with dd.stage("detect"):
                alerts = self.detector.observe(
                    event.device_id, result.template_id, timestamp, event.line
                )
            self._alerts(alerts)
        if self.worker is not None and self.stats.lines % self.enrich_every == 0:
            self.enrich()
        return result

    def enrich(self) -> CycleReport | None:
        if self.worker is None:
            return None
        self.store.flush()
        with self.datadog.stage("enrich"):
            report = self.worker.run_cycle(now=self._event_time)
        if report.buffered == 0:
            return None
        self.stats.cycles.append(report)
        return report

    def _alerts(self, alerts: list[AnomalyAlert]) -> None:
        if alerts:
            self.store.insert_alerts(alerts)
            self.stats.alerts.extend(alerts)
            self.datadog.incr("anomalies", len(alerts))
            self.datadog.log_alerts(alerts)

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
