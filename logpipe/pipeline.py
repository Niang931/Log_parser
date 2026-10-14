"""The ingest loop: parse → store → detect, with periodic enrichment cycles.

Kafka/MQTT sources slot in later; anything yielding
:class:`~logpipe.sources.file_replay.LogEvent` works.
"""
from __future__ import annotations

import random
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from .anomaly.detector import StreamingDetector
from .enrich.worker import CycleReport, EnrichmentWorker
from .parsing.frozen_parser import ParseResult, ParserHandle, Status
from .sources.file_replay import LogEvent
from .sources.timestamps import event_time
from .storage.base import AnomalyAlert, LogRow, LogStore


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
            "templates_learned": sum(len(c.published) for c in self.cycles),
            "aliases": sum(len(c.aliased) for c in self.cycles),
            "adapter_calls": sum(c.llm_calls for c in self.cycles),
            "alerts": len(self.alerts),
            "parse_latency_us": self.latency_us(),
            "lines_per_second": round(self.lines / self.seconds, 1) if self.seconds else None,
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
    ):
        self.handle = handle
        self.store = store
        self.worker = worker
        self.detector = detector
        self.enrich_every = enrich_every
        self.stats = PipelineStats()
        self._event_time: datetime | None = None

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
            if self.worker is not None:
                self.worker.submit(row)

        if self.detector is not None:
            self._alerts(self.detector.observe(
                event.device_id, result.template_id, timestamp, event.line
            ))
        if self.worker is not None and self.stats.lines % self.enrich_every == 0:
            self.enrich()
        return result

    def enrich(self) -> CycleReport | None:
        if self.worker is None or not len(self.worker):
            return None
        self.store.flush()
        report = self.worker.run_cycle(now=self._event_time)
        self.stats.cycles.append(report)
        return report

    def _alerts(self, alerts: list[AnomalyAlert]) -> None:
        if alerts:
            self.store.insert_alerts(alerts)
            self.stats.alerts.extend(alerts)

    def run(self, events: Iterable[LogEvent]) -> PipelineStats:
        started = time.perf_counter()
        for event in events:
            self.process(event)
        self.finish()
        self.stats.seconds = time.perf_counter() - started
        return self.stats

    def finish(self) -> None:
        self.store.flush()
        self.enrich()
        if self.detector is not None:
            self._alerts(self.detector.flush())
        self.store.flush()
