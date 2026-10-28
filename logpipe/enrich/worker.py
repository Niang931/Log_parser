"""Async enrichment loop: learn templates for PENDING lines and backfill them.

One cycle:

1. Cluster the buffered PENDING lines with a *shadow* (learning) Drain built
   on the registry's masks, so a burst of 100k unknown lines costs one LLM
   call per new event type, not per line.
2. Validate each cluster's template (extractor must fit every sample, must
   not be all wildcards) and get key names from the LLM adapter.
3. Alias it to an existing template when :func:`find_alias` says so,
   otherwise register a new id.
4. Publish registry vN+1, hot-swap the parser, persist the registry.
5. Re-parse the batch with vN+1 and insert the resolved rows as new,
   higher-version rows (ReplacingMergeTree keeps the latest).
6. Re-queue what is still PENDING (or dead-letter it after ``max_attempts``
   failed), then commit the batch on the queue.

The PENDING lines come from an `class UnresolvedQueue` an in-process `class MemoryQueue`
by default, or a Kafka topic shared by a consumer group of workers.  The order in
step 5-6 (backfill, re-queue, *then* commit) is what makes delivery at-least-once: a crash
before the commit redelivers the batch, and the backfill is idempotent.

Masks are never changed here: a new mask would re-tokenise lines of
existing templates and silently break them.  Mask changes go through
offline recalibration.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..parsing.engine import make_engine
from ..parsing.frozen_parser import ParserHandle, Status
from ..registry import NewTemplate, SchemaRegistry
from ..storage.base import LogRow, LogStore
from ..unresolved import DEFAULT_MAX_ATTEMPTS, MemoryQueue, Message, UnresolvedQueue
from .llm import KeyVocabAware, LLMAdapter
from .similarity import find_alias
from .validate import clean_keys, generalize_digits, template_problem

LOGGER = logging.getLogger(__name__)


@dataclass
class CycleReport:
    buffered: int = 0
    clusters: int = 0
    published: list[int] = field(default_factory=list)
    aliased: dict[int, int] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)
    deferred: int = 0
    resolved: int = 0
    still_pending: int = 0
    dead_lettered: int = 0
    queue_lag: int | None = None
    llm_calls: int = 0
    llm_errors: int = 0
    registry_version: int = 0
    seconds: float = 0.0
    resolve_lag_sec: list[float] = field(default_factory=list)


class CircuitBreaker:
    """Stop calling the LLM after repeated failures, retry after ``cooldown`` cycles."""

    def __init__(self, failures: int = 3, cooldown: int = 5):
        self.threshold = failures
        self.cooldown = cooldown
        self.failures = 0
        self.wait_duration = 0

    @property
    def open(self) -> bool:
        return self.wait_duration > 0

    def tick(self) -> None:
        if self.wait_duration > 0:
            self.wait_duration -= 1

    def record(self, ok: bool) -> None:
        if ok:
            self.failures = 0
            return
        self.failures += 1
        if self.failures >= self.threshold:
            self.wait_duration, self.failures = self.cooldown, 0


class EnrichmentWorker:
    def __init__(
        self,
        registry: SchemaRegistry,
        handle: ParserHandle,
        store: LogStore,
        adapter: LLMAdapter,
        *,
        registry_path: Path | None = None,
        max_buffer: int = 200_000,
        min_cluster_size: int = 1,
        max_new_per_cycle: int = 50,
        alias_threshold: float = 0.75,
        drain_similarity: float = 0.4,
        on_publish: Callable[[int], None] | None = None,
        queue: UnresolvedQueue | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        consume_timeout: float = 1.0,
    ):
        self.registry = registry
        self.handle = handle
        self.store = store
        self.adapter = adapter
        self.registry_path = registry_path
        self.min_cluster_size = min_cluster_size
        self.max_new_per_cycle = max_new_per_cycle
        self.alias_threshold = alias_threshold
        self.drain_similarity = drain_similarity
        self.on_publish = on_publish
        self.breaker = CircuitBreaker()
        # Rows per cycle; with the in-process queue also its capacity.
        self.max_batch = max_buffer
        self.max_attempts = max_attempts
        self.consume_timeout = consume_timeout
        self.queue: UnresolvedQueue = queue if queue is not None else MemoryQueue(max_buffer)

    @property
    def dropped(self) -> int:
        """Rows the queue lost to a full local buffer (still PENDING in store)."""
        return self.queue.stats().dropped

    def submit(self, row: LogRow) -> None:
        self.queue.publish(row)

    def rehydrate(self, limit: int = 100_000) -> int:
        """Re-queue the store's PENDING rows

        A row that is also still in the queue is resolved twice, which is
        harmless: the backfill is idempotent.
        """
        rows = self.store.fetch_pending(limit)
        for row in rows:
            self.submit(row)
        return len(rows)

    def _call(self, report: CycleReport, fn, *args):
        """Call the adapter through the circuit breaker; ``None`` on failure."""
        if self.breaker.open:
            return None
        report.llm_calls += 1
        try:
            result = fn(*args)
        except Exception as error:  # noqa: BLE001 - vendor errors are not ours to enumerate
            LOGGER.warning("LLM call failed: %s", error)
            report.llm_errors += 1
            self.breaker.record(ok=False)
            return None
        self.breaker.record(ok=True)
        return result

    def run_cycle(self, now: datetime | None = None) -> CycleReport:
        # operation duration calculation
        started = time.perf_counter()

        batch = self.queue.consume(self.max_batch, timeout=self.consume_timeout)
        report = CycleReport(buffered=len(batch))
        self.breaker.tick()

        # nothing queued
        if not batch:
            report.registry_version = self.handle.version
            return report

        snapshot = self.registry.current
        if isinstance(self.adapter, KeyVocabAware):
            # Let the LLM reuse existing key names instead of inventing synonyms.
            self.adapter.key_vocab = sorted(snapshot.key_vocab)

        # create a new engine with the current mask snapshot to not interfere with the pipeline
        tmp_engine = make_engine(snapshot.masks, similarity_threshold=self.drain_similarity)
        members: dict[int, list[str]] = {}
        cluster_of: dict[int, int] = {}  # batch index -> shadow cluster id
        for index, message in enumerate(batch):
            cluster_id = tmp_engine.add_log(message.row.raw_message).cluster_id
            cluster_of[index] = cluster_id
            members.setdefault(cluster_id, []).append(message.row.raw_message)

        # sort from largest to smallest
        clusters = sorted(tmp_engine.clusters, key=lambda c: -len(members[c.cluster_id]))
        report.clusters = len(clusters)

        # What happened to each cluster, so a row still PENDING afterwards knows
        # whether this cycle counts as a failed attempt ("rejected", "tried") or
        # the cluster simply was not looked at ("deferred").
        outcome: dict[int, str] = {}
        new_templates: list[NewTemplate] = []
        seen_tokens: set[tuple[str, ...]] = set()
        for cluster in clusters:
            lines = members[cluster.cluster_id]
            tokens = generalize_digits(cluster.template)
            label = " ".join(tokens)

            # another shadow cluster generalised to the same template
            if tokens in seen_tokens:
                outcome[cluster.cluster_id] = "tried"
                continue

            if len(lines) < self.min_cluster_size:
                report.deferred += 1
                outcome[cluster.cluster_id] = "deferred"
                continue

            # TODO: Im not sure if I should check this or not, it would make sense for
            # more diverse ecosystem with different IoT but have to test the scale
            if len(new_templates) >= self.max_new_per_cycle:
                report.deferred += 1
                outcome[cluster.cluster_id] = "deferred"
                continue

            # template validation
            if problem := template_problem(tokens, lines):
                report.rejected[label] = problem
                outcome[cluster.cluster_id] = f"rejected: {problem}"
                continue

            seen_tokens.add(tokens)
            outcome[cluster.cluster_id] = "tried"

            # The cluster created for a new template gets compared with
            # pre-existing ones to see if we can merge them or not
            samples = lines[:5]
            decision = find_alias(
                tokens,
                snapshot,
                GuardedAdapter(self, report),
                samples,
                threshold=self.alias_threshold,
            )
            keys = clean_keys(self._call(report, self.adapter.name_keys, tokens, samples), tokens)
            # add the new template alias_of determine if its a new template or depends on existing one
            new_templates.append(NewTemplate(tokens, tuple(keys), decision.alias_of))
            if decision.alias_of is not None:
                report.aliased[len(new_templates) - 1] = decision.alias_of

        if new_templates:
            snapshot, ids = self.registry.publish(
                new_templates, key_vocab=[k for t in new_templates for k in t.keys]
            )
            report.published = ids
            report.aliased = {ids[i]: target for i, target in report.aliased.items()}
            
            # swapping the mainstream drain engine with new snapshoy 
            self.handle.swap(snapshot)

            # save the state if specified
            if self.registry_path is not None:
                self.registry.save(self.registry_path)

            # custom ending callback if specified
            if self.on_publish is not None:
                self.on_publish(snapshot.version)

        resolved: list[LogRow] = []
        remaining: list[tuple[Message, str]] = []

        # separating resolved and pending rows
        for index, message in enumerate(batch):
            result = self.handle.parse(message.row.raw_message)
            if result.status == Status.FINAL:
                resolved.append(message.row.resolved(result))
            else:
                remaining.append((message, outcome.get(cluster_of[index], "deferred")))

        # backfill all the resolved logs into store (dynamodb or clickhouse)
        if resolved:
            self.store.insert_logs(resolved)
            self.store.flush()

        # Re-queue the rest for re-processing
        for message, why in remaining:
            attempts = message.attempts + (0 if why == "deferred" else 1)
            if attempts >= self.max_attempts:
                self.queue.dead_letter(message, why)
                report.dead_lettered += 1
            else:
                self.queue.publish(message.row, attempts=attempts)
                report.still_pending += 1
        self.queue.commit()

        if now is not None:
            report.resolve_lag_sec = [(now - r.timestamp).total_seconds() for r in resolved]

        report.resolved = len(resolved)
        report.queue_lag = self.queue.stats().lag
        report.registry_version = self.handle.version
        report.seconds = time.perf_counter() - started
        return report


class GuardedAdapter:
    """Routes ``confirm_alias`` through the worker's circuit breaker and call counter."""

    def __init__(self, worker: EnrichmentWorker, report: CycleReport):
        self._worker = worker
        self._report = report

    def confirm_alias(self, new_tokens, old_tokens, new_samples) -> bool:
        result = self._worker._call(
            self._report, self._worker.adapter.confirm_alias, new_tokens, old_tokens, new_samples
        )
        return bool(result)
