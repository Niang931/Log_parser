from datetime import timedelta

from logpipe.anomaly.detector import DetectorConfig, StreamingDetector
from logpipe.enrich.llm import OfflineAdapter
from logpipe.enrich.worker import EnrichmentWorker
from logpipe.parsing.frozen_parser import ParserHandle
from logpipe.pipeline import Pipeline
from logpipe.registry import SchemaRegistry
from logpipe.sources.file_replay import LogEvent, replay
from logpipe.storage.memory import MemoryStore
from tests.conftest import FIXTURES
from tests.test_detector import FakeScorer

NEW = [
    f"2026-04-13 10:30:{i:02d}.000 [ERROR] OTA update failed code=0x{i:02X} retry={i}"
    for i in range(1, 6)
]


def _events():
    events = list(replay(FIXTURES))
    events += [LogEvent("thermostat-TH-45678", "new", i, line) for i, line in enumerate(NEW)]
    return events


def test_pipeline_closes_the_loop(snapshot):
    registry = SchemaRegistry(snapshot)
    handle = ParserHandle(snapshot)
    store = MemoryStore()
    worker = EnrichmentWorker(registry, handle, store, OfflineAdapter())
    detector = StreamingDetector(FakeScorer(), DetectorConfig(
        window=5, stride=5, min_len=2, lateness=timedelta(0)), resolver=lambda l: handle.parse(l).template_id)
    pipeline = Pipeline(handle, store, worker=worker, detector=detector, enrich_every=1_000)
    stats = pipeline.run(_events())

    summary = stats.summary()
    assert summary["lines"] == 17 + len(NEW)
    assert summary["pending_at_ingest"] == len(NEW)
    assert summary["resolved_by_enrichment"] == len(NEW)
    assert summary["templates_learned"] == 1
    assert store.count_by_status() == {"FINAL": 17 + len(NEW)}
    assert set(summary["parse_latency_us"]) == {"p50", "p90", "p99", "max"}
    assert detector.windows_scored > 0
    # The OTA lines are new to the model, so their windows raise novelty alerts.
    assert any("novelty" in a.reasons for a in stats.alerts)
    assert all(a in store.alerts for a in stats.alerts)


def test_pipeline_without_enrichment_leaves_rows_pending(snapshot):
    store = MemoryStore()
    stats = Pipeline(ParserHandle(snapshot), store).run(_events())
    assert stats.summary()["resolved_by_enrichment"] == 0
    assert store.count_by_status() == {"FINAL": 17, "PENDING": len(NEW)}
