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
    detector = StreamingDetector(
        FakeScorer(),
        DetectorConfig(window=5, stride=5, min_len=2, lateness=timedelta(0)),
        resolver=lambda line: handle.parse(line).template_id,
    )
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


def test_pipeline_reports_stage_timings_and_line_counts(snapshot):
    from collections import Counter

    from logpipe.observability import Datadog

    class Rec:
        def __init__(self):
            self.calls = Counter()

        def distribution(self, name, value, tags):
            self.calls[(name, tuple(tags))] += 1

        def increment(self, name, n=1, tags=()):
            self.calls[(name, tuple(tags))] += n

    rec = Rec()
    dd = Datadog()
    dd._statsd = rec
    Pipeline(ParserHandle(snapshot), MemoryStore(), datadog=dd).run(_events())
    assert rec.calls[("logpipe.stage.duration", ("stage:parse",))] == 17 + len(NEW)
    assert rec.calls[("logpipe.stage.duration", ("stage:store",))] == 17 + len(NEW)
    assert rec.calls[("logpipe.lines", ("status:final",))] == 17
    assert rec.calls[("logpipe.lines", ("status:pending",))] == len(NEW)


def test_pipeline_writes_json_lines_for_the_agent(snapshot, tmp_path, monkeypatch):
    import json

    from logpipe.observability import Datadog
    from logpipe.settings import DatadogSettings

    monkeypatch.setattr("ddtrace.llmobs.LLMObs.enable", lambda **_: None)
    dd = Datadog(DatadogSettings(log_file=tmp_path / "dd.jsonl"))
    assert dd.logging_on()
    Pipeline(ParserHandle(snapshot), MemoryStore(), datadog=dd).run(_events())
    rows = [json.loads(x) for x in (tmp_path / "dd.jsonl").read_text().splitlines()]
    assert len(rows) == 17 + len(NEW)
    assert all("message" in r and "template_id" in r for r in rows)
    assert sum(r["template"] is not None for r in rows) >= 17
