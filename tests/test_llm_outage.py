"""LLM outage: no line may be dead-lettered or get placeholder key names because the LLM was down."""

from datetime import UTC, datetime

from logpipe.enrich.llm import OfflineAdapter
from logpipe.enrich.worker import EnrichmentWorker
from logpipe.parsing.frozen_parser import ParserHandle
from logpipe.pipeline import CYCLE_TAIL, PipelineStats
from logpipe.registry import SchemaRegistry
from logpipe.storage.base import LogRow
from logpipe.storage.memory import MemoryStore
from tests.test_pipeline import NEW


class DownAdapter(OfflineAdapter):
    def name_keys(self, tokens, samples):
        raise ConnectionError("llm down")

    def confirm_alias(self, new_tokens, old_tokens, new_samples):
        raise ConnectionError("llm down")


def _worker(snapshot, adapter):
    registry = SchemaRegistry(snapshot)
    handle = ParserHandle(snapshot)
    worker = EnrichmentWorker(registry, handle, MemoryStore(), adapter, max_attempts=2,
                              consume_timeout=0)
    for line in NEW:
        row = LogRow.from_result(handle.parse(line), device_id="d",
                                 timestamp=datetime.now(UTC), raw_message=line)
        worker.submit(row)
    return worker, registry


def test_llm_down_learns_nothing_and_dead_letters_nothing(snapshot):
    worker, registry = _worker(snapshot, DownAdapter())
    version = registry.current.version
    for _ in range(6):  # more cycles than max_attempts
        assert worker.run_cycle().dead_lettered == 0
    assert registry.current.version == version


def test_rows_resolve_once_llm_is_back(snapshot):
    worker, registry = _worker(snapshot, DownAdapter())
    for _ in range(3):
        worker.run_cycle()
    worker.adapter = OfflineAdapter()
    worker.breaker.wait_duration = 0
    report = None
    for _ in range(3):
        report = worker.run_cycle()
        if report.resolved:
            break
    assert report is not None and report.resolved == len(NEW)


def test_stats_keep_totals_but_bounded_tail():
    from logpipe.enrich.worker import CycleReport

    stats = PipelineStats()
    for _ in range(CYCLE_TAIL + 50):
        stats.add_cycle(CycleReport(resolved=2, dead_lettered=1))
    assert len(stats.cycles) == CYCLE_TAIL
    assert stats.summary()["resolved_by_enrichment"] == 2 * (CYCLE_TAIL + 50)
    assert stats.summary()["dead_lettered"] == CYCLE_TAIL + 50


def test_outage_flag_file_switches_adapter_off(tmp_path, monkeypatch):
    import pytest

    from logpipe.enrich.llm import get_adapter

    flag = tmp_path / "down"
    monkeypatch.setenv("LLM_DOWN_FILE", str(flag))
    adapter = get_adapter("offline")
    assert adapter.name_keys(["a", "<*>"], ["a 1"])
    flag.touch()
    with pytest.raises(ConnectionError):
        adapter.name_keys(["a", "<*>"], ["a 1"])
    flag.unlink()
    assert adapter.name_keys(["a", "<*>"], ["a 1"])
