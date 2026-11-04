"""Soak helpers: the shadow Drain is freed after a cycle, soak_report.py judges RSS growth."""

import gc
import json
import subprocess
import sys
import weakref
from pathlib import Path

from logpipe.enrich import worker as worker_module
from logpipe.enrich.llm import OfflineAdapter
from tests.test_llm_outage import _worker

REPORT = Path(__file__).parent.parent / "scripts" / "soak_report.py"


def test_shadow_drain_is_freed_after_cycle(snapshot, monkeypatch):
    refs = []
    real = worker_module.make_engine

    def tracking(*args, **kwargs):
        engine = real(*args, **kwargs)
        refs.append(weakref.ref(engine))
        return engine

    monkeypatch.setattr(worker_module, "make_engine", tracking)
    worker, _ = _worker(snapshot, OfflineAdapter())
    worker.run_cycle()
    gc.collect()
    assert refs and all(r() is None for r in refs)


def _metrics(tmp_path, rss_of):
    path = tmp_path / "metrics.jsonl"
    rows = [
        {"t": i * 10, "generated": i * 1000, "lines_per_s": 100.0, "parse_us": {"p99": 300},
         "rss_mb": rss_of(i), "oldest_pending_s": 0.0, "status": {"FINAL": i * 1000}}
        for i in range(1, 101)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def _run(path):
    return subprocess.run([sys.executable, "-I", str(REPORT), str(path)],
                          capture_output=True, text=True)


def test_soak_report_passes_flat_rss(tmp_path):
    done = _run(_metrics(tmp_path, lambda i: 200.0 + (i % 3)))
    assert done.returncode == 0 and json.loads(done.stdout)["pass"] is True


def test_soak_report_fails_growing_rss(tmp_path):
    done = _run(_metrics(tmp_path, lambda i: 200.0 + i))
    assert done.returncode == 1 and json.loads(done.stdout)["pass"] is False
