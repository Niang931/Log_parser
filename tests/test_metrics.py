"""Built-in metrics: the hot path's counters become rates, percentiles and process load; the
System API serves them as series."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from logpipe.server.app import create_app
from logpipe.server.metrics import CountedAdapter, MemorySeries, Metrics
from tests.test_server import app_lines, env, events, pump, send  # noqa: F401 - env is a fixture


def test_sample_turns_counters_into_rates_and_percentiles():
    m = Metrics(MemorySeries())
    for i in range(100):
        m.parsed((i + 1) * 1000, unknown=i % 4 == 0, alerts=1 if i == 0 else 0)
    m.count("errors", 2)
    v = m.sample({"inbox": 7, "queue_lag": None})
    assert v["unknown_pct"] == 25.0 and v["inbox"] == 7 and "queue_lag" not in v
    assert 49 <= v["parse_p50_us"] <= 52 and v["parse_p99_us"] > 98
    assert v["lines_per_s"] > 0 and v["errors_per_s"] > 0 and v["alerts_per_min"] > 0
    assert v["rss_mb"] > 10 and v["threads"] >= 1
    again = m.sample()
    assert again["lines_per_s"] == 0 and "parse_p50_us" not in again and "unknown_pct" not in again


def test_latency_reservoir_stays_bounded():
    m = Metrics(MemorySeries())
    for _ in range(50_000):
        m.parsed(1000, unknown=False)
    assert len(m._lat) == 20_000
    assert m.sample()["parse_p99_us"] == 1.0


def test_memory_series_buckets_and_windows():
    s = MemorySeries()
    now = time.time()
    s.write(now - 7200, {"x": 100})  # outside the window
    s.write(now - 30, {"x": 1})
    s.write(now - 20, {"x": 3})
    out = s.query(minutes=60, step=3600)
    # one bucket, unless they straddle an hour
    assert [v for _, v in out["x"]] in ([2.0], [1.0, 3.0])


def test_counted_adapter_times_calls_and_forwards_attributes():
    class Inner:
        key_vocab: tuple = ()

        def name_keys(self, tokens, samples):
            return ["a"]

        def confirm_alias(self, *a):
            raise RuntimeError("429")

    m, inner = Metrics(MemorySeries()), Inner()
    a = CountedAdapter(inner, m)
    a.key_vocab = ["host"]
    assert inner.key_vocab == ["host"] and a.name_keys([], []) == ["a"]
    try:
        a.confirm_alias()
    except RuntimeError:
        pass
    v = m.sample()
    assert v["llm_calls_per_min"] > 0 and v["llm_errors_per_min"] > 0 and "llm_latency_s" in v


def test_system_api_reports_the_servers_own_load(env):  # noqa: F811
    server, client, headers = env
    send(server, client, headers, events(app_lines(200)))
    server.sample_metrics()
    body = TestClient(create_app(server)).get("/api/system?minutes=15").json()
    assert body["current"]["pending_rows"] == 200 and body["current"]["inbox"] == 0
    assert body["series"]["lines_per_s"][-1][1] > 0
    assert body["metrics"]["cpu_pct"]["group"] == "Process" and body["step"] >= 10


def test_json_log_format(tmp_path, monkeypatch):
    import json
    import logging

    from logpipe.server import setup_logging

    monkeypatch.setenv("LOGPIPE_LOG_FORMAT", "json")
    log = setup_logging(tmp_path)
    logging.getLogger("t").info("hello %s", "there")
    for h in logging.getLogger().handlers:
        h.flush()
    line = json.loads(log.read_text().splitlines()[-1])
    assert line["message"] == "hello there" and line["level"] == "info"
    logging.basicConfig(force=True)
