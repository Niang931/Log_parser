"""Stream: the live load generator in logpipe/eval/synth_iot.py."""

from __future__ import annotations

import json
import time

import pytest

from logpipe.calibrate import calibrate
from logpipe.cli import main
from logpipe.enrich.llm import OfflineAdapter
from logpipe.eval.synth_iot import Stream, generate
from logpipe.registry import SchemaRegistry


def test_one_second_at_1000_lines_per_second():
    stream = Stream(devices=8, rate=1000, duration_s=1)
    started = time.monotonic()
    events = list(stream)
    elapsed = time.monotonic() - started
    assert abs(len(events) - 1000) <= 50
    assert stream.generated == len(events)
    assert elapsed >= 0.9  # paced by the clock, not generated all at once
    assert len({d.id for d in stream.devices}) == 8
    assert {e.device_id for e in events} <= {d.id for d in stream.devices}


def test_bad_arguments_and_second_pass():
    for kwargs in ({"devices": 0, "rate": 10}, {"devices": 1, "rate": 0}):
        with pytest.raises(ValueError):
            Stream(**kwargs)
    stream = Stream(devices=2, rate=1000, duration_s=0.1)
    list(stream)
    started = time.monotonic()
    list(stream)  # a second pass must not wait for the first pass's schedule
    assert time.monotonic() - started < 0.5
    assert stream.generated == 200


def test_load_command_writes_metrics(snapshot, tmp_path, capsys):
    registry, metrics = tmp_path / "registry.json", tmp_path / "out" / "metrics.jsonl"
    SchemaRegistry(snapshot).save(registry)
    main(["load", "--registry", str(registry), "--devices", "8", "--rate", "500",
          "--duration", "2", "--rollout-at", "0.5", "--store", "memory", "--queue", "memory",
          "--no-enrich", "--metrics-out", str(metrics), "--metrics-every", "0.5"])
    records = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert len(records) >= 3  # about one per 0.5 s, plus the final one
    last = records[-1]
    assert last["generated"] == last["lines"] == 1000
    assert sum(last["status"].values()) == 1000
    assert last["queue"]["kind"] == "memory" and last["rss_mb"] > 0
    assert last["oldest_pending_s"] >= 0 and records[0]["parse_us"]["p99"] > 0
    assert "resolve_lag_p95_s" in last  # None here: --no-enrich resolves nothing
    assert '"lines": 1000' in capsys.readouterr().out


def test_rollout_lines_get_resolved_fast(tmp_path):
    """Task 3.3 rollout goal at small scale: new firmware lines go PENDING, then resolve."""
    history = [e.line for e in generate(devices_per_kind=2, history_hours=2).history]
    registry, metrics = tmp_path / "registry.json", tmp_path / "metrics.jsonl"
    SchemaRegistry(calibrate(history, OfflineAdapter()).snapshot).save(registry)
    main(["load", "--registry", str(registry), "--devices", "8", "--rate", "2000",
          "--duration", "4", "--rollout-at", "1", "--rollout-share", "1", "--store", "memory",
          "--queue", "memory", "--enrich-every", "500", "--llm", "offline",
          "--metrics-out", str(metrics), "--metrics-every", "1"])
    records = [json.loads(line) for line in metrics.read_text().splitlines()]
    lags = [r["resolve_lag_p95_s"] for r in records if r["resolve_lag_p95_s"] is not None]
    assert lags and max(lags) < 60  # target: p95 resolve lag < 60 s
    assert max(r["oldest_pending_s"] for r in records) < 300  # target: nothing PENDING over 5 min
    assert records[-1]["status"].get("PENDING", 0) < 0.05 * records[-1]["lines"]


def test_rollout_switches_firmware():
    rolled = Stream(devices=8, rate=1000, rollout_at_s=0, rollout_share=1, duration_s=0.2)
    lines = [e.line for e in rolled]
    assert all(d.firmware == 2 for d in rolled.devices)
    assert any("fw=2.1.0" in line for line in lines)

    idle = Stream(devices=8, rate=1000, duration_s=0.2)
    list(idle)
    assert all(d.firmware == 1 for d in idle.devices)
