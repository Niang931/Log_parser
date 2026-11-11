"""Drift: a stream's live anomaly-score distribution against the one its model was trained on."""

from __future__ import annotations

from types import SimpleNamespace

from logpipe.server.drift import Drift, histogram, psi
from logpipe.settings import get_settings

MIN_LIVE = get_settings().server.drift_min_live
from tests.test_server import app_lines, calibrated, env, events, send  # noqa: F401 - env is a fixture


def windows(misses, alerted=False):
    return [
        SimpleNamespace(
            score=SimpleNamespace(miss_ratio=m), reasons=("sequence",) if alerted else ()
        )
        for m in misses
    ]


def test_psi_is_zero_for_the_same_distribution_and_large_for_a_shift():
    base = histogram([0.05] * 90 + [0.15] * 10)
    assert psi(base, base) == 0
    assert psi(base, histogram([0.05] * 85 + [0.15] * 15)) < 0.1
    assert psi(base, histogram([0.6] * 50 + [0.05] * 50)) > 0.25


def test_report_needs_a_baseline_and_enough_live_windows():
    d = Drift()
    stream = {"id": "s", "name": "S", "miss_baseline": histogram([0.05] * 100)}
    d.observe("s", windows([0.05] * (MIN_LIVE - 1)))
    row = d.report([stream])[0]
    assert row["psi"] is None and row["live_windows"] == MIN_LIVE - 1 and row["alert_pct"] == 0
    d.observe("s", windows([0.9] * 100, alerted=True))
    row = d.report([stream])[0]
    assert row["psi"] > 0.25 and row["alert_pct"] > 30
    # trained before baselines
    assert d.report([{"id": "s", "name": "old model"}])[0]["psi"] is None
    d.reset("s")
    assert d.report([stream])[0]["live_windows"] == 0


def test_server_tracks_drift_per_stream(env):  # noqa: F811
    server, client, headers = env
    calibrated(server, client, headers, app_lines(300))
    stream = server.control.get("stream", "api")
    assert abs(sum(stream["miss_baseline"]) - 1) < 1e-3
    send(server, client, headers, events(app_lines(1200, seed=9), start=5000))
    row = client.get("/api/drift").json()["streams"][0]
    assert row["id"] == "api" and row["live_windows"] > 0 and row["has_baseline"]
    values = server.sample_metrics()
    assert "alert_window_pct" in values
    server.sample_metrics()
    assert server.sample_metrics()["new_templates_per_hour"] >= 0
