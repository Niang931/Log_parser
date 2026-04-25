"""Determinism + coverage checks for the offline synthesis stub."""
from __future__ import annotations

from deepparse.synth.r1_deepseek_stub import synthesize_offline


def test_offline_stub_returns_required_masks():
    logs = [
        "2024-01-01 00:00:00 INFO worker Completed job 1",
        "2024-01-01 00:00:01 WARN worker Completed job 2",
    ]
    masks = synthesize_offline(logs)
    labels = {mask.label for mask in masks}
    assert {"TIMESTAMP", "IPV4", "NUMBER", "LOGLEVEL"}.issubset(labels)


def test_offline_stub_is_deterministic():
    logs = [
        "2024-01-01 00:00:00 INFO worker Completed job 1",
        "2024-01-01 00:00:01 WARN worker Completed job 2",
        "2024-01-01 00:00:02 INFO worker /var/log/file.log rotated",
    ]
    a = synthesize_offline(logs)
    b = synthesize_offline(logs)
    assert [(m.label, m.pattern) for m in a] == [(m.label, m.pattern) for m in b]


def test_offline_stub_picks_up_optional_classes():
    logs = [
        "fetch 0xDEADBEEF",
        "uuid 550e8400-e29b-41d4-a716-446655440000 ok",
        "path /var/log/sys.log",
        "mail alice@example.com",
        "url https://example.com/page",
    ]
    labels = {m.label for m in synthesize_offline(logs)}
    assert {"HEX", "UUID", "PATH", "EMAIL", "URL"}.issubset(labels)
