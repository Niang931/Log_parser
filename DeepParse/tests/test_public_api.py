"""Tests for the lightweight synth/Drain public API."""
from __future__ import annotations

from deepparse import Drain, synth_masks


def test_public_api_roundtrip() -> None:
    logs = [
        "2024-01-15 10:30:45 INFO Auth user john_doe logged in from 192.168.1.10",
        "2024-01-15 10:30:46 WARN Memory usage at 85 percent for host 10.0.0.5",
        "2024-01-15 10:30:47 ERROR Database timeout after 45 seconds",
    ]

    patterns = synth_masks(logs, sample_size=2)
    assert isinstance(patterns, list)
    assert patterns, "synth_masks should return at least one mask"
    assert {entry["label"] for entry in patterns} >= {"TIMESTAMP", "IPV4", "NUMBER", "LOGLEVEL"}

    drain = Drain()
    drain.load_masks(patterns)
    parsed = drain.parse_all(logs)
    assert len(parsed) == len(logs)
    # Final templates use the LogHub-canonical <*> wildcard so they can be
    # compared directly against LogHub-2k ground truth.
    assert all("<*>" in template for template in parsed)


def test_listing1_workflow_from_paper() -> None:
    """Paper Listing 1: ``patterns = synth_masks(sys_logs,
    sample_size=50, temperature=0, max_length=512)`` then
    ``Drain().load_masks(patterns).parse_all(sys_logs)``.

    This test exercises the full workflow with the exact keyword
    arguments named in the paper to guarantee they remain stable.
    """
    sys_logs = [
        "2024-01-15 10:30:45 INFO User john_doe logged in from IP 192.168.1.100",
        "2024-01-15 10:30:46 INFO User alice logged in from IP 10.0.0.5",
        "2024-01-15 10:30:47 ERROR Database timeout after 45 seconds",
    ]
    patterns = synth_masks(sys_logs, sample_size=50, temperature=0, max_length=512)
    drain = Drain()
    drain.load_masks(patterns)
    parsed = drain.parse_all(sys_logs)
    assert len(parsed) == len(sys_logs)


def test_drain_is_deterministic() -> None:
    logs = [
        "2024-01-15 10:30:45 INFO server-1 started on port 8080",
        "2024-01-15 10:30:46 INFO server-2 started on port 8081",
        "2024-01-15 10:30:47 INFO server-3 started on port 8082",
    ]
    patterns = synth_masks(logs, sample_size=3)

    first = Drain()
    first.load_masks(patterns)
    second = Drain()
    second.load_masks(patterns)

    assert first.parse_all(logs) == second.parse_all(logs)
