"""Tests for the Drain engine and the mask-first preprocessing stage."""
from __future__ import annotations

from deepparse.drain.drain_engine import DrainEngine
from deepparse.drain.masks_application import MaskApplier
from deepparse.masks_types import Mask


def test_drain_applies_masks_before_parsing():
    masks = [Mask(label="NUMBER", pattern=r"\d+", justification="numbers")]
    engine = DrainEngine(masks=masks)
    logs = ["value 123", "value 456", "value 789"]
    templates = engine.parse(logs)
    # Final templates render typed placeholders as <*> for LogHub parity.
    assert templates[0] == templates[1] == templates[2] == "value <*>"


def test_drain_emits_wildcards_for_unmasked_variability():
    # Same prefix tokens within the configured depth, varying later token
    # → Drain clusters them and emits <*> for the variable position.
    engine = DrainEngine(masks=[], depth=2, similarity_threshold=0.4)
    logs = [
        "server started on alpha",
        "server started on beta",
        "server started on gamma",
    ]
    templates = engine.parse(logs)
    assert templates[0] == templates[1] == templates[2]
    assert "<*>" in templates[0]


def test_mask_applier_uses_typed_placeholders():
    masks = [
        Mask(label="IPV4", pattern=r"(?:\d{1,3}\.){3}\d{1,3}", justification="ip"),
        Mask(label="LOGLEVEL", pattern=r"\b(?:INFO|WARN|ERROR)\b", justification="lvl"),
    ]
    applier = MaskApplier(masks)
    out = applier.apply("INFO request from 192.168.0.1")
    assert "<VAR:IPV4>" in out
    assert "<VAR:LOGLEVEL>" in out


def test_drain_handles_duplicate_named_groups():
    # Multiple masks reusing the same group name must not raise.
    masks = [
        Mask(label="A", pattern=r"(?P<x>\d+)", justification=""),
        Mask(label="B", pattern=r"(?P<x>[a-z]+)", justification=""),
    ]
    engine = DrainEngine(masks=masks)
    out = engine.parse(["foo 42", "bar 99"])
    # Final templates use <*> wildcards (typed placeholders are internal).
    assert all("<*>" in line for line in out)


def test_drain_assigns_stable_cluster_ids():
    engine = DrainEngine(masks=[])
    pairs1 = engine.parse_with_ids(["a b c", "a b d", "a b c"])
    # Identical inputs -> identical templates AND identical cluster ids.
    assert pairs1[0][0] == pairs1[2][0]
    assert pairs1[0][1] == pairs1[2][1]


def test_identical_lines_get_identical_template_ids():
    """Paper claim (Section 'Integration with Drain'): identical log
    lines always receive the same template id.
    """
    masks = [Mask(label="NUMBER", pattern=r"\d+", justification="")]
    engine = DrainEngine(masks=masks)
    logs = [
        "user 42 logged in",
        "user 99 logged in",
        "user 42 logged in",            # duplicate of logs[0]
        "different event happened here",
        "user 42 logged in",            # another duplicate of logs[0]
    ]
    pairs = engine.parse_with_ids(logs)
    ids = [cid for cid, _ in pairs]
    assert ids[0] == ids[2] == ids[4]
    assert ids[0] == ids[1]  # same template after masking
    assert ids[3] != ids[0]


def test_drain_default_config_matches_paper():
    """Paper, Section 'Baseline Parsers': Drain3 defaults are
    depth=5, similarity=0.4, max_children=100.
    """
    engine = DrainEngine()
    assert engine.depth == 5
    assert engine.similarity_threshold == 0.4
    assert engine.max_children == 100
