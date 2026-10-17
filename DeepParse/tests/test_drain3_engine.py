"""Drain3Engine: the real drain3 library behind DeepParse's mask-first preprocessing."""
from __future__ import annotations

import pytest

pytest.importorskip("drain3")

from deepparse.drain.drain3_engine import Drain3Engine
from deepparse.drain.drain_engine import DrainEngine
from deepparse.utils.regex_library import canonical_masks

LINES = [
    "Failed password for root from 5.36.59.76 port 42393 ssh2",
    "Failed password for uucp from 195.154.37.122 port 59266 ssh2",
    "Failed password for admin from 10.0.0.7 port 1024 ssh2",
    "Received disconnect from 52.80.34.196: 11: Bye Bye [preauth]",
    "Received disconnect from 10.1.1.1: 3: Bye Bye [preauth]",
]


def test_drain3_groups_on_prefix_like_original_drain():
    engine = Drain3Engine(masks=canonical_masks())
    pairs = engine.parse_with_ids(LINES)
    ids = [cid for cid, _ in pairs]
    # Lines whose 4th token varies (root / uucp / admin) still share one cluster:
    # Drain3 keys the tree on depth-2 = 3 prefix tokens, not on the literal 4th token.
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] == ids[4] and ids[3] != ids[0]
    assert pairs[0][1] == "Failed password for <*> from <*> port <*> ssh2"
    assert pairs[3][1] == "Received disconnect from <*>: <*>: Bye Bye [preauth]"


def test_drain3_engine_matches_port_surface():
    masks = canonical_masks()
    d3, port = Drain3Engine(masks=masks), DrainEngine(masks=masks)
    assert d3.parse(LINES[:1]) == port.parse(LINES[:1])
    assert len(d3.parse_with_ids(LINES)) == len(LINES)
    assert d3.num_clusters == len(d3.clusters) == 2
    assert all(c.size >= 1 and c.template for c in d3.clusters)


def test_drain3_is_deterministic():
    a = Drain3Engine(masks=canonical_masks()).parse_with_ids(LINES)
    b = Drain3Engine(masks=canonical_masks()).parse_with_ids(LINES)
    assert a == b
