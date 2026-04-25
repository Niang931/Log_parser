"""Tests for entropy-greedy sampling (paper, Algorithm 1)."""
from __future__ import annotations

from deepparse.utils.sampling import (
    deterministic_sample,
    entropy_greedy_sample,
)


def test_returns_all_when_k_exceeds_corpus():
    logs = ["a", "b", "c"]
    assert deterministic_sample(logs, 10) == logs


def test_is_deterministic():
    logs = [f"line {i} value {i*7}" for i in range(50)]
    a = deterministic_sample(logs, 10)
    b = deterministic_sample(logs, 10)
    assert a == b


def test_rejects_near_duplicates_via_jaccard():
    # Five near-identical lines and one structurally distinct one.
    base = "user alice logged in from 10.0.0.1"
    logs = [
        base,
        base,
        base,
        base,
        base,
        "DEBUG memory pointer 0xCAFEBABE freed at /tmp/heap",
    ]
    sample = deterministic_sample(logs, 2)
    # Both classes should be represented: at least one base + the distinct line.
    assert any("memory pointer" in line for line in sample)
    assert any("logged in" in line for line in sample)


def test_picks_higher_entropy_first():
    logs = [
        "a a a a a",            # entropy 0
        "alpha beta gamma delta epsilon",  # high entropy
    ]
    indices = entropy_greedy_sample(logs, 1)
    assert indices == [1]


def test_normalisation_does_not_dominate_entropy():
    """Paper, Algorithm 1, Line 1: digits and hex tokens are normalised
    so numeric variation doesn't inflate entropy.

    The two lines below differ only in numeric/hex content and so should
    look the same to the entropy scorer (entropy ≈ identical), but the
    sampler should still pick one of them when k=1.
    """
    logs = [
        "request id 1234 took 56 ms hex 0xDEAD",
        "request id 9999 took 12 ms hex 0xBEEF",
    ]
    indices = entropy_greedy_sample(logs, 1)
    assert len(indices) == 1
    # Either index is acceptable — what matters is that we return one,
    # not an empty list (which would happen if Jaccard rejected both).
    assert indices[0] in (0, 1)


def test_returns_original_lines_not_normalised():
    """Paper, Section 'Entropy-Greedy Sampling': normalisation affects
    sampling selection only — the original (un-normalised) logs are
    passed to the LLM for mask synthesis.
    """
    logs = ["timestamp 0xCAFE event 42", "alpha beta gamma delta"]
    sample = deterministic_sample(logs, 2)
    # No "#H#" / "#N#" placeholders should appear in the returned lines.
    for line in sample:
        assert "#H#" not in line
        assert "#N#" not in line
    assert set(sample) == set(logs)
