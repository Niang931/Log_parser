"""Property-based tests for entropy-greedy sampling (paper Algorithm 1).

These checks use Hypothesis to verify *invariants* across a large
sweep of randomly-generated log corpora rather than spot examples.
The goal is to catch regressions where the sampler returns the wrong
shape or violates the diversity guarantees the paper depends on.
"""
from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from deepparse.utils.sampling import deterministic_sample, entropy_greedy_sample

# A log line: any printable string between 1 and 80 chars, ASCII-ish so the
# token regex inside `entropy_greedy_sample` is exercised the same way real
# log corpora exercise it.
_log_line = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters=" .-_/:",
    ),
    min_size=1,
    max_size=80,
).map(str.strip).filter(lambda s: len(s) > 0)


@given(logs=st.lists(_log_line, min_size=1, max_size=200), k=st.integers(1, 200))
@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
def test_sample_size_never_exceeds_k(logs, k):
    sample = entropy_greedy_sample(logs, k)
    assert len(sample) <= k
    assert len(sample) <= len(logs)


@given(logs=st.lists(_log_line, min_size=1, max_size=200), k=st.integers(1, 200))
@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
def test_sample_indices_are_unique_and_valid(logs, k):
    sample = entropy_greedy_sample(logs, k)
    assert len(set(sample)) == len(sample)  # no duplicate indices
    assert all(0 <= i < len(logs) for i in sample)


@given(logs=st.lists(_log_line, min_size=1, max_size=100), k=st.integers(1, 100))
@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
def test_sample_is_pure_function_of_inputs(logs, k):
    """Determinism: identical inputs -> identical outputs."""
    a = entropy_greedy_sample(logs, k)
    b = entropy_greedy_sample(logs, k)
    assert a == b


@given(logs=st.lists(_log_line, min_size=1, max_size=50))
@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
def test_returns_full_corpus_when_k_meets_or_exceeds(logs):
    """When k >= |L|, the sampler returns the entire corpus (paper edge case)."""
    sample = deterministic_sample(logs, len(logs))
    assert sample == list(logs)
    sample_more = deterministic_sample(logs, len(logs) + 100)
    assert sample_more == list(logs)
