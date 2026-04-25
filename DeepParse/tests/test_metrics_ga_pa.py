"""Tests for the GA / PA metric implementations."""
from __future__ import annotations

import pytest

from deepparse.metrics import grouping_accuracy, parsing_accuracy


def test_grouping_accuracy_perfect_with_renumbered_ids():
    # Identical partition, different cluster-id labels — paper-defined GA
    # is invariant to renumbering.
    true = ["a", "a", "b", "b", "c"]
    pred = [10, 10, 20, 20, 30]
    assert grouping_accuracy(true, pred) == 1.0


def test_grouping_accuracy_partial_partition_match():
    # Two ground-truth clusters: {0,1,2} and {3,4}.
    # Predicted: {0,1} and {2,3,4} — neither matches a ground-truth set
    # exactly, so 0/5 logs are correctly grouped.
    true = ["a", "a", "a", "b", "b"]
    pred = ["x", "x", "y", "y", "y"]
    assert grouping_accuracy(true, pred) == 0.0


def test_grouping_accuracy_partial_credit():
    # First three logs share gt-cluster A and pred-cluster X (set match).
    # Last two share gt-cluster B and pred-cluster Y (set match).
    true = ["a", "a", "a", "b", "b"]
    pred = ["x", "x", "x", "y", "y"]
    assert grouping_accuracy(true, pred) == 1.0


def test_grouping_accuracy_split_cluster_zero():
    # Ground-truth = single cluster of 4 logs.  Predicted = 2+2 split.
    # Neither predicted cluster matches the gt cluster set → 0/4.
    true = ["a", "a", "a", "a"]
    pred = ["x", "x", "y", "y"]
    assert grouping_accuracy(true, pred) == 0.0


def test_parsing_accuracy_exact():
    true = ["foo", "bar"]
    pred = ["foo", "baz"]
    assert parsing_accuracy(true, pred) == 0.5


def test_parsing_accuracy_empty_raises_for_mismatch():
    with pytest.raises(ValueError):
        parsing_accuracy(["a"], ["a", "b"])
