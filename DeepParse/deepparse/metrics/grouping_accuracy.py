"""Grouping Accuracy metric (paper, Section 'Background', Eq. for GA).

Standard LogHub / Drain definition: a log line is *correctly grouped*
iff the set of line indices in its predicted cluster is exactly equal
to the set of line indices in some ground-truth cluster.  GA is then::

    GA = (number of correctly grouped logs) / (total number of logs)

This is invariant to cluster-id renumbering: only the *partition* of
log indices matters, not the labels assigned to each part.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Sequence


def _partition(ids: Sequence[Hashable]) -> dict[Hashable, frozenset[int]]:
    """Group line indices by their cluster id; return ``{cluster_id: frozenset(indices)}``."""
    out: dict[Hashable, set[int]] = defaultdict(set)
    for idx, cluster_id in enumerate(ids):
        out[cluster_id].add(idx)
    return {cid: frozenset(indices) for cid, indices in out.items()}


def grouping_accuracy(
    true_group_ids: Sequence[Hashable],
    predicted_group_ids: Sequence[Hashable],
) -> float:
    if len(true_group_ids) != len(predicted_group_ids):
        raise ValueError("Mismatched lengths for GA computation")
    n = len(true_group_ids)
    if n == 0:
        return 0.0

    truth = _partition(true_group_ids)
    pred = _partition(predicted_group_ids)
    truth_sets = set(truth.values())

    correct = 0
    for indices in pred.values():
        if indices in truth_sets:
            correct += len(indices)
    return correct / n
