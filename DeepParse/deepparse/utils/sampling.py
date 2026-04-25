"""Sampling utilities for selecting a small, diverse log subset.

Two algorithms are exposed:

* :func:`entropy_greedy_sample` — implements Algorithm 1 from the paper
  ("Entropy-Greedy Sampling").  Logs are normalised (digits and hex
  tokens replaced with ``#`` placeholders), Shannon entropy is computed
  over the per-line token-frequency distribution, and the highest
  entropy candidates are selected greedily, rejecting any that have a
  Jaccard token-set similarity ≥ 0.8 with already-selected lines.
* :func:`deterministic_sample` — a stable wrapper that returns the
  *original* (non-normalised) log lines selected by the entropy
  algorithm.  This is the function consumed by the public API and CLI:
  the original lines are required because mask synthesis must see the
  un-normalised structural signals (paper, Section "Entropy-Greedy
  Sampling").
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence, Tuple

# Token splitter: whitespace + punctuation that shouldn't pollute counts.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:%@#-]+")
_DIGIT_RE = re.compile(r"\d+")
_HEX_RE = re.compile(r"\b(?:0x)?[0-9A-Fa-f]{4,}\b")
_JACCARD_THRESHOLD = 0.8


def _normalise(line: str) -> str:
    """Replace digits and hex literals with ``#`` so they don't dominate entropy."""
    line = _HEX_RE.sub("#H#", line)
    line = _DIGIT_RE.sub("#N#", line)
    return line


def _token_set(line: str) -> set[str]:
    return set(_TOKEN_RE.findall(line))


def _shannon_entropy(line: str) -> float:
    tokens = _TOKEN_RE.findall(line)
    if not tokens:
        return 0.0
    counts = Counter(tokens)

    # Just plain old entropy formula
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def entropy_greedy_sample(logs: Sequence[str], k: int) -> List[int]:
    """Return ``k`` indices selected by the entropy-greedy algorithm.

    Returns indices into the original ``logs`` sequence so callers can
    decide whether they want the normalised or original line.
    """
    if k <= 0 or not logs:
        return []
    if k >= len(logs):
        return list(range(len(logs)))

    # Normalize the log lines first
    normalised = [_normalise(line) for line in logs]
    entropies: List[Tuple[float, int]] = []

    for idx, norm in enumerate(normalised):
        entropies.append((_shannon_entropy(norm), idx))

    # Sort by (-entropy, idx) so ties break deterministically by index., which mean smaller index get chosen for tie break
    entropies.sort(key=lambda pair: (-pair[0], pair[1]))

    selected: List[int] = []
    selected_token_sets: List[set[str]] = []

    for _entropy, idx in entropies:
        candidate = _token_set(normalised[idx])

        if all(_jaccard(candidate, ts) < _JACCARD_THRESHOLD for ts in selected_token_sets):
            # If all canddicate jaccard value pass then append to candidate index list
            selected.append(idx)

            # Appeend the value itself for next iteration comparison
            selected_token_sets.append(candidate)

            # Break the moment the number of samples has been reached
            if len(selected) >= k:
                break

    # Paper Algorithm 1 returns S where |S| ≤ k.  If the corpus is so
    # homogeneous that the Jaccard < 0.8 constraint cannot be satisfied
    # k times, we deliberately return fewer than k items rather than
    # back-fill with near-duplicates — back-filling would silently
    # violate the diversity guarantee in Section "Entropy-Greedy
    # Sampling" and pollute the LLM training prompt with redundant logs.
    return selected


def deterministic_sample(logs: Sequence[str], k: int) -> List[str]:
    """Return ``k`` *original* log lines selected by the entropy algorithm."""
    # In case sample size is bigger than number of logs then just return everything
    if k >= len(logs):
        return list(logs)
    indices = entropy_greedy_sample(logs, k)
    return [logs[i] for i in indices]


def deterministic_indices(logs: Sequence[str], k: int) -> List[int]:
    """Return the indices selected by :func:`deterministic_sample`."""
    return entropy_greedy_sample(logs, k)
