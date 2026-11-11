"""Guards for anything a model or a mining step proposes.

LLM output and device-produced log text are both untrusted, so a proposed
template, key list or mask is accepted only if it passes these checks.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from deepparse.masks_types import Mask

from ..parsing.kv import SLOT_RE, compile_extractor, default_keys, extract

# Nested quantifiers such as (a+)+ or (.*)* backtrack catastrophically.
_NESTED_QUANTIFIER_RE = re.compile(r"\([^()]*[+*][^()]*\)[+*{]")
_DIGIT_RE = re.compile(r"\d")
KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

MAX_TEMPLATE_TOKENS = 256
MAX_TOKEN_LENGTH = 256  # Change if needed since I am not sure the value here
MAX_MASK_LENGTH = 256


def generalize_digits(tokens: Sequence[str]) -> tuple[str, ...]:
    """Turn literal tokens containing digits into ``<*>`` (Drain3's "has numbers" rule).

    A cluster mined from one or two lines keeps every literal, so without
    this a fault such as ``code=E165 retrying in 6548ms`` would get a new
    template id per occurrence.
    """
    return tuple(
        "<*>" if not SLOT_RE.search(tok) and _DIGIT_RE.search(tok) else tok for tok in tokens
    )


def template_problem(
    tokens: Sequence[str], samples: Sequence[str], *, min_literal_ratio: float = 0.2
) -> str | None:
    """Return why ``tokens`` is unsafe to publish, or ``None`` if it is fine."""

    # length guardrails
    if not tokens or len(tokens) > MAX_TEMPLATE_TOKENS:
        return f"template has {len(tokens)} tokens"

    # make sure there are at least 1 literal and literal ratio > min amount
    literals = sum(1 for tok in tokens if not SLOT_RE.fullmatch(tok))

    if literals == 0 or literals / len(tokens) < min_literal_ratio:
        return f"over-general: only {literals}/{len(tokens)} literal tokens"

    if any(not isinstance(tok, str) or len(tok) > MAX_TOKEN_LENGTH for tok in tokens):
        return "template contains an invalid or excessively long token"

    try:
        pattern = compile_extractor(tokens)
    except (re.error, ValueError):
        return "could not compile template extractor"

    keys = default_keys(tokens)

    misses = [s for s in samples if extract(pattern, keys, s) is None]

    if misses:
        return f"extractor misses {len(misses)}/{len(samples)} samples, e.g. {misses[0][:80]!r}"

    return None


def clean_keys(proposed: Sequence[str] | None, tokens: Sequence[str]) -> list[str]:
    """Use ``proposed`` keys only if they are well-formed, unique and one per slot."""
    fallback = default_keys(tokens)
    if (
        proposed is None
        or len(proposed) != len(fallback)
        or len(set(proposed)) != len(proposed)
        or not all(KEY_RE.match(k) for k in proposed)
    ):
        return fallback
    return list(proposed)


def safe_masks(masks: Sequence[Mask]) -> list[Mask]:
    """Drop masks that do not compile or that risk catastrophic backtracking."""
    kept: list[Mask] = []

    for mask in masks:
        pattern = mask.pattern

        # Reject empty or excessively long patterns
        if not pattern or len(pattern) > MAX_MASK_LENGTH:
            continue

        # Reject known high-risk regex structures
        if _NESTED_QUANTIFIER_RE.search(pattern):
            continue

        # Reject quantified groups containing alternation,
        # e.g. (a|aa)+, which can cause excessive backtracking
        if re.search(r"\([^)]*\|[^)]*\)\s*[+*{]", pattern):
            continue

        try:
            re.compile(pattern)
        except re.error:
            continue

        kept.append(mask)

    return kept
