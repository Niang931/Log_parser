"""Tokenisation helpers used by the Drain engine and metrics layer."""
from __future__ import annotations

import re
from collections.abc import Sequence

from deepparse.utils.regex_library import classify_token

TOKEN_SPLIT = re.compile(r"\s+")


def tokenize(line: str) -> list[str]:
    """Split a log line on whitespace, dropping empty tokens."""
    return [tok for tok in TOKEN_SPLIT.split(line.strip()) if tok]


def mask_tokens(tokens: Sequence[str]) -> list[str]:
    """Replace each token that matches a canonical class with ``<CLASS>``.

    This is a *fallback* tokenisation step kept for backward
    compatibility with earlier callers and tests; the production
    pipeline uses :class:`deepparse.drain.masks_application.MaskApplier`
    on full lines, which handles multi-token spans.
    """
    masked: list[str] = []
    for token in tokens:
        cls = classify_token(token)
        masked.append(f"<{cls}>" if cls else token)
    return masked
