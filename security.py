"""
pipeline/security.py — Prompt-injection sanitisation
"""
from __future__ import annotations

import logging
import re

from pipeline.config import INJECTION_BLOCK_PATTERNS, INJECTION_MAX_LEN
from pipeline.telemetry import TELEMETRY

log = logging.getLogger("deepparse.security")

_INJECTION_RX = re.compile(
    "|".join(INJECTION_BLOCK_PATTERNS), re.IGNORECASE
)


def sanitize_for_llm(line: str, max_len: int = INJECTION_MAX_LEN) -> str | None:
    """
    Return sanitised line or None if prompt-injection pattern detected.
    Strips control characters and truncates to max_len.
    """
    truncated = line[:max_len]
    if _INJECTION_RX.search(truncated):
        log.warning("Injection pattern detected — line dropped: %.60s…", truncated)
        TELEMETRY.record("injection_blocked", snippet=truncated[:80])
        return None
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", truncated)