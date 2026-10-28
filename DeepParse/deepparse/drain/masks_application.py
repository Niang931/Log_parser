"""Deterministic application of mask regexes.

Implements the *Mask-First* strategy described in the DeepParse paper
(Section "Integration with Drain"): incoming lines are matched against the
synthesised regex bundle and matched spans are substituted with
typed placeholders such as ``<VAR:IP>`` or ``<VAR:TIMESTAMP>`` before
Drain clusters them.  Using typed placeholders keeps the parse tree
shallow, deterministic, and stable while preserving structural intent.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from deepparse.masks_types import Mask


def _placeholder_for(label: str) -> str:
    """Return the typed placeholder (remove special characters) string for a mask label."""
    safe = re.sub(r"[^A-Z0-9]+", "", label.upper()) or "VAR"
    return f"<VAR:{safe}>"


def _strip_named_groups(pattern: str) -> str:
    """Drop ``(?P<name>...)`` group names so multiple masks can share a name."""
    return re.sub(r"\(\?P<[^>]+>", "(", pattern)


# ---- Mask safety and normalisation -----------------------------------------

# Patterns that are too broad: they match static key names or arbitrary words,
# collapsing all structure and producing wildcard-only templates.
_FORBIDDEN_PATTERNS: frozenset[str] = frozenset({
    r"[^\s]+",
    r"[\w.\-:/]+",
    r".*",
    r".+",
    # snake_case identifiers — matches static key names like cell_id,
    # power_source, mcc_mnc which must remain as static template text.
    r"\b[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+\b",
})

# Bare number patterns that lack word boundaries.  Without a lookbehind they
# greedily match digits embedded inside identifiers (``pm25``, ``C2D``,
# ``eth0``).  A lookbehind is prepended so they only fire when the digit
# sequence is preceded by a non-word, non-dot character (e.g. ``=``, space).
_BARE_NUMBER_PATTERNS: frozenset[str] = frozenset({
    r"-?\d+(?:\.\d+)?",
    r"\d+(?:\.\d+)?",
    r"-?\d+",
    r"\d+",
})
_NUMBER_LOOKBEHIND = r"(?<![\w.])"


def _normalise_pattern(pattern: str) -> str | None:
    """Return a safe variant of *pattern*, or ``None`` to drop it."""
    if pattern in _FORBIDDEN_PATTERNS:
        return None
    if pattern in _BARE_NUMBER_PATTERNS:
        return _NUMBER_LOOKBEHIND + pattern
    return pattern


# ---- Priority for span-based conflict resolution ---------------------------

# Higher value = more specific = wins when two masks match overlapping spans.
# For example IPV4 (``192.168.1.1``) must beat NUMBER (``192``) on the same
# region; HEX (``0xCC06``) must beat NUMBER (``0``).
_LABEL_PRIORITY: dict[str, int] = {
    "TIMESTAMP": 100,
    "IPV4": 90,
    "IP": 90,
    "UUID": 85,
    "MAC": 80,
    "HEX": 75,
    "PATH": 70,
    "LOGLEVEL": 60,
    "BOOLEAN": 50,
    "NUMBER": 10,
}
_DEFAULT_PRIORITY = 40


def _mask_priority(label: str) -> int:
    """Return priority for a mask label.  Higher = more specific."""
    upper = label.upper()
    for key, prio in _LABEL_PRIORITY.items():
        if key in upper:
            return prio
    return _DEFAULT_PRIORITY


# ---- MaskApplier -----------------------------------------------------------

class MaskApplier:
    """Apply regex masks to log lines before Drain clustering.

    Each mask substitutes its match span with a *typed placeholder*
    derived from the mask label, e.g. label ``"IPV4"`` → ``<VAR:IPV4>``.

    Masks are matched against the **original** line and non-overlapping
    spans are selected by priority (most-specific pattern wins).  This
    prevents the nested-placeholder bug where a general pattern (e.g.
    NUMBER) matches digits inside an already-replaced placeholder, and
    ensures that specific patterns like IPV4 take precedence over NUMBER
    on overlapping regions such as ``192.168.1.1``.

    Group names are stripped so that bundles emitted by the LLM (which
    often reuse identical group names) do not raise
    ``re.error: redefinition of group name``.
    """

    def __init__(self, masks: Sequence[Mask]):
        self._compiled: list[tuple[Mask, re.Pattern[str], str, int]] = []
        for mask in masks:
            pattern = _normalise_pattern(mask.pattern)
            if pattern is None:
                continue
            try:
                compiled = re.compile(_strip_named_groups(pattern))
            except re.error:
                # Skip patterns that fail to compile — caller is expected to
                # validate up-front via ``validate_regexes``.  This guard
                # keeps the parser resilient when an LLM emits a single
                # malformed entry inside an otherwise good bundle.
                continue
            placeholder = _placeholder_for(mask.label)
            priority = _mask_priority(mask.label)
            self._compiled.append((mask, compiled, placeholder, priority))

    @property
    def masks(self) -> list[Mask]:
        return [mask for mask, _, _, _ in self._compiled]

    def apply(self, line: str) -> str:
        """Replace variable spans with typed placeholders.

        All masks find matches on the **original** line.  Candidate spans
        are sorted so that the leftmost, highest-priority, longest match
        wins, and overlapping spans are discarded.  The selected spans
        are then substituted in a single pass.
        """
        # 1. Collect every candidate span from every mask.
        candidates: list[tuple[int, int, str, int]] = []
        for _, compiled, placeholder, priority in self._compiled:
            for m in compiled.finditer(line):
                candidates.append((m.start(), m.end(), placeholder, priority))

        if not candidates:
            return line

        # 2. Sort: leftmost first; ties broken by higher priority, then
        #    longer span.
        candidates.sort(key=lambda c: (c[0], -c[3], -(c[1] - c[0])))

        # 3. Greedy non-overlapping selection (left to right).
        selected: list[tuple[int, int, str]] = []
        covered_end = 0
        for start, end, placeholder, _ in candidates:
            if start >= covered_end:
                selected.append((start, end, placeholder))
                covered_end = end

        # 4. Build the result string.
        parts: list[str] = []
        pos = 0
        for start, end, placeholder in selected:
            parts.append(line[pos:start])
            parts.append(placeholder)
            pos = end
        parts.append(line[pos:])
        return "".join(parts)
