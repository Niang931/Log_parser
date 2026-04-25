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
from typing import List, Sequence

from ..masks_types import Mask


def _placeholder_for(label: str) -> str:
    """Return the typed placeholder string for a mask label."""
    safe = re.sub(r"[^A-Z0-9]+", "", label.upper()) or "VAR"
    return f"<VAR:{safe}>"


def _strip_named_groups(pattern: str) -> str:
    """Drop ``(?P<name>...)`` group names so multiple masks can share a name."""
    return re.sub(r"\(\?P<[^>]+>", "(", pattern)


class MaskApplier:
    """Apply regex masks to log lines before Drain clustering.

    Each mask substitutes its match span with a *typed placeholder*
    derived from the mask label, e.g. label ``"IPV4"`` → ``<VAR:IPV4>``.
    Masks are applied in declaration order; the first mask wins on
    overlapping spans.  Group names are stripped so that bundles emitted
    by the LLM (which often reuse identical group names) do not raise
    ``re.error: redefinition of group name``.
    """

    def __init__(self, masks: Sequence[Mask]):
        self._compiled: List[tuple[Mask, re.Pattern[str], str]] = []
        for mask in masks:
            try:
                compiled = re.compile(_strip_named_groups(mask.pattern))
            except re.error:
                # Skip patterns that fail to compile — caller is expected to
                # validate up-front via ``validate_regexes``.  This guard
                # keeps the parser resilient when an LLM emits a single
                # malformed entry inside an otherwise good bundle.
                continue
            self._compiled.append((mask, compiled, _placeholder_for(mask.label)))

    @property
    def masks(self) -> List[Mask]:
        return [mask for mask, _, _ in self._compiled]

    def apply(self, line: str) -> str:
        masked_line = line
        for _mask, compiled, placeholder in self._compiled:
            masked_line = compiled.sub(placeholder, masked_line)
        return masked_line
