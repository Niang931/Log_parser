"""Offline fallback synthesiser mimicking DeepSeek-R1 behaviour.

This module produces a regex *mask bundle* without invoking any LLM.
The bundle mirrors what the paper's fine-tuned ``DeepSeek-R1:8B``
checkpoint emits in Listing 2 of Section "LLM Configuration":

* Always include the four core variable classes (timestamp, log level,
  named identifier, IPv4 address).
* Mine optional classes (HEX, UUID, PATH, EMAIL, URL, MAC) from the
  sampled logs based on token shape.

The function is fully deterministic: the output is a function of the
sample contents only and never depends on insertion order, system
locale, or hash randomisation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from ..masks_types import Mask
from ..tokenize import tokenize
from ..utils.regex_library import REGEX_CLASSES


@dataclass
class StubConfig:
    require_core_classes: bool = True


CORE_LABELS = ("TIMESTAMP", "LOGLEVEL", "NUMBER", "IPV4")


# Optional mask classes the stub may add when their shape appears in the
# sample.  Keys are *label*, values are ``(detector_regex, free_pattern,
# justification)``.
_OPTIONAL_CLASSES = {
    "HEX": (
        re.compile(r"^0x[0-9a-fA-F]+$"),
        r"\b0x[0-9a-fA-F]+\b",
        "Hexadecimal identifiers (e.g. 0xDEADBEEF)",
    ),
    "UUID": (
        re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
        "RFC 4122 UUIDs",
    ),
    "PATH": (
        re.compile(r"^/[A-Za-z0-9_.\-/]+$"),
        r"(?<!\S)/[A-Za-z0-9_.\-/]+",
        "Unix-style filesystem paths",
    ),
    "EMAIL": (
        re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"),
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "Email addresses",
    ),
    "URL": (
        re.compile(r"^https?://[^\s]+$"),
        r"https?://[^\s]+",
        "HTTP(S) URLs",
    ),
    "MAC": (
        re.compile(r"^(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$"),
        r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b",
        "MAC addresses",
    ),
}


def _core_masks() -> List[Mask]:
    """Return the canonical four-class core bundle in deterministic order."""
    by_name = {cls.name: cls for cls in REGEX_CLASSES}
    return [
        Mask(label=name, pattern=by_name[name].free_pattern, justification=by_name[name].description)
        for name in CORE_LABELS
    ]


def _infer_optional_masks(logs: Sequence[str]) -> List[Mask]:
    found: dict[str, Mask] = {}
    for line in logs:
        for token in tokenize(line):
            for label, (detector, pattern, why) in _OPTIONAL_CLASSES.items():
                if label in found:
                    continue
                if detector.match(token):
                    found[label] = Mask(label=label, pattern=pattern, justification=why)
    # Sort keys deterministically for reproducibility.
    return [found[k] for k in sorted(found)]


def synthesize_offline(logs: Sequence[str], config: StubConfig | None = None) -> List[Mask]:
    """Synthesise a deterministic regex bundle from a log sample.

    Always includes the four core mask classes, then opportunistically
    adds optional classes when matching tokens appear in the sample.
    Raises ``ValueError`` if the required core classes cannot be
    produced (which only happens if the developer has tampered with the
    canonical regex library).
    """
    cfg = config or StubConfig()
    masks = _core_masks()
    masks.extend(_infer_optional_masks(logs))

    if cfg.require_core_classes:
        present = {mask.label for mask in masks}
        missing = set(CORE_LABELS) - present
        if missing:
            raise ValueError(f"Offline stub failed to synthesise required masks: {sorted(missing)}")
    return masks
