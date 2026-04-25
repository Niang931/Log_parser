"""Canonical regex classes shared between mask synthesisers and Drain.

The classes here encode the variable categories the paper expects a
properly trained synthesis model to emit (timestamps, IPv4 addresses,
hex literals, numbers, log levels, UUIDs, paths).  They serve two
roles:

* They are the building blocks of the offline synthesis stub when no
  Hugging Face checkpoint is available.
* They are used by the evaluation runner to derive *canonical* ground
  truth templates when a dataset is shipped without an explicit
  ``templates.json`` file.

The patterns deliberately use anchored, non-greedy expressions and
provide both an *anchored* form (for token classification) and a
*free* form (for substitution inside a longer line).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

from ..masks_types import Mask


@dataclass(frozen=True)
class RegexClass:
    name: str
    pattern: str           # anchored form for token classification
    free_pattern: str      # un-anchored form for line substitution
    description: str

    def compile(self) -> re.Pattern[str]:
        return re.compile(self.pattern)

    def compile_free(self) -> re.Pattern[str]:
        return re.compile(self.free_pattern)


REGEX_CLASSES: List[RegexClass] = [
    RegexClass(
        name="TIMESTAMP",
        pattern=r"^\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?$",
        free_pattern=r"\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?",
        description="ISO8601 timestamp",
    ),
    RegexClass(
        name="IPV4",
        pattern=r"^(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?$",
        free_pattern=r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b",
        description="IPv4 address with optional port",
    ),
    RegexClass(
        name="UUID",
        pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        free_pattern=r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
        description="UUID identifier",
    ),
    RegexClass(
        name="HEX",
        pattern=r"^0x[0-9a-fA-F]+$",
        free_pattern=r"\b0x[0-9a-fA-F]+\b",
        description="Hexadecimal identifier",
    ),
    RegexClass(
        name="LOGLEVEL",
        pattern=r"^(?:TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)$",
        free_pattern=r"\b(?:TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b",
        description="Log level token",
    ),
    RegexClass(
        name="PATH",
        pattern=r"^/[A-Za-z0-9_.\-/]+$",
        free_pattern=r"(?<!\S)/[A-Za-z0-9_.\-/]+",
        description="Unix-style filesystem path",
    ),
    RegexClass(
        name="NUMBER",
        pattern=r"^-?\d+(?:\.\d+)?$",
        free_pattern=r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])",
        description="Numeric literal",
    ),
]


def canonical_regex_map() -> Dict[str, re.Pattern[str]]:
    """Map class name → compiled anchored pattern (for token classification)."""
    return {cls.name: cls.compile() for cls in REGEX_CLASSES}


def classify_token(token: str) -> str | None:
    """Return the canonical class name for ``token`` or ``None``."""
    for cls in REGEX_CLASSES:
        if cls.compile().match(token):
            return cls.name
    return None


def canonical_masks() -> List[Mask]:
    """Return the canonical mask bundle used as ground-truth oracle."""
    return [
        Mask(
            label=cls.name,
            pattern=cls.free_pattern,
            justification=cls.description,
        )
        for cls in REGEX_CLASSES
    ]


def validate_regexes(regexes: Iterable[str], strict: bool = False) -> List[str]:
    """Validate that each regex compiles; in ``strict`` mode reject ``.*``.

    Returns the validated list (unchanged) so that callers can chain
    validation directly into a pipeline.
    """
    validated: List[str] = []
    for regex in regexes:
        if strict and ".*" in regex:
            raise ValueError(f"Strict mode forbids greedy pattern: {regex}")
        try:
            re.compile(regex)
        except re.error as exc:
            raise ValueError(f"Invalid regex {regex!r}: {exc}") from exc
        validated.append(regex)
    return validated
