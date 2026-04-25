"""Tests for the canonical regex library."""
from __future__ import annotations

import re

import pytest

from deepparse.utils.regex_library import (
    REGEX_CLASSES,
    canonical_masks,
    classify_token,
    validate_regexes,
)


def test_classify_token_matches_classes():
    assert classify_token("2024-01-01T00:00:00") == "TIMESTAMP"
    assert classify_token("192.168.1.10") == "IPV4"
    assert classify_token("0xDEADBEEF") == "HEX"
    assert classify_token("INFO") == "LOGLEVEL"
    assert classify_token("/var/log/app.log") == "PATH"
    assert classify_token("3.1415") == "NUMBER"
    assert classify_token("hello") is None


def test_validate_regexes_strict_blocks_greedy():
    with pytest.raises(ValueError, match="Strict mode"):
        validate_regexes([r".*"], strict=True)


def test_validate_regexes_rejects_invalid():
    with pytest.raises(ValueError):
        validate_regexes([r"("])


def test_regex_classes_compile_and_match_examples():
    examples = {
        "TIMESTAMP": "2024-01-01T00:00:00",
        "IPV4": "10.0.0.1",
        "UUID": "550e8400-e29b-41d4-a716-446655440000",
        "HEX": "0xCAFE",
        "LOGLEVEL": "ERROR",
        "PATH": "/etc/hosts",
        "NUMBER": "-123",
    }
    for cls in REGEX_CLASSES:
        anchored = cls.compile()
        free = cls.compile_free()
        assert anchored.match(examples[cls.name]), cls.name
        assert free.search(f"prefix {examples[cls.name]} suffix"), cls.name


def test_canonical_masks_compile():
    for mask in canonical_masks():
        re.compile(mask.pattern)
