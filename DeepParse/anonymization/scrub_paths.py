#!/usr/bin/env python3
"""Scrub user-specific tokens from files to preserve anonymity."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

PATTERNS = [
    (re.compile(r"/[\w\-_/]+"), "/anon/path"),
    (re.compile(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+"), "anon@example.com"),
    (re.compile(r"hostname=[^,\s]+"), "hostname=anon"),
]


def scrub(text: str) -> str:
    result = text
    for pattern, replacement in PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrub identifiable paths from files")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    content = args.input.read_text(encoding="utf-8")
    scrubbed = scrub(content)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(scrubbed, encoding="utf-8")
    print(f"Scrubbed content written to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
