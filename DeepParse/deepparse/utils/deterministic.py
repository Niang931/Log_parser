"""Helpers enforcing deterministic execution."""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator, Tuple


@contextmanager
def timed_block(label: str) -> Iterator[Tuple[str, float]]:
    start = perf_counter()
    try:
        yield label, start
    finally:
        duration = perf_counter() - start
        print(f"[{label}] {duration:.4f}s")
