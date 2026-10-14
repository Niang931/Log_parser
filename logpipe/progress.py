"""Progress bars for long runs (calibration, training, evaluation).

Bars only draw on a terminal, so tests and CI logs stay clean.  Print through
:func:`write` while a bar is open, or the line tears the bar.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import IO, TypeVar

from tqdm import tqdm

T = TypeVar("T")


def progress(
    iterable: Iterable[T], desc: str, *, total: int | None = None, unit: str = "it"
) -> tqdm[T]:
    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=False, disable=None,
                dynamic_ncols=True)


def copy_bytes(src: IO[bytes], dst: IO[bytes], desc: str, *, total: int | None = None) -> None:
    """Copy ``src`` to ``dst`` with a byte-count bar (``total`` from Content-Length, if known)."""
    with tqdm(total=total, desc=desc, unit="B", unit_scale=True, unit_divisor=1024,
              leave=False, disable=None, dynamic_ncols=True) as bar:
        while chunk := src.read(1 << 20):
            dst.write(chunk)
            bar.update(len(chunk))


def write(message: str) -> None:
    tqdm.write(message)
