"""Replay log files from a directory as a stream of events."""
from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LogEvent:
    device_id: str
    source: str
    line_no: int
    line: str


def replay(root: Path) -> Iterator[LogEvent]:
    paths = [root] if root.is_file() else sorted(
        Path(dirpath) / name for dirpath, _dirs, files in os.walk(root) for name in files
    )
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.rstrip("\r\n")
                if line.strip():
                    yield LogEvent(path.stem, str(path), line_no, line)
