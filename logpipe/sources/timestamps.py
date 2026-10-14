"""Best-effort event timestamps from raw log lines.

Device clocks drift and some lines carry no timestamp at all, so callers
fall back to ingest time when nothing parses.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # 2026-04-13 10:15:23.123 / 2026-04-13T10:15:23Z
    (re.compile(r"(\d{4})[-/](\d{2})[-/](\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:[.,](\d{1,6}))?"), "iso"),
    # BGL: 2005-06-03-15.42.50.675872
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})\.(\d{2})\.(\d{2})(?:\.(\d{1,6}))?"), "iso"),
    # HDFS: 081109 203615
    (re.compile(r"^(\d{2})(\d{2})(\d{2}) (\d{2})(\d{2})(\d{2})\b"), "yymmdd"),
]


def parse_timestamp(text: str) -> datetime | None:
    for pattern, kind in _PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groups()
        try:
            if kind == "yymmdd":
                yy, mo, dd, hh, mi, ss = (int(g) for g in groups)
                return datetime(2000 + yy, mo, dd, hh, mi, ss, tzinfo=UTC)
            year, mo, dd, hh, mi, ss = (int(g) for g in groups[:6])
            micro = int((groups[6] or "0").ljust(6, "0")[:6])
            return datetime(year, mo, dd, hh, mi, ss, micro, tzinfo=UTC)
        except ValueError:
            continue
    return None


def event_time(raw: str, payload: dict[str, str], fallback: datetime | None = None) -> datetime:
    for key in ("timestamp", "time", "ts"):
        if key in payload and (parsed := parse_timestamp(payload[key])):
            return parsed
    return parse_timestamp(raw) or fallback or datetime.now(UTC)
