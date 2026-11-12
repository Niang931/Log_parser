"""Best-effort event timestamps from raw log lines.

Device clocks drift and some lines carry no timestamp at all, so callers
fall back to ingest time when nothing parses.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone, tzinfo

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # 2026-04-13 10:15:23.123 / 2026-04-13T10:15:23Z / ... 10:15:23 UTC / ... 10:15:23+08:00
    (
        re.compile(
            r"(\d{4})[-/](\d{2})[-/](\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:[.,](\d{1,6}))?"
            r"(?:\s?(Z|UTC|GMT|[+-]\d{2}(?::?\d{2})?)(?![\d.:]))?"
        ),
        "iso",
    ),
    # BGL: 2005-06-03-15.42.50.675872
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})\.(\d{2})\.(\d{2})(?:\.(\d{1,6}))?"), "iso"),
    # HDFS: 081109 203615
    (re.compile(r"^(\d{2})(\d{2})(\d{2}) (\d{2})(\d{2})(\d{2})\b"), "yymmdd"),
]


def _zone(text: str | None, local: tzinfo) -> tzinfo:
    """The zone written after the time, else ``local`` (the host's clock for a zone-less time)."""
    if not text:
        return local
    if text in ("Z", "UTC", "GMT"):
        return UTC
    sign = -1 if text[0] == "-" else 1
    digits = text[1:].replace(":", "")
    return timezone(sign * timedelta(hours=int(digits[:2]), minutes=int(digits[2:4] or 0)))


def parse_timestamp(text: str, local: tzinfo = UTC) -> datetime | None:
    found = _parse(text, local)
    return found[0] if found else None


def _parse(text: str, local: tzinfo) -> tuple[datetime, bool] | None:
    """(time, whether its zone was written in the text)."""
    for pattern, kind in _PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groups()
        try:
            if kind == "yymmdd":
                yy, mo, dd, hh, mi, ss = (int(g) for g in groups)
                return datetime(2000 + yy, mo, dd, hh, mi, ss, tzinfo=local), False
            year, mo, dd, hh, mi, ss = (int(g) for g in groups[:6])
            micro = int((groups[6] or "0").ljust(6, "0")[:6])
            written = groups[7] if len(groups) > 7 else None
            return datetime(year, mo, dd, hh, mi, ss, micro, tzinfo=_zone(written, local)), bool(
                written
            )
        except ValueError:
            continue
    return None


def event_time(
    raw: str, payload: dict[str, str], fallback: datetime | None = None, local: tzinfo = UTC
) -> datetime:
    """The line's own time. A zone written in the line wins; a zone-less time is read as
    ``local`` (an agent's host clock), defaulting to UTC."""
    line = _parse(raw, local)
    for key in ("timestamp", "time", "ts"):
        if key in payload and (field := _parse(payload[key], local)):
            # An extracted field often stops before the zone (`... 09:23:24 UTC`): the line's zone wins.
            return line[0] if line and line[1] and not field[1] else field[0]
    return line[0] if line else fallback or datetime.now(UTC)
