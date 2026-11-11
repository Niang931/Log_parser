"""What every log store adapter offers the server: UI reads (the Log Explorer, facets, the
learning jobs) and a query compiler (the Query page's builder and Ask's reads). One adapter per
backend: ``memory`` (rows in Python, queries on a SQLite copy), ``clickhouse`` and ``mongodb``.

Everything here is shared: how a line's level is read, the builder's vocabulary, and how rows
and alerts are turned into JSON.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from datetime import UTC, datetime, timedelta

from ...enrich.validate import KEY_RE
from ...storage.base import LogRow

FILTERS = ("host", "service", "source", "status", "template_id", "level")
FACETS = ("level", "service", "host", "source", "status")
LEVELS = ("error", "warn", "info", "debug")
HEAD = 120  # explicit level words are looked for in the line's first characters only

# (level, regex, head only). Written in the regex subset Python, ClickHouse (RE2) and MongoDB
# (PCRE) share, so facet counts in the store and rows read in Python always agree. First match
# wins: a declared level beats heuristics, the way Datadog's status remapper works.
LEVEL_RULES = (
    ("error", r"\b(FATAL|CRIT|CRITICAL|ERROR|ERR|SEVERE|EMERG|PANIC)\b", True),
    ("warn", r"\b(WARN|WARNING)\b", True),
    ("debug", r"\b(DEBUG|TRACE)\b", True),
    ("info", r"\b(INFO|NOTICE)\b", True),
    ("error", r"HTTP/[0-9.]+\" 5[0-9][0-9] |(?i:\b(exception|traceback|panic)\b)", False),
    ("warn", r"HTTP/[0-9.]+\" 4[0-9][0-9] ", False),
)
_RULES = [(lvl, re.compile(rx), head) for lvl, rx, head in LEVEL_RULES]

# The Query builder's fields: name -> what it is. Each compiler maps them to its own columns.
FIELDS = {
    "host": "host that shipped the line",
    "service": "service set on the agent source",
    "source": "source set on the agent source (e.g. python, nginx)",
    "level": "error / warn / info / debug, read from the line",
    "status": "FINAL (parsed) or PENDING",
    "template_id": "template the line matched",
    "message": "the raw line",
}
MEASURES = ("count", "count_distinct", "avg", "min", "max", "sum", "p95")
OPS = ("=", "!=", "contains", "not contains", ">", "<", "in")  # in: comma-separated values


def level(message: str, tags=()) -> str:
    """A severity the sender stated (OTLP severity -> ``severity:<level>`` tag) wins; else read
    from the text."""
    for tag in tags:
        if tag.startswith("severity:") and tag[9:] in LEVELS:
            return tag[9:]
    for lvl, rx, head in _RULES:
        if rx.search(message[:HEAD] if head else message):
            return lvl
    return "info"


def step_for(minutes: int) -> int:
    """Bucket seconds giving 40-90 bars over the range, on round intervals."""
    for step in (10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800):
        if minutes * 60 / step <= 90:
            return step
    return 21600


def histogram_keys(by: str) -> tuple[str, ...]:
    return LEVELS if by == "level" else ("FINAL", "PENDING")


def zero_fill(
    minutes: int, step: int, counts: dict[int, Counter], keys: tuple[str, ...]
) -> list[dict]:
    """Every bucket in the range, empty ones included, so bars keep their width and gaps show."""
    end = int(datetime.now(UTC).timestamp()) // step * step
    start = end - (minutes * 60 // step) * step
    return [
        {"t": t * 1000, **{k: counts.get(t, Counter())[k] for k in keys}}
        for t in range(start + step, end + step, step)
    ]


def iso(t) -> str:
    if isinstance(t, datetime):
        return (t if t.tzinfo else t.replace(tzinfo=UTC)).astimezone(UTC).isoformat()
    return str(t)


def row_json(r: LogRow) -> dict:
    return {
        "log_id": str(r.log_id),
        "timestamp": r.timestamp.isoformat(),
        "host": r.device_id,
        "service": r.service,
        "source": r.source,
        "tags": list(r.tags),
        "status": str(r.status),
        "template_id": r.template_id,
        "payload": r.payload,
        "message": r.raw_message,
        "level": level(r.raw_message, r.tags),
    }


def alert_json(a: dict) -> dict:
    return {
        "alert_id": str(a["alert_id"]),
        "key": a["device_id"],
        "window_start": iso(a["window_start"]),
        "window_end": iso(a["window_end"]),
        "template_ids": list(a["template_ids"]),
        "miss_ratio": a["miss_ratio"],
        "novelty_ratio": a["novelty_ratio"],
        "reasons": list(a["reasons"]),
        "model_version": a["model_version"],
    }


def absolute_range(spec: dict) -> tuple[datetime, datetime] | None:
    """``start``/``end`` (ISO 8601) when a builder spec has them: an absolute range instead of the
    last ``minutes``."""
    if not (spec.get("start") or spec.get("end")):
        return None
    try:
        start, end = (datetime.fromisoformat(str(spec[k])) for k in ("start", "end"))
    except (KeyError, ValueError):
        raise ValueError("start and end must both be ISO 8601 times") from None
    start, end = (t if t.tzinfo else t.replace(tzinfo=UTC) for t in (start, end))
    if not timedelta(0) < end - start <= timedelta(days=90):
        raise ValueError("time range must be between 1 minute and 90 days")
    return start.astimezone(UTC), end.astimezone(UTC)


def relative_minutes(spec: dict) -> int:
    minutes = int(spec.get("minutes") or 60)
    if not 1 <= minutes <= 60 * 24 * 90:
        raise ValueError("time range must be between 1 minute and 90 days")
    return minutes


def span_minutes(spec: dict) -> int:
    rng = absolute_range(spec)
    return (int((rng[1] - rng[0]).total_seconds() // 60) or 1) if rng else relative_minutes(spec)


def check_field(name: str) -> str:
    """A builder field or ``@key`` for a field a template extracted; ValueError otherwise."""
    if name.startswith("@"):
        if not KEY_RE.match(name[1:]):
            raise ValueError(f"field {name!r}: extracted field names are lower_snake_case")
        return name
    if name not in FIELDS:
        raise ValueError(
            f"unknown field {name!r}; use one of {', '.join(FIELDS)} or @extracted_field"
        )
    return name


def alias(field: str, i: int) -> str:
    """A result column name for a group-by field."""
    bare = field.lstrip("@")
    return bare if KEY_RE.match(bare) else f"g{i}"


def in_values(name: str, value: str) -> list[str]:
    values = [v.strip() for v in value.split(",") if v.strip()]
    if not values:
        raise ValueError(f"{name} in needs at least one value")
    if name == "template_id" and not all(v.lstrip("-").isdigit() for v in values):
        raise ValueError("template_id in needs numbers")
    return values


def measure_of(spec: dict) -> tuple[str, str | None, str]:
    """(fn, field, result column) of a builder spec's measure."""
    measure = spec.get("measure") or {"fn": "count"}
    fn = measure.get("fn", "count")
    if fn not in MEASURES:
        raise ValueError(f"measure must be one of {', '.join(MEASURES)}")
    if fn == "count":
        return fn, None, "lines"
    target = str(measure.get("field") or "")
    if not target:
        raise ValueError(f"{fn} needs a field")
    check_field(target)
    return fn, target, f"{fn}_{target.lstrip('@')}"


class Compiler(ABC):
    """Turns a request into the exact query text that will run (shown to a person first), for
    one backend. Field names are checked and values quoted here, so a spec can't inject."""

    engine: str  # sqlite | clickhouse | mongodb
    table: str

    @abstractmethod
    def build(self, spec: dict) -> str:
        """The Query page's visual builder: time range, filters, a measure, group-by."""

    @abstractmethod
    def schema(self, fields: list[str]) -> dict:
        """What the Query page's sidebar lists."""

    @abstractmethod
    def rows(
        self,
        filters: list[dict],
        start: datetime,
        end: datetime,
        limit: int = 200,
        newest_first: bool = False,
        payload: bool = False,
    ) -> str:
        """Matching lines (log_id, timestamp, host, service, source, template_id, level, message)."""

    @abstractmethod
    def template_counts(self, filters: list[dict], start: datetime, end: datetime) -> str:
        """Per (template, service): count, first and last time, and the first line (example)."""

    @abstractmethod
    def buckets(
        self, filters: list[dict], start: datetime, end: datetime, step: int, by: str
    ) -> str:
        """Lines per ``by`` (template_id or service) per ``step`` seconds, bucket 0 at ``start``."""

    @abstractmethod
    def template_sample(self, template_id: int, keys: list[str]) -> str:
        """The latest lines of one template with its extracted fields as columns."""


class StoreReads(ABC):
    """The UI's reads on one backend, plus ``query``: free-form text on the read-only path."""

    compiler: Compiler

    @property
    def engine(self) -> str:
        return self.compiler.engine

    @abstractmethod
    def recent(
        self, filters: dict, limit: int, text: str = "", minutes: int | None = None
    ) -> list[dict]: ...

    @abstractmethod
    def between(
        self,
        start: datetime,
        end: datetime,
        limit: int,
        host: str | None = None,
        service: str | None = None,
    ) -> list[dict]: ...

    @abstractmethod
    def histogram(
        self, minutes: int, by: str, filters: dict | None = None, text: str = ""
    ) -> dict: ...

    @abstractmethod
    def top(
        self, field: str, minutes: int, limit: int, filters: dict | None = None, text: str = ""
    ) -> list[tuple]: ...

    @abstractmethod
    def source_lines(self, host: str, service: str, source: str, limit: int) -> list[str]: ...

    @abstractmethod
    def source_rows(self, host: str, service: str, source: str, limit: int) -> list[tuple]: ...

    @abstractmethod
    def alerts(self, limit: int) -> list[dict]: ...

    @abstractmethod
    def query(self, text: str, max_rows: int | None = None) -> dict:
        """Run SQL (or a pipeline) on the read-only path: {columns, rows, engine, table}."""

    def records(self, text: str, max_rows: int) -> list[dict]:
        """``query`` with rows as dicts, times as ISO text and a level for rows that carry tags."""
        out = self.query(text, max_rows=max_rows)
        rows = [dict(zip(out["columns"], r)) for r in out["rows"]]
        for r in rows:
            for k in ("timestamp", "first", "last"):
                if k in r:
                    r[k] = iso(r[k])
            if "tags" in r:  # a backend without a level column: from the first line, like level()
                r["level"] = level(r.get("example") or "", r.pop("tags") or ())
        return rows
