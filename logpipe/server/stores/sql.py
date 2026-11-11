"""The SQL side: one compiler for ClickHouse (``iot_logs``) and SQLite (the memory store's copy,
``logs``). A dialect subclass supplies the few expressions that differ; the query shapes live here
once. Plus ``guard_sql``, which every free-form SQL query passes before it runs.
"""

from __future__ import annotations

import re
from abc import abstractmethod
from datetime import UTC, datetime
from typing import ClassVar

from ...settings import get_settings
from .base import (
    FIELDS,
    MEASURES,
    OPS,
    Compiler,
    absolute_range,
    alias,
    check_field,
    in_values,
    measure_of,
    relative_minutes,
    span_minutes,
    step_for,
)

_LITERAL = re.compile(r"'(?:[^'\\]|\\.|'')*'|--[^\n]*|/\*.*?\*/|#[^\n]*", re.DOTALL)
_QUOTED_NAME = re.compile(r'"((?:[^"]|"")*)"|`((?:[^`]|``)*)`')
_TABLE_FUNCTION = re.compile(r"\b(?:from|join)\s+([a-z_][\w]*)\s*\(", re.IGNORECASE)
_DENIED = re.compile(
    r"\b(?:into\s+outfile|attach|detach|pragma|vacuum|load_extension|readfile|writefile|"
    r"fts3_tokenizer|system\s*\.|information_schema\s*\.)",
    re.IGNORECASE,
)


def guard_sql(sql: str) -> str:
    """One read-only SELECT over the log tables, or ValueError. Table functions (url, file, remote,
    s3, mysql, ...) would read or reach outside the store, so none are allowed; nor are system
    tables, file output or SQLite's ATTACH/PRAGMA. The database's own read-only mode (and, on
    ClickHouse, a read-only user) stay the real barrier; this refuses early with a clear message."""
    sql = sql.strip().rstrip(";").strip()
    bare = _QUOTED_NAME.sub(lambda m: m.group(1) or m.group(2) or "", _LITERAL.sub("''", sql))
    if not re.match(r"(?is)\s*(select|with)\b", bare):
        raise ValueError("only SELECT / WITH queries")
    if ";" in bare:
        raise ValueError("one statement at a time")
    if m := _TABLE_FUNCTION.search(bare):
        raise ValueError(f"table function {m.group(1)}() is not allowed; query iot_logs / logs")
    if m := _DENIED.search(bare):
        raise ValueError(f"{' '.join(m.group(0).split()).rstrip('.')} is not allowed in a query")
    return sql


class SqlCompiler(Compiler):
    columns: ClassVar[dict[str, str]]  # builder field -> column expression
    source: str  # FROM clause
    about_table: ClassVar[
        dict[str, str]
    ] = {}  # sidebar names that differ from the column expression

    @abstractmethod
    def quote(self, value: str) -> str: ...

    @abstractmethod
    def extracted(self, key: str, numeric: bool) -> str:
        """A field a template extracted (``payload``), as text or as a number."""

    def payload_json(self) -> str:
        """The extracted fields as JSON text."""
        return "payload"

    @abstractmethod
    def time_col(self) -> str:
        """``timestamp`` as something that compares with ``instant``."""

    @abstractmethod
    def instant(self, t: datetime) -> str: ...

    @abstractmethod
    def since_minutes(self, minutes: int) -> str: ...

    @abstractmethod
    def contains(self, col: str, value: str) -> str: ...

    @abstractmethod
    def unix_seconds(self) -> str: ...

    @abstractmethod
    def time_bucket(self, step: int) -> str: ...

    @abstractmethod
    def aggregate(self, fn: str, col: str) -> str: ...

    log_id = "log_id"
    count = "count(*)"

    def field(self, name: str, numeric: bool = False) -> str:
        check_field(name)
        return self.extracted(name[1:], numeric) if name.startswith("@") else self.columns[name]

    def time_where(self, spec: dict) -> list[str]:
        if rng := absolute_range(spec):
            return [
                f"{self.time_col()} >= {self.instant(rng[0])}",
                f"{self.time_col()} <= {self.instant(rng[1])}",
            ]
        return [self.since_minutes(relative_minutes(spec))]

    def filter_where(self, filters: list[dict] | None) -> list[str]:
        where = []
        for f in filters or []:
            op, value, name = f.get("op", "="), str(f.get("value", "")), str(f.get("field", ""))
            if op not in OPS:
                raise ValueError(f"operator must be one of {', '.join(OPS)}")
            col = self.field(name, numeric=op in (">", "<"))
            if op == "in":
                values = in_values(name, value)
                quoted = values if name == "template_id" else [self.quote(v) for v in values]
                where.append(f"{col} IN ({', '.join(quoted)})")
            elif op in ("contains", "not contains"):
                where.append(("NOT " if op == "not contains" else "") + self.contains(col, value))
            elif op in (">", "<"):
                try:
                    number = float(value)
                except ValueError:
                    raise ValueError(f"{name} {op} needs a number, got {value!r}") from None
                where.append(f"{col} {op} {number}")
            else:
                where.append(f"{col} {op} {self.quote(value)}")
        return where

    def _where(self, filters: list[dict], start: datetime, end: datetime) -> str:
        rng = {"start": start.astimezone(UTC).isoformat(), "end": end.astimezone(UTC).isoformat()}
        return "\n  AND ".join(self.time_where(rng) + self.filter_where(filters))

    def build(self, spec: dict) -> str:
        where = self.time_where(spec) + self.filter_where(spec.get("filters"))
        fn, target, label = measure_of(spec)
        value = (
            self.count
            if fn == "count"
            else self.aggregate(fn, self.field(target or "", numeric=fn != "count_distinct"))
        )
        select, group = [], []
        if spec.get("over_time"):
            select.append(f"{self.time_bucket(step_for(span_minutes(spec)))} AS time")
            group.append("time")
        for i, g in enumerate((spec.get("group_by") or [])[:2]):
            select.append(f"{self.field(str(g))} AS {alias(str(g), i)}")
            group.append(alias(str(g), i))
        select.append(f"{value} AS {label}")
        limit = min(int(spec.get("limit") or 100), get_settings().server.max_builder_rows)
        sql = f"SELECT {', '.join(select)}\nFROM {self.source}\nWHERE " + "\n  AND ".join(where)
        if group:
            sql += f"\nGROUP BY {', '.join(group)}"
        return (
            sql
            + "\nORDER BY "
            + ("time" if spec.get("over_time") else f"{label} DESC")
            + f"\nLIMIT {limit}"
        )

    def schema(self, fields: list[str]) -> dict:
        cols = [{"name": "timestamp", "about": "when the line was logged"}]
        cols += [
            {"name": self.about_table.get(k, self.columns[k]), "builder": k, "about": about}
            for k, about in FIELDS.items()
        ]
        cols.append({"name": "payload", "about": "fields extracted by the line's template (JSON)"})
        return {
            "engine": self.engine,
            "table": self.table,
            "columns": cols,
            "fields": sorted(fields),
            "payload_access": self.extracted("name", False),
            "measures": list(MEASURES),
            "ops": list(OPS),
        }

    def rows(
        self,
        filters: list[dict],
        start: datetime,
        end: datetime,
        limit: int = 200,
        newest_first: bool = False,
        payload: bool = False,
    ) -> str:
        limit = max(1, min(int(limit), get_settings().ask.row_cap))
        extra = f"{self.payload_json()} AS payload, " if payload else ""
        return (
            f"SELECT {self.log_id} AS log_id, timestamp, device_id AS host, service, source, template_id, "
            f"{self.columns['level']} AS level, {extra}raw_message AS message\n"
            f"FROM {self.source}\nWHERE {self._where(filters, start, end)}\n"
            f"ORDER BY {self.time_col()} {'DESC' if newest_first else 'ASC'}\nLIMIT {limit}"
        )

    def buckets(
        self,
        filters: list[dict],
        start: datetime,
        end: datetime,
        step: int,
        by: str = "template_id",
    ) -> str:
        if by not in ("template_id", "service"):
            raise ValueError("buckets are by template_id or service")
        step, origin = max(1, int(step)), int(start.timestamp())
        return (
            f"SELECT {by} AS key, {self.bucket_index(origin, step)} AS bucket, {self.count} AS n\n"
            f"FROM {self.source}\nWHERE {self._where(filters, start, end)}\n"
            f"GROUP BY key, bucket\nORDER BY bucket\nLIMIT {get_settings().ask.agg_cap}"
        )

    def bucket_index(self, origin: int, step: int) -> str:
        return f"({self.unix_seconds()} - {origin}) / {step}"
