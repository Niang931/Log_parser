"""The log-store side of Ask: a few typed reads, each compiled to one SQL query (ClickHouse, or
the memory store's SQLite) or one aggregation pipeline (MongoDB) *before* it runs, so the person
approving it sees exactly what will execute. Every one of them runs through ``reads.query``: the
read-only path (``guard_sql`` + ClickHouse ``readonly=2`` / a read-only user, SQLite
``query_only``, the Mongo stage whitelist + a read-only user).

Values never reach the query text unquoted: names are checked against fixed columns, strings go
through the builders' quoting, numbers are ints.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from ..server import querybuild as qb
from ..server import querybuild_mongo as qbm
from ..server.reads import LEVEL_SQL
from .spec import QuerySpec, levels_at_least

ROW_CAP = 2000  # lines one Ask read may return
AGG_CAP = 20_000  # rows of an aggregate (template x bucket)


def engine_of(reads) -> str:
    return {"MemoryReads": "sqlite", "MongoReads": "mongodb"}.get(type(reads).__name__, "clickhouse")


@dataclass
class Filters:
    services: list[str] = field(default_factory=list)
    hosts: list[str] = field(default_factory=list)
    levels: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    template_ids: list[int] = field(default_factory=list)

    @classmethod
    def of(cls, spec: QuerySpec) -> Filters:
        return cls(list(spec.services), list(spec.hosts), levels_at_least(spec.min_level),
                   list(spec.keywords), list(spec.template_ids))

    def builder(self) -> list[dict]:
        """As the Query builder's filter list (the builders quote and check every value)."""
        out = []
        for name, values in (("service", self.services), ("host", self.hosts), ("level", self.levels)):
            if values:
                out.append({"field": name, "op": "in", "value": ",".join(v.replace(",", " ") for v in values)})
        if self.template_ids:
            out.append({"field": "template_id", "op": "in", "value": ",".join(str(int(t)) for t in self.template_ids)})
        out += [{"field": "message", "op": "contains", "value": k} for k in self.keywords]
        return out

    def describe(self) -> str:
        parts = [f"{k}: {', '.join(map(str, v))}" for k, v in asdict(self).items() if v]
        return "; ".join(parts) or "no filters"


@dataclass
class Query:
    """One read, as shown for approval: what it is for, and the exact text that will run."""
    id: str
    tool: str  # rows | template_counts | buckets | aggregate | alerts | agents | semantic
    title: str
    why: str
    args: dict
    compiled: str
    engine: str  # the store's engine, or "local" for reads that touch no log data
    status: str = "pending"  # pending | approved | rejected | done | failed
    error: str | None = None
    result_summary: str | None = None


def _iso(t) -> str:
    if isinstance(t, datetime):
        return (t if t.tzinfo else t.replace(tzinfo=UTC)).astimezone(UTC).isoformat()
    return str(t)


def _range(start: datetime, end: datetime) -> dict:
    return {"start": start.astimezone(UTC).isoformat(), "end": end.astimezone(UTC).isoformat()}


class LogStore:
    def __init__(self, reads, engine: str | None = None):
        self.reads = reads
        self.engine = engine or engine_of(reads)

    # --- compile ------------------------------------------------------------------------
    def _sql_where(self, f: Filters, start: datetime, end: datetime) -> str:
        where = qb.time_where(_range(start, end), self.engine) + qb.filter_where(f.builder(), self.engine)
        return "\n  AND ".join(where)

    def _match(self, f: Filters, start: datetime, end: datetime) -> dict:
        return {"$match": {"$and": qbm.time_match(_range(start, end)) + qbm.filter_match(f.builder())}}

    def rows(self, f: Filters, start: datetime, end: datetime, limit: int = 200, newest_first: bool = False) -> str:
        limit = max(1, min(int(limit), ROW_CAP))
        if self.engine == "mongodb":
            return json.dumps([self._match(f, start, end), {"$sort": {"timestamp": -1 if newest_first else 1}},
                               {"$limit": limit},
                               {"$project": {"_id": 0, "log_id": "$_id", "timestamp": 1, "host": "$device_id",
                                             "service": 1, "source": 1, "template_id": 1, "level": qbm.LEVEL_EXPR,
                                             "message": "$raw_message"}}], indent=1)
        ch = self.engine == "clickhouse"
        order = "timestamp" if ch else "julianday(timestamp)"
        return (f"SELECT {'toString(log_id)' if ch else 'log_id'} AS log_id, timestamp, device_id AS host, service, "
                f"source, template_id, {LEVEL_SQL if ch else 'level'} AS level, raw_message AS message\n"
                f"FROM {'iot_logs FINAL' if ch else 'logs'}\nWHERE {self._sql_where(f, start, end)}\n"
                f"ORDER BY {order} {'DESC' if newest_first else 'ASC'}\nLIMIT {limit}")

    def template_counts(self, f: Filters, start: datetime, end: datetime) -> str:
        """Per (template, service): how many lines, first and last time, and the first line itself."""
        if self.engine == "mongodb":
            return json.dumps([self._match(f, start, end), {"$sort": {"timestamp": 1}},
                               {"$group": {"_id": {"t": "$template_id", "s": "$service"}, "n": {"$sum": 1},
                                           "first": {"$first": "$timestamp"}, "last": {"$last": "$timestamp"},
                                           "example": {"$first": "$raw_message"}, "log_id": {"$first": "$_id"},
                                           "host": {"$first": "$device_id"}, "tags": {"$first": "$tags"}}},
                               {"$project": {"_id": 0, "template_id": "$_id.t", "service": "$_id.s", "n": 1, "first": 1,
                                             "last": 1, "example": 1, "log_id": 1, "host": 1, "tags": 1}},
                               {"$sort": {"first": 1}}, {"$limit": AGG_CAP}], indent=1)
        if self.engine == "clickhouse":
            return ("SELECT template_id, service, count() AS n, min(timestamp) AS first, max(timestamp) AS last,\n"
                    "  argMin(raw_message, timestamp) AS example, argMin(toString(log_id), timestamp) AS log_id,\n"
                    f"  argMin(device_id, timestamp) AS host, argMin({LEVEL_SQL}, timestamp) AS level\n"
                    f"FROM iot_logs FINAL\nWHERE {self._sql_where(f, start, end)}\n"
                    f"GROUP BY template_id, service\nORDER BY first\nLIMIT {AGG_CAP}")
        return ("SELECT template_id, service, n, timestamp AS first, last, raw_message AS example, log_id,\n"
                "  device_id AS host, level\nFROM (SELECT *, ROW_NUMBER() OVER w AS rn, COUNT(*) OVER p AS n,\n"
                "    MAX(timestamp) OVER p AS last\n"
                f"  FROM logs WHERE {self._sql_where(f, start, end)}\n"
                "  WINDOW p AS (PARTITION BY template_id, service),\n"
                "    w AS (PARTITION BY template_id, service ORDER BY julianday(timestamp)))\n"
                f"WHERE rn = 1\nORDER BY julianday(first)\nLIMIT {AGG_CAP}")

    def buckets(self, f: Filters, start: datetime, end: datetime, step: int, by: str = "template_id") -> str:
        """Lines per ``by`` (template_id or service) per ``step``-second bucket, buckets numbered
        from ``start`` (bucket 0)."""
        if by not in ("template_id", "service"):
            raise ValueError("buckets are by template_id or service")
        step = max(1, int(step))
        origin = int(start.timestamp())
        if self.engine == "mongodb":
            idx = {"$floor": {"$divide": [{"$subtract": [{"$toLong": "$timestamp"}, origin * 1000]}, step * 1000]}}
            return json.dumps([self._match(f, start, end), {"$group": {"_id": {"k": f"${by}", "b": idx}, "n": {"$sum": 1}}},
                               {"$project": {"_id": 0, "key": "$_id.k", "bucket": "$_id.b", "n": 1}},
                               {"$sort": {"bucket": 1}}, {"$limit": AGG_CAP}], indent=1)
        secs = "toUnixTimestamp(timestamp)" if self.engine == "clickhouse" else "CAST(strftime('%s', timestamp) AS INTEGER)"
        return (f"SELECT {by} AS key, intDiv({secs} - {origin}, {step}) AS bucket, count() AS n\n"
                if self.engine == "clickhouse" else
                f"SELECT {by} AS key, ({secs} - {origin}) / {step} AS bucket, count(*) AS n\n") + (
                f"FROM {'iot_logs FINAL' if self.engine == 'clickhouse' else 'logs'}\n"
                f"WHERE {self._sql_where(f, start, end)}\nGROUP BY key, bucket\nORDER BY bucket\nLIMIT {AGG_CAP}")

    def aggregate(self, spec: QuerySpec, start: datetime, end: datetime) -> str:
        agg = spec.aggregate
        assert agg is not None
        b = {**_range(start, end), "filters": Filters.of(spec).builder(),
             "measure": {"fn": agg.fn, "field": agg.field}, "group_by": agg.group_by,
             "over_time": agg.over_time, "limit": spec.limit}
        return qbm.build(b) if self.engine == "mongodb" else qb.build(b, self.engine)

    # --- run ----------------------------------------------------------------------------
    def run(self, compiled: str, max_rows: int = ROW_CAP) -> list[dict]:
        """Run one compiled read through the read-only path; rows as dicts, times as ISO text."""
        out = self.reads.query(compiled, max_rows=max_rows)
        cols = out["columns"]
        rows = [dict(zip(cols, r)) for r in out["rows"]]
        for r in rows:
            for k in ("timestamp", "first", "last"):
                if k in r:
                    r[k] = _iso(r[k])
            if self.engine == "mongodb" and "tags" in r:  # level from the first line, like reads.level()
                from ..server.reads import level

                r["level"] = level(r.get("example") or "", r.pop("tags") or ())
        return rows
