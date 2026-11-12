"""MongoDB: reads on the ``logs`` collection (one document per log_id, so no FINAL), and a
compiler that turns the same requests into aggregation pipelines (JSON text).

Every pipeline passes ``check``: only the stages in ``STAGES``, and none of the operators that run
code or write (``$where``, ``$function``, ``$accumulator``, ``$out``, ``$merge``) at any depth. With
``MONGO_READONLY_URI`` set, free-form pipelines also run as a read-only user.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from itertools import islice
from typing import ClassVar

from bson import json_util
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError

from ...settings import get_settings
from ...storage.mongo import MongoStore, _row
from .base import (
    FIELDS,
    HEAD,
    LEVEL_RULES,
    LEVELS,
    MEASURES,
    OPS,
    Compiler,
    StoreReads,
    absolute_range,
    alert_json,
    alias,
    check_field,
    histogram_keys,
    in_values,
    level,
    measure_of,
    relative_minutes,
    row_json,
    span_minutes,
    step_for,
    zero_fill,
)

# `level` is computed: LEVEL_EXPR
PATHS = {
    "host": "device_id",
    "service": "service",
    "source": "source",
    "status": "status",
    "template_id": "template_id",
    "message": "raw_message",
}
STAGES = {
    "$match",
    "$group",
    "$project",
    "$sort",
    "$limit",
    "$skip",
    "$unwind",
    "$addFields",
    "$set",
    "$unset",
    "$count",
    "$sortByCount",
    "$replaceRoot",
    "$replaceWith",
    "$bucket",
    "$bucketAuto",
}
BANNED = {"$where", "$function", "$accumulator", "$out", "$merge"}
# level() as an aggregation expression: a stated severity tag, then LEVEL_RULES in order.
LEVEL_EXPR = {
    "$switch": {
        "branches": [
            *(
                {"case": {"$in": [f"severity:{lvl}", {"$ifNull": ["$tags", []]}]}, "then": lvl}
                for lvl in LEVELS
            ),
            *(
                {
                    "case": {
                        "$regexMatch": {
                            "input": {"$substrCP": ["$raw_message", 0, HEAD]}
                            if head
                            else "$raw_message",
                            "regex": rx,
                        }
                    },
                    "then": lvl,
                }
                for lvl, rx, head in LEVEL_RULES
            ),
        ],
        "default": "info",
    }
}


def check(pipeline) -> list[dict]:
    """The pipeline if it is read-only; ValueError otherwise."""
    if (
        not isinstance(pipeline, list)
        or not pipeline
        or not all(isinstance(s, dict) and len(s) == 1 for s in pipeline)
    ):
        raise ValueError(
            'a pipeline is a JSON list of single-stage objects, e.g. [{"$match": {...}}]'
        )
    for stage in pipeline:
        name = next(iter(stage))
        if name not in STAGES:
            raise ValueError(f"stage {name} is not allowed; use one of {', '.join(sorted(STAGES))}")

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in BANNED:
                    raise ValueError(f"{k} is not allowed in a query")
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(pipeline)
    return pipeline


def mongo_date(t: datetime) -> dict:
    """Extended JSON, read back as a date by ``MongoReads.query``."""
    return {"$date": t.astimezone(UTC).isoformat().replace("+00:00", "Z")}


def _path(name: str) -> str:
    check_field(name)
    if name == "level":
        raise ValueError("level is computed; filter or group by it instead")
    return "payload." + name[1:] if name.startswith("@") else PATHS[name]


def _num(path: str) -> dict:
    return {"$convert": {"input": "$" + path, "to": "double", "onError": None, "onNull": None}}


def _value(field: str, value: str):
    if field == "template_id":
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"template_id needs a number, got {value!r}") from None
    return value


def _dump(pipeline: list[dict]) -> str:
    return json.dumps(pipeline, indent=1)


class MongoCompiler(Compiler):
    engine, table = "mongodb", "logs"

    def time_match(self, spec: dict) -> list[dict]:
        if rng := absolute_range(spec):
            return [{"timestamp": {"$gte": mongo_date(rng[0]), "$lte": mongo_date(rng[1])}}]
        since = {
            "$dateSubtract": {
                "startDate": "$$NOW",
                "unit": "minute",
                "amount": relative_minutes(spec),
            }
        }
        return [{"$expr": {"$gte": ["$timestamp", since]}}]

    def filter_match(self, filters: list[dict] | None) -> list[dict]:
        match: list[dict] = []
        for f in filters or []:
            op, value, name = f.get("op", "="), str(f.get("value", "")), str(f.get("field", ""))
            if op not in OPS:
                raise ValueError(f"operator must be one of {', '.join(OPS)}")
            if name == "level":
                if op not in ("=", "!=", "in"):
                    raise ValueError("level takes =, != or in")
                cond = {"$in": [LEVEL_EXPR, in_values(name, value) if op == "in" else [value]]}
                match.append({"$expr": cond if op != "!=" else {"$not": [cond]}})
                continue
            path = _path(name)
            if op == "in":
                match.append({path: {"$in": [_value(name, v) for v in in_values(name, value)]}})
            elif op in ("contains", "not contains"):
                rx = {"$regex": re.escape(value), "$options": "i"}
                match.append({path: rx if op == "contains" else {"$not": rx}})
            elif op in (">", "<"):
                try:
                    number = float(value)
                except ValueError:
                    raise ValueError(f"{name} {op} needs a number, got {value!r}") from None
                if not math.isfinite(number):
                    raise ValueError("numeric filters must use finite numbers")
                cmp = "$gt" if op == ">" else "$lt"
                match.append(
                    {"$expr": {"$and": [{"$ne": [_num(path), None]}, {cmp: [_num(path), number]}]}}
                )
            else:
                v = _value(name, value)
                match.append({path: v if op == "=" else {"$ne": v}})
        return match

    def _match(self, filters: list[dict], start: datetime, end: datetime) -> dict:
        rng = {"start": start.astimezone(UTC).isoformat(), "end": end.astimezone(UTC).isoformat()}
        return {"$match": {"$and": self.time_match(rng) + self.filter_match(filters)}}

    def build(self, spec: dict) -> str:
        match = self.time_match(spec) + self.filter_match(spec.get("filters"))
        fn, target, label = measure_of(spec)
        after: dict = {}  # a $set after the $group, turning the accumulator's output into the value
        if fn == "count":
            acc: dict = {"$sum": 1}
        else:
            path = _path(target or "")
            # shortcut: $addToSet holds every distinct value in memory per group
            if fn == "count_distinct":
                acc, after = {"$addToSet": "$" + path}, {label: {"$size": "$" + label}}
            elif fn == "p95":
                acc = {"$percentile": {"input": _num(path), "p": [0.95], "method": "approximate"}}
                after = {label: {"$arrayElemAt": ["$" + label, 0]}}
            else:
                acc = {"$" + fn: _num(path)}
        ids, cols = {}, []
        if spec.get("over_time"):
            ms = step_for(span_minutes(spec)) * 1000
            ids["time"] = {
                "$toDate": {
                    "$subtract": [
                        {"$toLong": "$timestamp"},
                        {"$mod": [{"$toLong": "$timestamp"}, ms]},
                    ]
                }
            }
            cols.append("time")
        for i, g in enumerate((spec.get("group_by") or [])[:2]):
            ids[alias(str(g), i)] = LEVEL_EXPR if g == "level" else "$" + _path(str(g))
            cols.append(alias(str(g), i))
        pipeline: list[dict] = [
            {"$match": {"$and": match}},
            {"$group": {"_id": ids or None, label: acc}},
        ]
        if after:
            pipeline.append({"$set": after})
        # all computed: keeps column order
        project = {"_id": 0} | {c: "$_id." + c for c in cols} | {label: "$" + label}
        pipeline += [
            {"$project": project},
            {"$sort": {"time": 1} if spec.get("over_time") else {label: -1}},
            {
                "$limit": max(
                    1, min(int(spec.get("limit") or 100), get_settings().server.max_builder_rows)
                )
            },
        ]
        return _dump(pipeline)

    def schema(self, fields: list[str]) -> dict:
        cols = [{"name": "timestamp", "about": "when the line was logged"}] + [
            {"name": PATHS.get(k, "level (computed)"), "builder": k, "about": about}
            for k, about in FIELDS.items()
        ]
        cols.append(
            {"name": "payload", "about": "fields extracted by the line's template (a sub-document)"}
        )
        return {
            "engine": self.engine,
            "table": "logs",
            "columns": cols,
            "fields": sorted(fields),
            "payload_access": "payload.name",
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
        return _dump(
            [
                self._match(filters, start, end),
                {"$sort": {"timestamp": -1 if newest_first else 1}},
                {"$limit": limit},
                {
                    "$project": {
                        "_id": 0,
                        "log_id": "$_id",
                        "timestamp": 1,
                        "host": "$device_id",
                        "service": 1,
                        "source": 1,
                        "template_id": 1,
                        "level": LEVEL_EXPR,
                        "message": "$raw_message",
                        **({"payload": 1} if payload else {}),
                    }
                },
            ]
        )

    def template_counts(self, filters: list[dict], start: datetime, end: datetime) -> str:
        return _dump(
            [
                self._match(filters, start, end),
                {"$sort": {"timestamp": 1}},
                {
                    "$group": {
                        "_id": {"t": "$template_id", "s": "$service"},
                        "n": {"$sum": 1},
                        "first": {"$first": "$timestamp"},
                        "last": {"$last": "$timestamp"},
                        "example": {"$first": "$raw_message"},
                        "log_id": {"$first": "$_id"},
                        "host": {"$first": "$device_id"},
                        "tags": {"$first": "$tags"},
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "template_id": "$_id.t",
                        "service": "$_id.s",
                        "n": 1,
                        "first": 1,
                        "last": 1,
                        "example": 1,
                        "log_id": 1,
                        "host": 1,
                        "tags": 1,
                    }
                },
                {"$sort": {"first": 1}},
                {"$limit": get_settings().ask.agg_cap},
            ]
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
        idx = {
            "$floor": {
                "$divide": [{"$subtract": [{"$toLong": "$timestamp"}, origin * 1000]}, step * 1000]
            }
        }
        return _dump(
            [
                self._match(filters, start, end),
                {"$group": {"_id": {"k": f"${by}", "b": idx}, "n": {"$sum": 1}}},
                {"$project": {"_id": 0, "key": "$_id.k", "bucket": "$_id.b", "n": 1}},
                {"$sort": {"bucket": 1}},
                {"$limit": get_settings().ask.agg_cap},
            ]
        )

    def template_sample(self, template_id: int, keys: list[str]) -> str:
        return _dump(
            [
                {"$match": {"template_id": int(template_id)}},
                {"$sort": {"timestamp": -1}},
                {"$limit": 50},
                {
                    "$project": {
                        "_id": 0,
                        "timestamp": 1,
                        "host": "$device_id",
                        "service": 1,
                        **{k: f"$payload.{k}" for k in keys},
                    }
                },
            ]
        )


class MongoReads(StoreReads):
    """``level`` is not stored: the UI reads that filter or group by it compute it in Python
    (streamed, three fields only); compiled queries use LEVEL_EXPR."""

    _LEVEL_FIELDS: ClassVar[dict[str, int]] = {"timestamp": 1, "raw_message": 1, "tags": 1}

    def __init__(self, store: MongoStore):
        self.store = store
        self.logs = store.logs
        self.alerts_col = store.alerts
        self.compiler = MongoCompiler()
        ro = get_settings().mongo.readonly_uri
        self.query_logs = MongoClient(ro, tz_aware=True)[store.db.name]["logs"] if ro else self.logs

    @staticmethod
    def _where(filters: dict, text: str, minutes: int | None) -> dict:
        """A find() filter for everything except ``level`` (computed in Python)."""
        where: dict = {}
        for k, v in filters.items():
            if k == "level":
                continue
            if k == "template_id":  # a non-number matches nothing, like the other stores
                where[PATHS[k]] = int(v) if str(v).lstrip("-").isdigit() else None
            else:
                where[PATHS[k]] = str(v)
        if text:
            where["raw_message"] = {"$regex": re.escape(text), "$options": "i"}
        if minutes:
            where["timestamp"] = {"$gte": datetime.now(UTC) - timedelta(minutes=minutes)}
        return where

    def recent(
        self, filters: dict, limit: int, text: str = "", minutes: int | None = None
    ) -> list[dict]:
        cur = self.logs.find(self._where(filters, text, minutes)).sort("timestamp", DESCENDING)
        if "level" not in filters:
            return [row_json(_row(d)) for d in cur.limit(limit)]
        rows = (row_json(_row(d)) for d in cur)  # stops reading once `limit` rows matched
        return list(islice((j for j in rows if j["level"] == str(filters["level"])), limit))

    def between(
        self,
        start: datetime,
        end: datetime,
        limit: int,
        host: str | None = None,
        service: str | None = None,
    ) -> list[dict]:
        where: dict = {"timestamp": {"$gte": start, "$lte": end}}
        if host is not None:
            where["device_id"] = host
        if service is not None:
            where["service"] = service
        return [
            row_json(_row(d))
            for d in self.logs.find(where).sort("timestamp", ASCENDING).limit(limit)
        ]

    def _level_docs(self, filters: dict, text: str, minutes: int | None):
        """(document, level) for the Python path, level filter applied."""
        want = str(filters["level"]) if "level" in filters else None
        for d in self.logs.find(
            self._where(filters, text, minutes),
            {**self._LEVEL_FIELDS, **{f: 1 for f in PATHS.values()}},
        ):
            lv = level(d["raw_message"], d["tags"])
            if want is None or lv == want:
                yield d, lv

    def histogram(self, minutes: int, by: str, filters: dict | None = None, text: str = "") -> dict:
        filters = filters or {}
        step = step_for(minutes)
        counts: dict[int, Counter] = {}
        if by != "level" and "level" not in filters:
            ms = {"$toLong": "$timestamp"}
            pipeline = [
                {"$match": self._where(filters, text, minutes)},
                {
                    "$group": {
                        "_id": {
                            "t": {"$subtract": [ms, {"$mod": [ms, step * 1000]}]},
                            "k": "$status",
                        },
                        "n": {"$sum": 1},
                    }
                },
            ]
            for g in self.logs.aggregate(pipeline):
                counts.setdefault(g["_id"]["t"] // 1000, Counter())[g["_id"]["k"]] = g["n"]
        else:
            for d, lv in self._level_docs(filters, text, minutes):
                t = int(d["timestamp"].timestamp()) // step * step
                counts.setdefault(t, Counter())[lv if by == "level" else d["status"]] += 1
        return {"step": step, "points": zero_fill(minutes, step, counts, histogram_keys(by))}

    def top(
        self, field: str, minutes: int, limit: int, filters: dict | None = None, text: str = ""
    ) -> list[tuple]:
        filters = filters or {}
        if field != "level" and "level" not in filters:
            pipeline = [
                {"$match": self._where(filters, text, minutes)},
                {"$group": {"_id": f"${PATHS[field]}", "n": {"$sum": 1}}},
                {"$sort": {"n": -1}},
                {"$limit": limit},
            ]
            return [(g["_id"], g["n"]) for g in self.logs.aggregate(pipeline)]
        return Counter(
            lv if field == "level" else d[PATHS[field]]
            for d, lv in self._level_docs(filters, text, minutes)
        ).most_common(limit)

    def _source(self, host: str, service: str, source: str, limit: int, fields: dict) -> list[dict]:
        cur = self.logs.find({"device_id": host, "service": service, "source": source}, fields)
        return list(cur.sort("timestamp", DESCENDING).limit(limit))[::-1]  # oldest first

    def source_lines(self, host: str, service: str, source: str, limit: int) -> list[str]:
        return [
            d["raw_message"] for d in self._source(host, service, source, limit, {"raw_message": 1})
        ]

    def source_rows(self, host: str, service: str, source: str, limit: int) -> list[tuple]:
        return [
            (d["timestamp"], d["template_id"])
            for d in self._source(host, service, source, limit, {"timestamp": 1, "template_id": 1})
        ]

    def alerts(self, limit: int) -> list[dict]:
        cur = self.alerts_col.find().sort("window_end", DESCENDING).limit(limit)
        return [alert_json({**d, "alert_id": d["_id"]}) for d in cur]

    def query(self, text: str, max_rows: int | None = None) -> dict:
        """Run a JSON aggregation pipeline over ``logs``. Extended JSON (``{"$date": "..."}``) is
        read, so time ranges can use the timestamp index."""
        try:
            pipeline = json_util.loads(text)
        except ValueError:
            raise ValueError(
                "raw SQL is not supported on MongoDB; send a JSON aggregation pipeline"
            ) from None
        try:
            docs = list(
                islice(
                    self.query_logs.aggregate(check(pipeline), maxTimeMS=10_000),
                    max_rows or get_settings().server.max_query_rows,
                )
            )
        except PyMongoError as error:
            raise ValueError(str(error).splitlines()[0]) from error
        columns = list(dict.fromkeys(k for d in docs for k in d))
        return {
            "columns": columns,
            "rows": [[d.get(c) for c in columns] for d in docs],
            "engine": "mongodb",
            "table": "logs",
        }
