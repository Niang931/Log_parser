"""The Query page on MongoDB: the same builder spec as ``querybuild``, turned into an aggregation
pipeline (JSON text), plus the guard every pipeline passes before it runs.

The UI's query path is read-only: only stages in ``STAGES`` are allowed, and the operators that run
code or write (``$where``, ``$function``, ``$accumulator``, ``$out``, ``$merge``) are refused at any depth.
"""
from __future__ import annotations

import json
import re

from ..enrich.validate import KEY_RE
from .querybuild import COLUMNS, MAX_LIMIT, MEASURES, OPS, absolute_range
from .reads import HEAD, LEVEL_RULES, LEVELS, step_for

FIELDS = {"host": "device_id", "service": "service", "source": "source", "status": "status",
          "template_id": "template_id", "message": "raw_message"}  # `level` is computed: LEVEL_EXPR
STAGES = {"$match", "$group", "$project", "$sort", "$limit", "$skip", "$unwind", "$addFields", "$set",
          "$unset", "$count", "$sortByCount", "$replaceRoot", "$replaceWith", "$bucket", "$bucketAuto"}
BANNED = {"$where", "$function", "$accumulator", "$out", "$merge"}
# reads.level() as an aggregation expression: a stated severity tag, then LEVEL_RULES in order.
# The rules are written in the regex subset Python, RE2 and PCRE share, so all stores agree.
LEVEL_EXPR = {"$switch": {"branches": [
    *({"case": {"$in": [f"severity:{lvl}", {"$ifNull": ["$tags", []]}]}, "then": lvl} for lvl in LEVELS),
    *({"case": {"$regexMatch": {"input": {"$substrCP": ["$raw_message", 0, HEAD]} if head else "$raw_message",
                                "regex": rx}}, "then": lvl} for lvl, rx, head in LEVEL_RULES)],
    "default": "info"}}


def check(pipeline) -> list[dict]:
    """The pipeline if it is read-only; ValueError otherwise."""
    if not isinstance(pipeline, list) or not pipeline or not all(isinstance(s, dict) and len(s) == 1 for s in pipeline):
        raise ValueError("a pipeline is a JSON list of single-stage objects, e.g. [{\"$match\": {...}}]")
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


def _field(name: str) -> str:
    if name.startswith("@"):
        if not KEY_RE.match(name[1:]):
            raise ValueError(f"field {name!r}: extracted field names are lower_snake_case")
        return "payload." + name[1:]
    if name not in FIELDS:
        raise ValueError(f"unknown field {name!r}; use one of {', '.join(FIELDS)} or @extracted_field")
    return FIELDS[name]


def _num(path: str) -> dict:
    return {"$convert": {"input": "$" + path, "to": "double", "onError": None, "onNull": None}}


def _value(field: str, value: str):
    if field == "template_id":
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"template_id needs a number, got {value!r}") from None
    return value


def mongo_date(t) -> dict:
    """Extended JSON, read back as a date by ``MongoReads.query``."""
    return {"$date": t.isoformat().replace("+00:00", "Z")}


def time_match(spec: dict) -> list[dict]:
    if rng := absolute_range(spec):
        return [{"timestamp": {"$gte": mongo_date(rng[0]), "$lte": mongo_date(rng[1])}}]
    minutes = int(spec.get("minutes") or 60)
    if not 1 <= minutes <= 60 * 24 * 90:
        raise ValueError("time range must be between 1 minute and 90 days")
    since = {"$dateSubtract": {"startDate": "$$NOW", "unit": "minute", "amount": minutes}}
    return [{"$expr": {"$gte": ["$timestamp", since]}}]


def filter_match(filters) -> list[dict]:
    match: list[dict] = []
    for f in filters or []:
        op, value = f.get("op", "="), str(f.get("value", ""))
        if op not in OPS:
            raise ValueError(f"operator must be one of {', '.join(OPS)}")
        name = str(f.get("field", ""))
        if name == "level":
            if op not in ("=", "!=", "in"):
                raise ValueError("level takes =, != or in")
            values = [v.strip() for v in value.split(",") if v.strip()] if op == "in" else [value]
            cond = {"$in": [LEVEL_EXPR, values]}
            match.append({"$expr": cond if op != "!=" else {"$not": [cond]}})
            continue
        path = _field(name)
        if op == "in":
            values = [_value(name, v.strip()) for v in value.split(",") if v.strip()]
            if not values:
                raise ValueError(f"{name} in needs at least one value")
            match.append({path: {"$in": values}})
        elif op in ("contains", "not contains"):
            rx = {"$regex": re.escape(value), "$options": "i"}
            match.append({path: rx if op == "contains" else {"$not": rx}})
        elif op in (">", "<"):
            try:
                number = float(value)
            except ValueError:
                raise ValueError(f"{name} {op} needs a number, got {value!r}") from None
            cmp = "$gt" if op == ">" else "$lt"
            match.append({"$expr": {"$and": [{"$ne": [_num(path), None]}, {cmp: [_num(path), number]}]}})
        else:
            v = _value(name, value)
            match.append({path: v if op == "=" else {"$ne": v}})
    return match


def build(spec: dict, engine: str = "mongodb") -> str:
    match = time_match(spec) + filter_match(spec.get("filters"))
    rng = absolute_range(spec)
    minutes = int((rng[1] - rng[0]).total_seconds() // 60) or 1 if rng else int(spec.get("minutes") or 60)

    measure = spec.get("measure") or {"fn": "count"}
    fn = measure.get("fn", "count")
    if fn not in MEASURES:
        raise ValueError(f"measure must be one of {', '.join(MEASURES)}")
    after: dict = {}  # a $set run after the $group, to turn the accumulator's output into the value
    if fn == "count":
        acc, label = {"$sum": 1}, "lines"
    else:
        target = str(measure.get("field") or "")
        if not target:
            raise ValueError(f"{fn} needs a field")
        path, label = _field(target), f"{fn}_{target.lstrip('@')}"
        if fn == "count_distinct":  # ponytail: $addToSet holds every distinct value in memory per group
            acc, after = {"$addToSet": "$" + path}, {label: {"$size": "$" + label}}
        elif fn == "p95":
            acc = {"$percentile": {"input": _num(path), "p": [0.95], "method": "approximate"}}
            after = {label: {"$arrayElemAt": ["$" + label, 0]}}
        else:
            acc = {"$" + fn: _num(path)}

    ids, cols = {}, []
    if spec.get("over_time"):
        ms = step_for(minutes) * 1000
        ids["time"] = {"$toDate": {"$subtract": [{"$toLong": "$timestamp"}, {"$mod": [{"$toLong": "$timestamp"}, ms]}]}}
        cols.append("time")
    for i, g in enumerate((spec.get("group_by") or [])[:2]):
        alias = str(g).lstrip("@") if KEY_RE.match(str(g).lstrip("@")) else f"g{i}"
        ids[alias] = LEVEL_EXPR if g == "level" else "$" + _field(str(g))
        cols.append(alias)

    pipeline: list[dict] = [{"$match": {"$and": match}}, {"$group": {"_id": ids or None, label: acc}}]
    if after:
        pipeline.append({"$set": after})
    project = {"_id": 0} | {c: "$_id." + c for c in cols} | {label: "$" + label}  # all computed: keeps column order
    pipeline += [{"$project": project},
                 {"$sort": {"time": 1} if spec.get("over_time") else {label: -1}},
                 {"$limit": min(int(spec.get("limit") or 100), MAX_LIMIT)}]
    return json.dumps(pipeline, indent=1)


def schema(engine: str, fields: list[str]) -> dict:
    """What the Query page's sidebar lists."""
    cols = [{"name": "timestamp", "about": "when the line was logged"}] + [
        {"name": FIELDS.get(k, "level (computed)"), "builder": k, "about": about}
        for k, (_, _, about) in COLUMNS.items() if k in FIELDS or k == "level"]
    cols.append({"name": "payload", "about": "fields extracted by the line's template (a sub-document)"})
    return {"engine": engine, "table": "logs", "columns": cols, "fields": sorted(fields),
            "payload_access": "payload.name", "measures": list(MEASURES), "ops": list(OPS)}
