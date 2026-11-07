"""The Query page's visual builder: a small spec (time range, filters, a measure, group-by)
turned into read-only SQL for ClickHouse (``iot_logs``) or the memory store's SQLite (``logs``).

Field names are checked against a fixed list or ``KEY_RE`` (extracted fields), and values are
quoted here, so a spec can't inject SQL; the result still runs through the read-only query path.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ..enrich.validate import KEY_RE
from .reads import LEVEL_SQL, step_for

COLUMNS = {  # builder name -> (clickhouse, sqlite, what it is)
    "host": ("device_id", "device_id", "host that shipped the line"),
    "service": ("service", "service", "service set on the agent source"),
    "source": ("source", "source", "source set on the agent source (e.g. python, nginx)"),
    "level": (LEVEL_SQL, "level", "error / warn / info / debug, read from the line"),
    "status": ("toString(log_status)", "status", "FINAL (parsed) or PENDING"),
    "template_id": ("template_id", "template_id", "template the line matched"),
    "message": ("raw_message", "raw_message", "the raw line"),
}
MEASURES = ("count", "count_distinct", "avg", "min", "max", "sum", "p95")
OPS = ("=", "!=", "contains", "not contains", ">", "<", "in")  # in: comma-separated values
MAX_LIMIT = 500


def _quote(value: str, engine: str) -> str:
    if engine == "clickhouse":
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    return "'" + value.replace("'", "''") + "'"


def _field(name: str, engine: str, numeric: bool = False) -> str:
    """A column, or ``@key`` for a field extracted by a template."""
    if name.startswith("@"):
        key = name[1:]
        if not KEY_RE.match(key):
            raise ValueError(f"field {name!r}: extracted field names are lower_snake_case")
        if engine == "clickhouse":
            raw = f"JSONExtractString(toJSONString(payload), {_quote(key, engine)})"
            return f"toFloat64OrNull({raw})" if numeric else raw
        raw = f"json_extract(payload, '$.{key}')"
        return f"CAST({raw} AS REAL)" if numeric else raw
    if name not in COLUMNS:
        raise ValueError(f"unknown field {name!r}; use one of {', '.join(COLUMNS)} or @extracted_field")
    return COLUMNS[name][0 if engine == "clickhouse" else 1]


def absolute_range(spec: dict) -> tuple[datetime, datetime] | None:
    """``start``/``end`` (ISO 8601) when the spec has them: an absolute range instead of the last
    ``minutes``."""
    if not (spec.get("start") or spec.get("end")):
        return None
    try:
        start, end = (datetime.fromisoformat(str(spec[k]).replace("Z", "+00:00")) for k in ("start", "end"))
    except (KeyError, ValueError):
        raise ValueError("start and end must both be ISO 8601 times") from None
    start, end = (t if t.tzinfo else t.replace(tzinfo=UTC) for t in (start, end))
    if not timedelta(0) < end - start <= timedelta(days=90):
        raise ValueError("time range must be between 1 minute and 90 days")
    return start.astimezone(UTC), end.astimezone(UTC)


def sql_time(t: datetime, engine: str) -> str:
    """A UTC instant as a literal ``timestamp`` (ClickHouse) or ``julianday(timestamp)`` compares against."""
    if engine == "clickhouse":
        return f"toDateTime64('{t.astimezone(UTC):%Y-%m-%d %H:%M:%S.%f}', 3, 'UTC')"
    return f"julianday('{t.astimezone(UTC):%Y-%m-%dT%H:%M:%S.%f}+00:00')"


def time_where(spec: dict, engine: str) -> list[str]:
    if rng := absolute_range(spec):
        col = "timestamp" if engine == "clickhouse" else "julianday(timestamp)"  # SQLite: ISO text with offsets
        return [f"{col} >= {sql_time(rng[0], engine)}", f"{col} <= {sql_time(rng[1], engine)}"]
    minutes = int(spec.get("minutes") or 60)
    if not 1 <= minutes <= 60 * 24 * 90:
        raise ValueError("time range must be between 1 minute and 90 days")
    return [f"timestamp >= now() - INTERVAL {minutes} MINUTE" if engine == "clickhouse"
            else f"timestamp >= strftime('%Y-%m-%dT%H:%M:%S', 'now', '-{minutes} minutes')"]


def filter_where(filters, engine: str) -> list[str]:
    where = []
    for f in filters or []:
        op, value = f.get("op", "="), str(f.get("value", ""))
        if op not in OPS:
            raise ValueError(f"operator must be one of {', '.join(OPS)}")
        numeric = op in (">", "<")
        name = str(f.get("field", ""))
        col = _field(name, engine, numeric)
        if op == "in":
            values = [v.strip() for v in value.split(",") if v.strip()]
            if not values:
                raise ValueError(f"{name} in needs at least one value")
            if name == "template_id":
                if not all(v.isdigit() for v in values):
                    raise ValueError("template_id in needs numbers")
                where.append(f"{col} IN ({', '.join(values)})")
            else:
                where.append(f"{col} IN ({', '.join(_quote(v, engine) for v in values)})")
        elif op in ("contains", "not contains"):
            expr = (f"positionCaseInsensitive({col}, {_quote(value, engine)}) > 0" if engine == "clickhouse"
                    else f"instr(lower({col}), lower({_quote(value, engine)})) > 0")
            where.append(("NOT " if op == "not contains" else "") + expr)
        elif numeric:
            try:
                number = float(value)
            except ValueError:
                raise ValueError(f"{f.get('field')} {op} needs a number, got {value!r}") from None
            where.append(f"{col} {op} {number}")
        else:
            where.append(f"{col} {'=' if op == '=' else '!='} {_quote(value, engine)}")
    return where


def build(spec: dict, engine: str) -> str:
    table = "iot_logs FINAL" if engine == "clickhouse" else "logs"
    where = time_where(spec, engine) + filter_where(spec.get("filters"), engine)
    rng = absolute_range(spec)
    minutes = int((rng[1] - rng[0]).total_seconds() // 60) or 1 if rng else int(spec.get("minutes") or 60)

    measure = spec.get("measure") or {"fn": "count"}
    fn = measure.get("fn", "count")
    if fn not in MEASURES:
        raise ValueError(f"measure must be one of {', '.join(MEASURES)}")
    if fn == "count":
        value_sql, label = "count()" if engine == "clickhouse" else "count(*)", "lines"
    else:
        target = str(measure.get("field") or "")
        if not target:
            raise ValueError(f"{fn} needs a field")
        col = _field(target, engine, numeric=fn not in ("count_distinct",))
        label = f"{fn}_{target.lstrip('@')}"
        if fn == "count_distinct":
            value_sql = f"uniqExact({col})" if engine == "clickhouse" else f"count(DISTINCT {col})"
        elif fn == "p95":
            if engine != "clickhouse":
                raise ValueError("p95 needs ClickHouse; the memory store's SQLite has no percentile")
            value_sql = f"quantile(0.95)({col})"
        else:
            value_sql = f"{fn}({col})"

    select, group = [], []
    if spec.get("over_time"):
        step = step_for(minutes)
        bucket = (f"toStartOfInterval(timestamp, INTERVAL {step} SECOND)" if engine == "clickhouse"
                  else f"datetime((CAST(strftime('%s', timestamp) AS INTEGER) / {step}) * {step}, 'unixepoch')")
        select.append(f"{bucket} AS time")
        group.append("time")
    for i, g in enumerate((spec.get("group_by") or [])[:2]):
        alias = str(g).lstrip("@") if KEY_RE.match(str(g).lstrip("@")) else f"g{i}"
        select.append(f"{_field(str(g), engine)} AS {alias}")
        group.append(alias)
    select.append(f"{value_sql} AS {label}")
    limit = min(int(spec.get("limit") or 100), MAX_LIMIT)
    sql = f"SELECT {', '.join(select)}\nFROM {table}\nWHERE " + "\n  AND ".join(where)
    if group:
        sql += f"\nGROUP BY {', '.join(group)}"
    sql += "\nORDER BY " + ("time" if spec.get("over_time") else f"{label} DESC") + f"\nLIMIT {limit}"
    return sql


def schema(engine: str, fields: list[str]) -> dict:
    """What the Query page's sidebar lists."""
    table = "iot_logs" if engine == "clickhouse" else "logs"
    def column(k: str, ch: str, lite: str) -> str:
        return lite if engine != "clickhouse" else "level (computed)" if k == "level" else ch
    cols = [{"name": "timestamp", "about": "when the line was logged"}] + [
        {"name": column(k, ch, lite), "builder": k, "about": about} for k, (ch, lite, about) in COLUMNS.items()]
    cols.append({"name": "payload", "about": "fields extracted by the line's template (JSON)"})
    payload = ("JSONExtractString(toJSONString(payload), 'name')" if engine == "clickhouse"
               else "json_extract(payload, '$.name')")
    return {"engine": engine, "table": table, "columns": cols, "fields": sorted(fields),
            "payload_access": payload, "measures": list(MEASURES), "ops": list(OPS)}
