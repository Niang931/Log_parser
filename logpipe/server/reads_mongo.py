"""UI reads on MongoDB. One document per log_id, so no FINAL is needed.

``level`` is not stored; it is computed by ``reads.level`` in Python, only for the calls that filter
or group by it (streamed, 3 fields only).
# ponytail: level scans the time window; if the bench finds it slow, generate a $switch from LEVEL_RULES.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from itertools import islice

from pymongo import ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

from ..storage.mongo import MongoStore, _row
from .querybuild_mongo import check
from .reads import MAX_QUERY_ROWS, _keys, alert_json, level, row_json, step_for, zero_fill

FIELDS = {"host": "device_id", "service": "service", "source": "source",
          "status": "status", "template_id": "template_id"}
LEVEL_FIELDS = {"timestamp": 1, "raw_message": 1, "tags": 1}


class MongoReads:
    def __init__(self, store: MongoStore):
        self.store = store
        self.logs = store.logs
        self.alerts_col = store.alerts

    @staticmethod
    def _where(filters: dict, text: str, minutes: int | None) -> dict:
        """Mongo filter for everything except ``level`` (computed in Python)."""
        where: dict = {}
        for k, v in filters.items():
            if k == "level":
                continue
            if k == "template_id":  # a non-number matches nothing, like the other stores
                v = int(v) if str(v).lstrip("-").isdigit() else None
            where[FIELDS[k]] = v if k == "template_id" else str(v)
        if text:
            where["raw_message"] = {"$regex": re.escape(text), "$options": "i"}
        if minutes:
            where["timestamp"] = {"$gte": datetime.now(UTC) - timedelta(minutes=minutes)}
        return where

    def recent(self, filters: dict, limit: int, text: str = "", minutes: int | None = None) -> list[dict]:
        cur = self.logs.find(self._where(filters, text, minutes)).sort("timestamp", DESCENDING)
        if "level" not in filters:
            return [row_json(_row(d)) for d in cur.limit(limit)]
        rows = (row_json(_row(d)) for d in cur)  # stops reading once `limit` rows matched
        return list(islice((j for j in rows if j["level"] == str(filters["level"])), limit))

    def between(self, start: datetime, end: datetime, limit: int, host: str | None = None,
                service: str | None = None) -> list[dict]:
        where: dict = {"timestamp": {"$gte": start, "$lte": end}}
        if host is not None:
            where["device_id"] = host
        if service is not None:
            where["service"] = service
        return [row_json(_row(d)) for d in self.logs.find(where).sort("timestamp", ASCENDING).limit(limit)]

    def _level_docs(self, filters: dict, text: str, minutes: int | None):
        """(timestamp, level, status/field source doc) for the Python path, level filter applied."""
        want = str(filters["level"]) if "level" in filters else None
        for d in self.logs.find(self._where(filters, text, minutes), {**LEVEL_FIELDS, **{f: 1 for f in FIELDS.values()}}):
            lv = level(d["raw_message"], d["tags"])
            if want is None or lv == want:
                yield d, lv

    def histogram(self, minutes: int, by: str, filters: dict | None = None, text: str = "") -> dict:
        filters = filters or {}
        step = step_for(minutes)
        counts: dict[int, Counter] = {}
        if by != "level" and "level" not in filters:
            ms = {"$toLong": "$timestamp"}
            pipeline = [{"$match": self._where(filters, text, minutes)},
                        {"$group": {"_id": {"t": {"$subtract": [ms, {"$mod": [ms, step * 1000]}]}, "k": "$status"},
                                    "n": {"$sum": 1}}}]
            for g in self.logs.aggregate(pipeline):
                counts.setdefault(g["_id"]["t"] // 1000, Counter())[g["_id"]["k"]] = g["n"]
        else:
            for d, lv in self._level_docs(filters, text, minutes):
                t = int(d["timestamp"].timestamp()) // step * step
                counts.setdefault(t, Counter())[lv if by == "level" else d["status"]] += 1
        return {"step": step, "points": zero_fill(minutes, step, counts, _keys(by))}

    def top(self, field: str, minutes: int, limit: int, filters: dict | None = None,
            text: str = "") -> list[tuple]:
        filters = filters or {}
        if field != "level" and "level" not in filters:
            pipeline = [{"$match": self._where(filters, text, minutes)},
                        {"$group": {"_id": f"${FIELDS[field]}", "n": {"$sum": 1}}},
                        {"$sort": {"n": -1}}, {"$limit": limit}]
            return [(g["_id"], g["n"]) for g in self.logs.aggregate(pipeline)]
        return Counter(lv if field == "level" else d[FIELDS[field]]
                       for d, lv in self._level_docs(filters, text, minutes)).most_common(limit)

    def _source(self, host: str, service: str, source: str, limit: int, fields: dict) -> list[dict]:
        cur = self.logs.find({"device_id": host, "service": service, "source": source}, fields)
        return list(cur.sort("timestamp", DESCENDING).limit(limit))[::-1]  # oldest first

    def source_lines(self, host: str, service: str, source: str, limit: int) -> list[str]:
        return [d["raw_message"] for d in self._source(host, service, source, limit, {"raw_message": 1})]

    def source_rows(self, host: str, service: str, source: str, limit: int) -> list[tuple]:
        return [(d["timestamp"], d["template_id"])
                for d in self._source(host, service, source, limit, {"timestamp": 1, "template_id": 1})]

    def alerts(self, limit: int) -> list[dict]:
        cur = self.alerts_col.find().sort("window_end", DESCENDING).limit(limit)
        return [alert_json({**d, "alert_id": d["_id"]}) for d in cur]

    def query(self, text: str) -> dict:
        """Run a JSON aggregation pipeline over ``logs`` (the Query page's text box)."""
        try:
            pipeline = json.loads(text)
        except ValueError:
            raise ValueError("raw SQL is not supported on MongoDB; send a JSON aggregation pipeline") from None
        try:
            docs = list(islice(self.logs.aggregate(check(pipeline), maxTimeMS=10_000), MAX_QUERY_ROWS))
        except PyMongoError as error:
            raise ValueError(str(error).splitlines()[0]) from error
        columns = list(dict.fromkeys(k for d in docs for k in d))
        return {"columns": columns, "rows": [[d.get(c) for c in columns] for d in docs],
                "engine": "mongodb", "table": "logs"}
