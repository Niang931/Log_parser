"""MongoDB backend: one plain collection, ``_id = log_id``.

Time-series collections reject upsert and unique indexes, so they cannot turn a PENDING row
into its FINAL version. A plain collection with a version-guarded replace gives the same
"highest version wins, then last insert" rule as ReplacingMergeTree, with no ``FINAL`` on reads.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence

from pymongo import ASCENDING, MongoClient, ReplaceOne
from pymongo.database import Database
from pymongo.errors import BulkWriteError

from ..parsing.frozen_parser import Status
from ..settings import get_settings
from .base import AnomalyAlert, LogRow

DUPLICATE_KEY = 11000


def _doc(r: LogRow) -> dict:
    return {
        "_id": str(r.log_id), 
        "timestamp": r.timestamp, 
        "device_id": r.device_id,
        "template_id": r.template_id, 
        "status": str(r.status),
        "registry_version": r.registry_version, 
        "version": r.version, 
        "payload": r.payload,
        "raw_message": r.raw_message, 
        "service": r.service, 
        "source": r.source,
        "tags": list(r.tags),
    }


def _row(d: dict) -> LogRow:
    return LogRow(
        log_id=uuid.UUID(d["_id"]), 
        timestamp=d["timestamp"], 
        device_id=d["device_id"],
        template_id=d["template_id"], 
        status=Status(d["status"]),
        registry_version=d["registry_version"], 
        payload=d["payload"],
        raw_message=d["raw_message"], 
        version=d["version"], service=d["service"],
        source=d["source"], tags=tuple(d["tags"]),
    )


class MongoStore:
    def __init__(self, db: Database, *, retention_days: int | None = None) -> None:
        self.db = db
        self.logs = db["logs"]
        self.alerts = db["alerts"]
        self.logs.create_index([("status", ASCENDING), ("timestamp", ASCENDING)])
        # UI reads: time-range scans, and the per-source tail. (timestamp, device_id) rather than
        # `timestamp` alone, because the TTL index below owns the plain `timestamp` key.
        self.logs.create_index([("timestamp", ASCENDING), ("device_id", ASCENDING)])
        self.logs.create_index([("device_id", ASCENDING), ("service", ASCENDING),
                                ("source", ASCENDING), ("timestamp", ASCENDING)])
        if retention_days:
            self.logs.create_index("timestamp", expireAfterSeconds=retention_days * 86_400)

    @classmethod
    def from_env(cls, database: str | None = None, **kwargs) -> MongoStore:
        """``MONGO_URI`` / ``MONGO_DB`` settings; ``database`` overrides ``MONGO_DB``."""
        cfg = get_settings().mongo
        client = MongoClient(cfg.uri, tz_aware=True)
        return cls(client[database or cfg.db], **kwargs)

    def insert_logs(self, rows: Sequence[LogRow]) -> None:
        if not rows:
            return
        ops = [ReplaceOne({"_id": str(r.log_id), "version": {"$lte": r.version}}, _doc(r), upsert=True)
               for r in rows]
        try:
            self.logs.bulk_write(ops, ordered=False)
        except BulkWriteError as e:
            # Duplicate key = the stored row has a higher version, so this write loses.
            if any(err["code"] != DUPLICATE_KEY for err in e.details["writeErrors"]):
                raise

    def insert_alerts(self, alerts: Sequence[AnomalyAlert]) -> None:
        if not alerts:
            return
        self.alerts.insert_many([
            {"_id": str(a.alert_id), "device_id": a.device_id, "window_start": a.window_start,
             "window_end": a.window_end, "template_ids": list(a.template_ids),
             "miss_ratio": a.miss_ratio, "novelty_ratio": a.novelty_ratio, "distance": a.distance,
             "reasons": list(a.reasons), "had_unknowns": a.had_unknowns,
             "registry_version": a.registry_version, "model_version": a.model_version}
            for a in alerts
        ])

    def fetch_pending(self, limit: int = 10_000, template_id: int | None = None) -> list[LogRow]:
        """PENDING rows, or with ``template_id`` that template's rows (to re-extract them)."""
        where = {"status": str(Status.PENDING)} if template_id is None else {"template_id": template_id}
        return [_row(d) for d in self.logs.find(where).sort("timestamp", ASCENDING).limit(limit)]

    def count_by_status(self) -> dict[str, int]:
        return {d["_id"]: d["n"] for d in self.logs.aggregate(
            [{"$group": {"_id": "$status", "n": {"$sum": 1}}}])}

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.db.client.close()  # not a `client` attribute: Server.store_usage treats that as ClickHouse


class MongoDocs:
    """``control_docs``: one document per ``(kind, id)``; a delete really deletes."""

    def __init__(self, db: Database) -> None:
        self.docs = db["control_docs"]

    def load(self) -> list[tuple[str, str, dict]]:
        return [(d["_id"]["k"], d["_id"]["i"], d["doc"]) for d in self.docs.find()]

    def write(self, kind: str, doc_id: str, doc: dict | None) -> None:
        key = {"k": kind, "i": doc_id}
        if doc is None:
            self.docs.delete_one({"_id": key})
        else:
            self.docs.replace_one({"_id": key}, {"_id": key, "doc": doc}, upsert=True)
