"""Wire format of an unresolved row: one JSON object per queue message.

The payload carries the whole :class:`~logpipe.storage.base.LogRow` so a
learning worker in another process can backfill the row without reading the
store first. ``log_id`` is the idempotency key: a resolved row inserted twice
(after a redelivery) collapses to one in the ReplacingMergeTree.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime

from ..enrich.validate import generalize_digits
from ..parsing.frozen_parser import Status
from ..storage.base import LogRow


def encode_row(row: LogRow) -> bytes:
    return json.dumps(
        {
            "log_id": str(row.log_id),
            "timestamp": row.timestamp.isoformat(),
            "device_id": row.device_id,
            "template_id": row.template_id,
            "status": str(row.status),
            "registry_version": row.registry_version,
            "payload": row.payload,
            "raw_message": row.raw_message,
            "version": row.version,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_row(data: bytes | str) -> LogRow:
    obj = json.loads(data)
    return LogRow(
        log_id=uuid.UUID(obj["log_id"]),
        timestamp=datetime.fromisoformat(obj["timestamp"]),
        device_id=obj["device_id"],
        template_id=int(obj["template_id"]),
        status=Status(obj["status"]),
        registry_version=int(obj["registry_version"]),
        payload=dict(obj.get("payload") or {}),
        raw_message=obj["raw_message"],
        version=int(obj.get("version", 0)),
    )


def shape_key(line: str) -> bytes:
    """Partition key from the line's *shape*: tokens with digits collapsed.
    Lines of same not-yet-known template will land in same partition -> 1 worker see whole cluster and learns the template from a full sample.
    """
    digest = hashlib.blake2b(" ".join(generalize_digits(line.split())).encode(), digest_size=8)
    return digest.hexdigest().encode("ascii")


def partition_key(row: LogRow, key_by: str = "shape") -> bytes:
    """``shape`` (default) groups by candidate template, ``device`` by device id."""
    if key_by == "device":
        return row.device_id.encode("utf-8")
    
    if key_by == "shape":
        return shape_key(row.raw_message)
    
    raise ValueError(f"unknown key_by {key_by!r}; expected 'shape' or 'device'")
