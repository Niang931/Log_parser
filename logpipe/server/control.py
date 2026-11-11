"""The server's control plane: agents, API keys, remote configs, streams, warnings, jobs and
settings, kept as one JSON document per ``(kind, id)``.

The in-memory copy is the source of truth while the server runs; every write also goes to
ClickHouse's ``control_docs`` (newest version wins) so a restart picks up where it left off.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Protocol


class DocSink(Protocol):
    def load(self) -> list[tuple[str, str, dict]]: ...

    def write(self, kind: str, doc_id: str, doc: dict | None) -> None: ...


class ControlStore:
    def __init__(self, sink: DocSink | None = None):
        self.sink = sink
        self._lock = threading.Lock()
        self._docs: dict[str, dict[str, dict]] = {}
        for kind, doc_id, doc in sink.load() if sink else ():
            self._docs.setdefault(kind, {})[doc_id] = doc

    def get(self, kind: str, doc_id: str) -> dict | None:
        with self._lock:
            doc = self._docs.get(kind, {}).get(doc_id)
            return dict(doc) if doc is not None else None

    def all(self, kind: str) -> list[dict]:
        with self._lock:
            return [dict(d) for d in self._docs.get(kind, {}).values()]

    def put(self, kind: str, doc_id: str, doc: dict) -> dict:
        with self._lock:
            self._docs.setdefault(kind, {})[doc_id] = dict(doc)
        if self.sink:
            self.sink.write(kind, doc_id, doc)
        return doc

    def update(self, kind: str, doc_id: str, **fields) -> dict:
        with self._lock:
            doc = {**self._docs.get(kind, {}).get(doc_id, {}), **fields}
            self._docs.setdefault(kind, {})[doc_id] = doc
        if self.sink:
            self.sink.write(kind, doc_id, doc)
        return dict(doc)

    def delete(self, kind: str, doc_id: str) -> None:
        with self._lock:
            self._docs.get(kind, {}).pop(doc_id, None)
        if self.sink:
            self.sink.write(kind, doc_id, None)


class ClickHouseDocs:
    """``control_docs`` rows; ``version`` is a nanosecond clock so the latest write wins."""

    def __init__(self, client):
        self.client = client

    def load(self) -> list[tuple[str, str, dict]]:
        rows = self.client.query(
            "SELECT kind, id, doc FROM control_docs FINAL WHERE deleted = 0"
        ).result_rows
        return [(kind, doc_id, json.loads(doc)) for kind, doc_id, doc in rows]

    def write(self, kind: str, doc_id: str, doc: dict | None) -> None:
        self.client.insert(
            "control_docs",
            [[kind, doc_id, json.dumps(doc or {}, default=str), time.time_ns(), int(doc is None)]],
            column_names=["kind", "id", "doc", "version", "deleted"],
        )
