"""Log store adapters: each backend's UI reads and query compiler behind one interface
(:class:`~.base.StoreReads`, :class:`~.base.Compiler`)."""

from __future__ import annotations

from .base import FACETS, Compiler, StoreReads, level


def reads_for(store) -> StoreReads:
    from ...storage.memory import MemoryStore

    if isinstance(store, MemoryStore):
        from .memory import MemoryReads

        return MemoryReads(store)
    if type(store).__name__ == "MongoStore":
        from .mongo import MongoReads

        return MongoReads(store)
    if hasattr(store, "client"):  # the ClickHouse store; MongoStore has none on purpose
        from .clickhouse import ClickHouseReads

        return ClickHouseReads(store)
    raise ValueError(f"no reads for store {type(store).__name__}")


__all__ = ["FACETS", "Compiler", "StoreReads", "level", "reads_for"]
