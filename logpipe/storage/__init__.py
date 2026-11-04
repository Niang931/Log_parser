from .base import AnomalyAlert, LogRow, LogStore
from .memory import MemoryStore

__all__ = ["AnomalyAlert", "LogRow", "LogStore", "MemoryStore", "open_store"]


def open_store(kind: str) -> LogStore:
    """``memory`` or ``clickhouse`` (connection settings come from the environment)."""
    if kind == "memory":
        return MemoryStore()
    if kind == "clickhouse":
        from .clickhouse import ClickHouseStore

        return ClickHouseStore.from_env()
    # TODO: add mongo later
    # if kind == "mongo":
    #     from .mongo import MongoStore
    #     return MongoStore.from_env()
    raise ValueError(f"unknown store {kind!r}; expected 'memory' or 'clickhouse'")
