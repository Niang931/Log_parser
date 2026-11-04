import uuid
from pathlib import Path

import pytest

from logpipe.calibrate import calibrate
from logpipe.enrich.llm import OfflineAdapter
from logpipe.sources.file_replay import replay

FIXTURES = Path(__file__).parent / "fixtures" / "iot"


@pytest.fixture(scope="session")
def fixture_lines() -> list[str]:
    return [event.line for event in replay(FIXTURES)]


@pytest.fixture(scope="session")
def snapshot(fixture_lines):
    return calibrate(fixture_lines, OfflineAdapter()).snapshot


@pytest.fixture
def clickhouse_store():
    """A ClickHouseStore on a fresh database, dropped afterwards (integration tests only)."""
    from logpipe.storage.clickhouse import ClickHouseStore, connect

    try:
        admin = connect()
        admin.command("SELECT 1")
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"ClickHouse not reachable: {exc}")
    database = f"test_{uuid.uuid4().hex[:8]}"
    admin.command(f"CREATE DATABASE {database}")
    store = ClickHouseStore.from_env(database=database)
    try:
        yield store
    finally:
        store.client.close()
        admin.command(f"DROP DATABASE {database} SYNC")
        admin.close()
