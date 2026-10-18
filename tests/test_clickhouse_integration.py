"""Integration smoke test against a live ClickHouse (``pytest -m integration``).

Locally: ``make test-integration`` starts the compose service first.  In CI the
``integration`` job provides a ClickHouse service container and the same
``CLICKHOUSE_*`` variables as ``.env.example``.

Only the HTTP interface is used, with the standard library, so this test has no
dependency on the storage backend that is still being merged; when
``logpipe.storage.clickhouse`` lands its own tests join this marker and run
in the same job.
"""

from __future__ import annotations

import base64
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid

import pytest

pytestmark = pytest.mark.integration

HOST = os.environ.get("CLICKHOUSE_HOST", "127.0.0.1")
PORT = os.environ.get("CLICKHOUSE_PORT", "8123")
USER = os.environ.get("CLICKHOUSE_USER", "logpipe")
PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD", "change-me")
DATABASE = os.environ.get("CLICKHOUSE_DB", "logpipe")


def _query(sql: str, *, database: str | None = None) -> str:
    # The SQL goes in the POST body: ClickHouse treats GET as read-only, so a
    # CREATE / INSERT / ALTER sent in the URL fails with "Cannot execute query
    # in readonly mode" (code 164).
    params = {"database": database} if database else {}
    url = f"http://{HOST}:{PORT}/" + (f"?{urllib.parse.urlencode(params)}" if params else "")
    token = base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()
    request = urllib.request.Request(
        url, data=sql.encode(), method="POST", headers={"Authorization": f"Basic {token}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read().decode().strip()
    except urllib.error.HTTPError as exc:
        # ClickHouse puts the real error in the body; surface it instead of a bare 500.
        body = exc.read().decode(errors="replace").strip()
        raise AssertionError(f"ClickHouse HTTP {exc.code} for {sql!r}: {body}") from None


@pytest.fixture(scope="module")
def clickhouse() -> None:
    try:
        assert _query("SELECT 1") == "1"
    except (urllib.error.URLError, OSError) as exc:  # pragma: no cover - env dependent
        pytest.skip(f"ClickHouse not reachable at {HOST}:{PORT}: {exc}")


def test_server_answers_and_database_exists(clickhouse: None) -> None:
    assert _query("SELECT version()")
    databases = _query("SHOW DATABASES").splitlines()
    assert DATABASE in databases


def test_round_trip_insert_and_read(clickhouse: None) -> None:
    table = f"ci_smoke_{uuid.uuid4().hex[:8]}"
    _query(
        f"CREATE TABLE {table} (device_id String, ts DateTime64(3), event_id UInt32, "
        f"status Enum8('FINAL' = 1, 'PENDING' = 2)) ENGINE = MergeTree ORDER BY (device_id, ts)",
        database=DATABASE,
    )
    try:
        _query(
            f"INSERT INTO {table} VALUES "
            "('dev-1', '2026-01-01 00:00:00.000', 7, 'FINAL'), "
            "('dev-1', '2026-01-01 00:00:01.000', 1, 'PENDING')",
            database=DATABASE,
        )
        pending = _query(f"SELECT count() FROM {table} WHERE status = 'PENDING'", database=DATABASE)
        assert pending == "1"
        _query(
            f"ALTER TABLE {table} UPDATE status = 'FINAL', event_id = 9 WHERE event_id = 1",
            database=DATABASE,
        )
        _query(
            f"SELECT count() FROM {table}", database=DATABASE
        )  # mutation is async; the table stays readable
    finally:
        _query(f"DROP TABLE IF EXISTS {table}", database=DATABASE)
