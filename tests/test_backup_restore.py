"""scripts/backup.sh then scripts/restore.sh against the real bucket (``pytest -m integration``).

Needs ClickHouse, Docker and the BACKUP_B2_* (or BACKUP_S3_*) settings; skipped without a
bucket. Files go to ``test/<random>/`` in every configured bucket and are deleted afterwards.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from logpipe.parsing.frozen_parser import ParseResult, Status
from logpipe.registry import SchemaRegistry
from logpipe.storage.base import AnomalyAlert, LogRow
from logpipe.storage.clickhouse import LOG_COLUMNS

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
# The day backup.sh exports by default; recent, so no TTL touches it.
TS = (datetime.now(UTC) - timedelta(days=2)).replace(hour=12, minute=0, second=0, microsecond=0)
DAY = TS.strftime("%Y%m%d")
TABLES = {"logs": "timestamp", "alerts": "window_start", "logs_dead_letter": "timestamp"}


def _script(name: str, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(ROOT / "scripts" / name), *args],
        env=env, capture_output=True, text=True, timeout=600,
    )


def _day(store) -> dict[str, list]:
    """Every row of DAY, per table (one row each in this test)."""
    rows = {}
    for table, column in TABLES.items():
        final = "FINAL" if table == "logs" else ""
        rows[table] = store.client.query(
            f"SELECT * FROM {table} {final} WHERE toYYYYMMDD({column}) = {DAY}"
        ).result_rows
    return rows


def _fill(store) -> None:
    pending = LogRow.from_result(
        ParseResult(-1, Status.PENDING, {}, registry_version=3),
        device_id="d1", timestamp=TS, raw_message="line; with ' and \\",
    )
    # payload keys an LLM may produce, and characters that SQL text must escape
    resolved = pending.resolved(
        ParseResult(7, Status.FINAL, {"a.b": "x;y", "$k": "it's"}, registry_version=4)
    )
    other_day = LogRow.from_result(
        ParseResult(7, Status.FINAL, {}, registry_version=4),
        device_id="d2", timestamp=TS + timedelta(days=1), raw_message="not in the backup",
    )
    store.insert_logs([pending, resolved, other_day])
    store.insert_alerts([AnomalyAlert(
        "d1", TS, TS + timedelta(minutes=5), (7, -1), 0.5, 0.25, None, ("miss",), True, 4, "m1",
    )])
    store.flush()
    dead = LogRow.from_result(
        ParseResult(-1, Status.PENDING, {}, registry_version=3),
        device_id="d3", timestamp=TS, raw_message="dead",
    )
    store.client.insert(
        "logs_dead_letter",
        [[getattr(dead, c) for c in LOG_COLUMNS] + [TS, "pending_max_age"]],
        column_names=[*LOG_COLUMNS, "dead_lettered_at", "reason"],
    )


def test_backup_then_restore_gives_back_the_same_day(clickhouse_store, snapshot, tmp_path):
    if not (os.environ.get("BACKUP_B2_BUCKET") or os.environ.get("BACKUP_S3_BUCKET")):
        pytest.skip("no backup bucket configured (BACKUP_B2_BUCKET / BACKUP_S3_BUCKET)")
    target = "B2" if os.environ.get("BACKUP_B2_BUCKET") else "S3"
    store = clickhouse_store
    _fill(store)
    before = _day(store)
    assert [len(before[t]) for t in TABLES] == [1, 1, 1]

    registry = tmp_path / "registry.json"
    SchemaRegistry(snapshot).save(registry)
    model = tmp_path / "logbert"
    model.mkdir()
    (model / "model.pt").write_bytes(os.urandom(1024))
    (model / "config.json").write_text('{"version": "m1"}')
    env = os.environ | {
        "CLICKHOUSE_DB": store.client.database,
        "BACKUP_PREFIX": f"test/{uuid.uuid4().hex[:8]}/",
        "DAY": DAY,
        "REGISTRY_PATH": str(registry),
        "MODEL_PATH": str(model),
    }
    tag = f"{DAY}-r{snapshot.version}"
    restored = tmp_path / "restored"
    try:
        backup = _script("backup.sh", env=env)
        assert backup.returncode == 0, backup.stderr
        assert f"tag {tag}" in backup.stdout

        refused = _script("restore.sh", tag, target, env=env)
        assert refused.returncode != 0 and "restore into an empty day only" in refused.stderr

        for table in TABLES:
            store.client.command(f"ALTER TABLE {table} DROP PARTITION {DAY}")
        assert _day(store) == {t: [] for t in TABLES}

        env |= {"REGISTRY_PATH": str(restored / "registry.json"),
                "MODEL_PATH": str(restored / "logbert")}
        restore = _script("restore.sh", tag, target, env=env)
        assert restore.returncode == 0, restore.stderr
    finally:
        subprocess.run(
            ["bash", "-c", 'source scripts/backup_lib.sh; for t in $(enabled_targets); do '
             '(use_target "$t" && s3 rm --recursive --quiet "$(url "")"); done'],
            cwd=ROOT, env=env, timeout=300, check=False,
        )

    assert _day(store) == before
    assert store.count_by_status() == {"FINAL": 2}
    assert (restored / "registry.json").read_bytes() == registry.read_bytes()
    for name in ("model.pt", "config.json"):
        assert (restored / "logbert" / name).read_bytes() == (model / name).read_bytes()
