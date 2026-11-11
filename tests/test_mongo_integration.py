"""MongoStore / MongoDocs on a live MongoDB (``pytest -m integration``; ``MONGO_URI``)."""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from logpipe.parsing.frozen_parser import ParseResult, Status
from logpipe.server.control import ControlStore
from logpipe.server.metrics import MemorySeries, MongoSeries
from logpipe.server.reads import MemoryReads
from logpipe.server.reads_mongo import MongoReads
from logpipe.storage.base import AnomalyAlert, LogRow
from logpipe.storage.memory import MemoryStore
from logpipe.storage.mongo import MongoDocs, MongoStore

pytestmark = pytest.mark.integration

T0 = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def db(monkeypatch):
    monkeypatch.delenv("MONGO_READONLY_URI", raising=False)  # .env may leak in (load_env); tests set it themselves
    client = MongoClient(os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017"),
                         tz_aware=True, serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")
        name = f"logpipe_test_{uuid.uuid4().hex[:8]}"
        client[name].list_collection_names()  # ping works without auth; this needs rights on a new db
    except PyMongoError:
        pytest.skip("no MongoDB reachable or not allowed to use it (set MONGO_URI with a user that can create databases)")
    yield client[name]
    client.drop_database(name)
    client.close()


def pending(i: int = 0, version: int = 1) -> LogRow:
    result = ParseResult(1, Status.PENDING, {}, version)
    return LogRow.from_result(result, device_id="h1", timestamp=T0 + timedelta(seconds=i),
                              raw_message=f"line {i}", service="svc", source="a.log", tags=("t",))


def final(row: LogRow, version: int, template_id: int = 7) -> LogRow:
    return row.resolved(ParseResult(template_id, Status.FINAL, {"k": "v"}, version))


def test_round_trip(db):
    store = MongoStore(db)
    row = replace(pending(), payload={"a": "b"})
    store.insert_logs([row])
    assert store.fetch_pending() == [row]
    assert store.count_by_status() == {"PENDING": 1}


def test_pending_becomes_final(db):
    store = MongoStore(db)
    row = pending(version=1)
    store.insert_logs([row])
    store.insert_logs([final(row, version=2)])
    assert store.fetch_pending() == []
    assert store.count_by_status() == {"FINAL": 1}


def test_lower_version_ignored_equal_later_wins(db):
    store = MongoStore(db)
    row = pending(version=1)
    store.insert_logs([final(row, version=3)])
    store.insert_logs([row, final(row, version=2)])  # both lower: lose, rest of batch still applies
    assert store.count_by_status() == {"FINAL": 1}
    store.insert_logs([replace(row, version=3)])  # equal version: later insert wins
    assert store.count_by_status() == {"PENDING": 1}


def test_batch_keeps_other_rows_when_one_loses(db):
    store = MongoStore(db)
    old, new = pending(0, version=1), pending(1, version=1)
    store.insert_logs([final(old, version=3)])
    store.insert_logs([old, new])
    assert store.count_by_status() == {"FINAL": 1, "PENDING": 1}


def test_fetch_pending_by_template_returns_final_rows(db):
    store = MongoStore(db)
    row = pending()
    store.insert_logs([final(row, version=2, template_id=7), pending(1)])
    assert [r.template_id for r in store.fetch_pending(template_id=7)] == [7]
    assert len(store.fetch_pending()) == 1


def test_fetch_pending_sorted_and_limited(db):
    store = MongoStore(db)
    store.insert_logs([pending(i) for i in (3, 1, 2)])
    assert [r.raw_message for r in store.fetch_pending(limit=2)] == ["line 1", "line 2"]


def test_ttl_index_only_with_retention(db):
    MongoStore(db)
    assert all("expireAfterSeconds" not in i for i in db["logs"].index_information().values())
    MongoStore(db, retention_days=2)
    ttl = [i["expireAfterSeconds"] for i in db["logs"].index_information().values()
           if "expireAfterSeconds" in i]
    assert ttl == [2 * 86_400]


def test_control_docs_survive_restart(db):
    ctl = ControlStore(MongoDocs(db))
    ctl.put("agents", "a1", {"name": "one"})
    ctl.put("agents", "a2", {"name": "two"})
    ctl.update("agents", "a1", name="uno")
    ctl.delete("agents", "a2")
    again = ControlStore(MongoDocs(db))
    assert again.all("agents") == [{"name": "uno"}]


# UI reads: MongoReads must give the same answers as MemoryReads on the same rows

LINES = ["ERROR [db] lost connection", "WARN disk 91% full", "INFO started", "DEBUG cache miss",
         'GET /a HTTP/1.1" 503 12', 'GET /b HTTP/1.1" 404 7', "Exception in thread main", "job done"]


def both(db):
    """The same 40 rows, in a MemoryStore and in a MongoStore; returns (memory reads, mongo reads)."""
    now = datetime.now(UTC).replace(microsecond=0)
    rows = []
    for i in range(40):
        r = LogRow.from_result(
            ParseResult(i % 3, Status.FINAL if i % 2 else Status.PENDING, {}, 1), device_id=f"h{i % 2}",
            timestamp=now - timedelta(seconds=30 * (i + 2)), raw_message=LINES[i % len(LINES)],
            service=f"s{i % 3}", source=f"f{i % 2}.log", tags=("severity:warn",) if i % 7 == 0 else ())
        rows.append(r)
    alerts = [AnomalyAlert("h0", now - timedelta(minutes=k + 1), now - timedelta(minutes=k), (1, 2), 0.5, 0.1,
                           None, ("novel",), False, 1, "m1") for k in (2, 1, 0)]  # inserted oldest first
    mem, mongo = MemoryStore(), MongoStore(db)
    for st in (mem, mongo):
        st.insert_logs(rows[::-1])  # oldest first, like real ingest (MemoryReads.source_* keep insert order)
        st.insert_alerts(alerts)
    return MemoryReads(mem), MongoReads(mongo)


def test_reads_match_memory_store(db):
    mem, mongo = both(db)
    now = datetime.now(UTC)
    for filters, text in [({}, ""), ({"level": "error"}, ""), ({"level": "warn"}, ""),
                          ({"host": "h1", "status": "FINAL"}, "disk"), ({"template_id": "2"}, ""),
                          ({"template_id": "x"}, ""), ({"service": "s1", "level": "info"}, "")]:
        assert mongo.recent(filters, 100, text, 30) == mem.recent(filters, 100, text, 30), filters
    assert mongo.recent({}, 5) == mem.recent({}, 5)
    assert mongo.between(now - timedelta(minutes=5), now, 100, "h0") == mem.between(now - timedelta(minutes=5), now, 100, "h0")
    assert mongo.between(now - timedelta(minutes=30), now, 7, None, "s2") == mem.between(now - timedelta(minutes=30), now, 7, None, "s2")
    for by in ("level", "status"):
        for filters in ({}, {"host": "h0"}, {"level": "error"}):
            assert mongo.histogram(30, by, filters) == mem.histogram(30, by, filters), (by, filters)
    for field in ("host", "service", "source", "status", "template_id", "level"):
        for filters in ({}, {"level": "error"}, {"host": "h0"}):
            assert dict(mongo.top(field, 30, 50, filters)) == dict(mem.top(field, 30, 50, filters)), (field, filters)
    assert len(mongo.top("host", 30, 1)) == 1
    assert mongo.source_lines("h0", "s0", "f0.log", 3) == mem.source_lines("h0", "s0", "f0.log", 3)
    assert mongo.source_rows("h0", "s0", "f0.log", 3) == mem.source_rows("h0", "s0", "f0.log", 3)
    assert mongo.alerts(2) == mem.alerts(2)


def test_reads_query_is_not_sql(db):
    _, mongo = both(db)
    with pytest.raises(ValueError, match="not supported on MongoDB"):
        mongo.query("SELECT 1")


def test_series_match_memory_and_have_ttl(db):
    mem, mongo = MemorySeries(), MongoSeries(db)
    base = int(time.time()) - 600
    for i in range(30):
        for series in (mem, mongo):
            series.write(base + i * 10, {"lines": float(i), "cpu": i * 1.5})
    assert mongo.query(30, 60) == mem.query(30, 60)
    MongoSeries(db)  # a restart finds the collection and does not fail
    opts = db.command("listCollections", filter={"name": "server_metrics"})["cursor"]["firstBatch"][0]["options"]
    assert opts["expireAfterSeconds"] == 30 * 86_400
    assert opts["timeseries"]["timeField"] == "ts"


def test_free_form_pipelines_use_the_read_only_uri(db, monkeypatch):
    _, mongo = both(db)
    assert mongo.query_logs is mongo.logs  # no MONGO_READONLY_URI: the main connection
    monkeypatch.setenv("MONGO_READONLY_URI", os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017"))
    ro = MongoReads(MongoStore(db))
    assert ro.query_logs.database.client is not ro.logs.database.client  # its own client
    assert ro.query('[{"$group": {"_id": null, "n": {"$sum": 1}}}]')["rows"] == [[None, 40]]


def test_bench_on_mongodb(db):  # `db` only skips when MongoDB is unreachable; the bench makes its own database
    from logpipe.bench import run

    result = run("mongodb", rates=(500,), step_seconds=2, flood=False, say=lambda m: None)
    paced = result["steps"][0]
    assert result["store"] == "mongodb" and paced["kept_up"] and paced["inserts"] > 0 and paced["insert_ms"]["p99"] > 0
    assert result["disk"]["lines"] > 0 and result["disk"]["data_bytes"] > 0 and result["disk"]["index_bytes"] > 0
