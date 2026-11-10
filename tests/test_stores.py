"""The store adapters' compiled reads (shown before they run) and the read-only guarantees under them."""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from logpipe.ask.query import Filters
from logpipe.parsing.frozen_parser import Status
from logpipe.server.stores.base import level
from logpipe.server.stores.memory import MemoryReads
from logpipe.server.stores.mongo import LEVEL_EXPR, MongoCompiler, check
from logpipe.server.stores.sql import guard_sql
from logpipe.storage.base import LogRow
from logpipe.storage.memory import MemoryStore

T = datetime(2026, 11, 8, 10, 0, tzinfo=UTC)


def row(sec: float, msg: str, tid: int, service: str = "api", host: str = "web-1") -> LogRow:
    return LogRow(uuid.uuid4(), T + timedelta(seconds=sec), host, tid, Status.FINAL, 1, {"upstream": "pay-1"} if tid == 4 else {}, msg,
                  service=service, source="java")


@pytest.fixture
def store():
    m = MemoryStore()
    m.insert_logs([row(i * 10, f"order {i} confirmed", 3) for i in range(30)]
                  + [row(100, "ERROR db pool exhausted", 4, "db", "db-1"), row(130, "ERROR db pool exhausted", 4, "db", "db-1"),
                     row(120, "WARN slow query 900ms", 5, "db", "db-1"), row(250, "order 99 confirmed", 3, "web")])
    return MemoryReads(m)


def run(reads, text: str) -> list[dict]:
    return reads.records(text, 2000)


def test_rows_filters_and_order(store):
    f = Filters(services=["db"], levels=["error"])
    q = store.compiler.rows(f.builder(), T, T + timedelta(hours=1), 200)
    assert "service IN ('db')" in q and "level IN ('error')" in q
    rows = run(store, q)
    assert [r["message"] for r in rows] == ["ERROR db pool exhausted"] * 2
    assert rows[0]["timestamp"] < rows[1]["timestamp"] and rows[0]["level"] == "error" and "payload" not in rows[0]
    with_fields = run(store, store.compiler.rows(f.builder(), T, T + timedelta(hours=1), payload=True))
    assert json.loads(with_fields[0]["payload"]) == {"upstream": "pay-1"}  # JSON text; LogLine reads it
    newest = run(store, store.compiler.rows(Filters(keywords=["ORDER 2"]).builder(), T, T + timedelta(hours=1), newest_first=True))
    assert [r["message"] for r in newest] == [f"order {i} confirmed" for i in (29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 2)]
    # an absolute range is inclusive at both ends, to the microsecond
    edge = run(store, store.compiler.rows(Filters(template_ids=[3], services=["api"]).builder(), T + timedelta(seconds=20), T + timedelta(seconds=40)))
    assert [r["message"] for r in edge] == ["order 2 confirmed", "order 3 confirmed", "order 4 confirmed"]


def test_template_counts_and_buckets(store):
    counts = run(store, store.compiler.template_counts(Filters().builder(), T, T + timedelta(hours=1)))
    by = {(c["template_id"], c["service"]): c for c in counts}
    assert by[(3, "api")]["n"] == 30 and by[(4, "db")]["n"] == 2 and by[(3, "web")]["n"] == 1
    assert by[(4, "db")]["example"] == "ERROR db pool exhausted" and by[(4, "db")]["first"].startswith("2026-11-08T10:01:40")
    assert by[(4, "db")]["level"] == "error" and by[(4, "db")]["host"] == "db-1"
    assert next(c["template_id"] for c in counts) == 3  # first seen first
    b = run(store, store.compiler.buckets(Filters(services=["db"]).builder(), T, T + timedelta(hours=1), 60, by="service"))
    assert sorted((r["bucket"], r["n"]) for r in b) == [(1, 1), (2, 2)]  # 100s -> bucket 1; 120s, 130s -> 2
    with pytest.raises(ValueError):
        store.compiler.buckets(Filters().builder(), T, T + timedelta(hours=1), 60, by="raw_message")


def test_aggregate_through_the_builder(store):
    rows = run(store, store.compiler.build({"start": T.isoformat(), "end": (T + timedelta(hours=1)).isoformat(),
                                            "group_by": ["service"]}))
    assert {r["service"]: r["lines"] for r in rows} == {"api": 30, "db": 3, "web": 1}


def test_values_stay_values(store):
    evil = Filters(services=["x') OR 1=1 --"], keywords=["'; DROP TABLE logs; --"])
    assert run(store, store.compiler.rows(evil.builder(), T, T + timedelta(hours=1))) == []
    assert len(run(store, store.compiler.rows(Filters().builder(), T, T + timedelta(hours=1)))) == 34  # still there


@pytest.mark.parametrize("sql", [
    "INSERT INTO logs VALUES (1)", "DELETE FROM logs", "DROP TABLE logs", "UPDATE logs SET level = 'x'",
    "SELECT 1; DROP TABLE logs", "SELECT * FROM url('http://169.254.169.254/latest', CSV)",
    "SELECT * FROM file('/etc/passwd', 'LineAsString')", "SELECT * FROM remote('10.0.0.1', db.t)",
    "SELECT * FROM s3('https://bucket/x')", "select * from `url`('http://x', CSV)",
    'SELECT * FROM "file"(\'/etc/passwd\')', "SELECT * FROM logs JOIN mysql('h', 'd', 't', 'u', 'p') m ON 1",
    "SELECT * FROM system.users", "SELECT name FROM system . tables", "SELECT 1 INTO OUTFILE '/tmp/x'",
    "WITH x AS (SELECT 1) SELECT * FROM x; ATTACH DATABASE '/tmp/x.db' AS y", "ATTACH DATABASE '/tmp/x' AS y",
    "PRAGMA writable_schema = 1", "VACUUM INTO '/tmp/copy.db'", "SELECT * FROM postgresql('h', 'd', 't', 'u', 'p')",
])
def test_writes_and_table_functions_are_refused(store, sql):
    with pytest.raises(ValueError):
        store.query(sql)
    assert len(run(store, store.compiler.rows(Filters().builder(), T, T + timedelta(hours=1)))) == 34


@pytest.mark.parametrize("sql", [
    "SELECT count(*) FROM logs WHERE raw_message LIKE '%from url(%'",  # inside a string: just text
    "select level, count(*) from logs group by level",
    "WITH e AS (SELECT * FROM logs WHERE level = 'error') SELECT count(*) FROM e",
    "SELECT * FROM (SELECT service FROM logs) s",
    "SELECT format('{}', service) FROM logs -- from file(",
])
def test_guard_lets_reads_through(store, sql):
    assert guard_sql(sql) == sql
    store.query(sql)


def test_sqlite_is_query_only_underneath(store):
    """Even past the guard (a CTE can't write), SQLite itself refuses."""
    with pytest.raises(ValueError):
        store.query("WITH x AS (SELECT 1) INSERT INTO logs (log_id) SELECT 1")


def test_mongo_reads_compile_to_checked_pipelines():
    s = MongoCompiler()
    f = Filters(services=["db"], levels=["error", "warn"], keywords=["pool.x"], template_ids=[4])
    for text in (s.rows(f.builder(), T, T + timedelta(hours=1), 200), s.template_counts(f.builder(), T, T + timedelta(hours=1)),
                 s.buckets(f.builder(), T, T + timedelta(hours=1), 60)):
        p = json.loads(text)
        assert check(p) == p
        match = p[0]["$match"]["$and"]
        assert match[0] == {"timestamp": {"$gte": {"$date": "2026-11-08T10:00:00Z"}, "$lte": {"$date": "2026-11-08T11:00:00Z"}}}
        assert {"service": {"$in": ["db"]}} in match and {"template_id": {"$in": [4]}} in match
        assert {"raw_message": {"$regex": r"pool\.x", "$options": "i"}} in match
        assert {"$expr": {"$in": [LEVEL_EXPR, ["error", "warn"]]}} in match


def test_mongo_level_expression_mirrors_reads_level():
    """LEVEL_EXPR evaluated by a tiny interpreter, against reads.level() on the same lines."""
    import re

    def ev(node, doc):
        if isinstance(node, str) and node.startswith("$"):
            return doc[node[1:]]
        if not isinstance(node, dict):
            return node
        (op, arg), = node.items()
        if op == "$switch":
            for b in arg["branches"]:
                if ev(b["case"], doc):
                    return b["then"]
            return arg["default"]
        if op == "$in":
            return ev(arg[0], doc) in ev(arg[1], doc)
        if op == "$ifNull":
            v = ev(arg[0], doc)
            return v if v is not None else arg[1]
        if op == "$substrCP":
            return ev(arg[0], doc)[arg[1]:arg[1] + arg[2]]
        if op == "$regexMatch":
            return re.search(arg["regex"], ev(arg["input"], doc)) is not None
        raise AssertionError(op)

    lines = [("2026 ERROR x", ()), ("WARN y", ()), ("plain", ()), ("x" * 200 + " ERROR late", ()),
             ('1.2.3.4 "GET / HTTP/1.1" 503 12', ()), ("java.lang.Exception: boom", ()), ("ok", ("severity:warn",)),
             ("DEBUG z", ()), ('"GET / HTTP/1.1" 404 1', ())]
    for msg, tags in lines:
        assert ev(LEVEL_EXPR, {"raw_message": msg, "tags": list(tags)}) == level(msg, tags), msg
