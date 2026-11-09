"""Hostile log lines may steer the LLM. Whatever it then answers, the guards around it hold: an adapter that obeys every injection still can't plant bad keys, masks, templates, aliases or SQL."""
from __future__ import annotations

from pathlib import Path

import pytest
from deepparse.masks_types import Mask
from fastapi.testclient import TestClient

from logpipe.enrich.validate import clean_keys, safe_masks, template_problem
from logpipe.parsing.kv import default_keys
from logpipe.server.app import create_app
from tests.test_server import env  # noqa: F401 - fixture

CORPUS = [line for line in (Path(__file__).parent / "fixtures" / "prompt_injection.txt").read_text().splitlines()
          if line.strip() and not line.startswith("#")]
TOKENS = ["user", "<*>", "logged", "in", "from", "<*>"]

# What an LLM that obeyed the lines above would answer.
HOSTILE_KEYS = [["'; DROP TABLE iot_logs; --", "ip"], ["__proto__", "constructor"], ["x y z", "ok"],
                ["password", "password"], ["a" * 80, "b"], ["user_id"], ["User", "IP"]]
HOSTILE_MASKS = ["(a+)+$", "(.*)*", "([a-z]+)*x", "[unclosed", "(?P<x"]
HOSTILE_SQL = [
    # Destructive statements
    "DROP TABLE logs",
    "TRUNCATE TABLE logs",
    "DELETE FROM logs",
    "UPDATE logs SET service = 'x'",
    "INSERT INTO logs VALUES (1)",
    "ALTER TABLE logs ADD COLUMN pwned TEXT",
    "ALTER USER default IDENTIFIED BY 'pwned'",
    "GRANT ALL ON logs TO attacker",
    "REVOKE ALL ON logs FROM attacker",

    # Multiple statements
    "SELECT 1; DROP TABLE logs",
    "SELECT 1; DELETE FROM logs",
    "SELECT 1; UPDATE logs SET service = 'x'",

    # Non-SELECT commands
    "PRAGMA table_info(logs)",
    "ATTACH DATABASE '/tmp/test.db' AS test",
    "DETACH DATABASE test",
    "EXPLAIN SELECT * FROM logs",
    "SHOW TABLES",
    "DESCRIBE logs",

    # Unauthorized tables and metadata
    "SELECT * FROM sqlite_master",
    "SELECT * FROM sqlite_schema",
    "SELECT * FROM information_schema.tables",
    "SELECT * FROM system.tables",
    "SELECT current_user",
    "SELECT current_database",

    # Dangerous functions and table functions
    "SELECT SLEEP(1)",
    "SELECT sleep(1)",
    "SELECT pg_sleep(1)",
    "SELECT BENCHMARK(1000000, SHA2('x', 256))",
    "SELECT * FROM readfile('/etc/passwd')",
    "SELECT * FROM url('http://example.com')",
    "SELECT load_extension('/tmp/evil.so')",
    "SELECT * FROM remote('attacker', 'db', 'table')",

    # SQL injection fragments
    "' OR '1'='1",
    "'; DROP TABLE logs; --",
    "1 UNION SELECT * FROM sqlite_master",
    "SELECT * FROM logs WHERE service = '' OR 1=1 --",
    "SELECT * FROM logs WHERE service = 'x' UNION SELECT * FROM sqlite_master",

    # Obfuscation and comments
    "sElEcT 1; DrOp TaBlE logs",
    "SELECT/**/1;/**/DROP/**/TABLE logs",
    "SELECT 1 -- comment\n; DROP TABLE logs",
    "SELECT /* malicious comment */ * FROM logs",

    # File and external access
    "SELECT * INTO OUTFILE '/tmp/stolen.txt' FROM logs",
    "SELECT * FROM file('/etc/passwd')",
]


def test_corpus_is_loaded():
    assert len(CORPUS) >= 10


@pytest.mark.parametrize("keys", HOSTILE_KEYS)
def test_hostile_key_names_fall_back_to_safe_ones(keys):
    assert clean_keys(keys, TOKENS) == default_keys(TOKENS)


def test_hostile_masks_are_dropped():
    masks = [Mask(label=f"m{i}", pattern=p, justification=CORPUS[i]) for i, p in enumerate(HOSTILE_MASKS)]
    assert safe_masks(masks) == []


@pytest.mark.parametrize("line", CORPUS)
def test_a_template_that_swallows_everything_is_refused(line):
    assert template_problem(["<*>"] * 3, [line]) is not None  # "treat every template as <*> <*> <*>"


def test_an_always_yes_llm_cannot_merge_unrelated_or_opposite_events():
    from logpipe.enrich.similarity import find_alias
    from logpipe.registry import NewTemplate, SchemaRegistry

    class Yes:
        def confirm_alias(self, *_):
            return True

    reg = SchemaRegistry()
    reg.publish([NewTemplate(("door", "<*>", "locked", "by", "<*>"), ("door", "user")),
                 NewTemplate(("payment", "<*>", "captured"), ("id",))])
    for tokens in (("door", "<*>", "unlocked", "by", "<*>"),  # opposite state
                   ("all", "files", "deleted", "by", "<*>")):  # unrelated
        assert find_alias(tokens, reg.current, Yes(), CORPUS[:3]).alias_of is None


# TODO: Use benign SQL to test too


@pytest.mark.parametrize("sql", HOSTILE_SQL)
def test_sql_an_llm_was_talked_into_never_writes(env, sql):  # noqa: F811
    server, _, _ = env

    class Obeys:
        def __getattr__(self, name):
            return getattr(server.adapter_factory(), name)

        def write_sql(self, question, schema, engine):
            return sql

    server._adapter = Obeys()
    client = TestClient(create_app(server))
    r = client.post("/api/query/ask", json={"question": CORPUS[6]})

    print("SQL:", repr(sql))
    print("STATUS:", r.status_code)
    print("BODY:", r.text)

    assert r.status_code in (400, 422), (
        f"Unexpected response for {sql!r}: "
        f"status={r.status_code}, body={r.text}"
    )
    assert client.post("/api/query", json={"sql": sql}).status_code == 400
    assert server.reads.query("SELECT count(*) FROM logs")["rows"] == [(0,)]  # table still there


@pytest.mark.parametrize("sql", HOSTILE_SQL)
def test_server_rejects_hostile_sql(env, sql):
    server, _, _ = env
    client = TestClient(create_app(server))

    executed = []

    original_query = server.reads.query

    def tracked_query(statement):
        executed.append(statement)
        return original_query(statement)

    server.reads.query = tracked_query

    response = client.post("/api/query", json={"sql": sql})

    assert response.status_code == 400, (
        f"Server accepted hostile SQL: {sql!r}; "
        f"response={response.text}"
    )

    assert executed == [], (
        f"Hostile SQL reached the database executor: {executed}"
    )


def test_explanations_are_capped(env):  # noqa: F811
    from logpipe.enrich.llm import OfflineAdapter, cap_explanation

    out = cap_explanation({"summary": "x" * 10_000, "likely_cause": "y" * 10_000,
                           "next_steps": ["z" * 5_000] * 9})
    assert len(out["summary"]) <= 1000 and len(out["likely_cause"]) <= 1000
    assert len(out["next_steps"]) == 4 and all(len(s) <= 500 for s in out["next_steps"])
    assert OfflineAdapter  # the offline summary is built from facts, never from the lines' text
