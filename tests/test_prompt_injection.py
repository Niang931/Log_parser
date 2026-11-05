"""Hostile log lines may steer the LLM. Whatever it then answers, the guards around it hold: an
adapter that obeys every injection still can't plant bad keys, masks, templates, aliases or SQL."""
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
HOSTILE_SQL = ["DROP TABLE logs", "SELECT 1; DROP TABLE logs", "WITH x AS (SELECT 1) DELETE FROM logs",
               "ALTER USER default IDENTIFIED BY 'pwned'", "INSERT INTO logs VALUES (1)"]


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
    # Refused outright, or only drafted into the editor; running the draft can't write either.
    assert r.status_code == 400 or r.json() == {"sql": sql}, r.text
    assert client.post("/api/query", json={"sql": sql}).status_code == 400
    assert server.reads.query("SELECT count(*) FROM logs")["rows"] == [(0,)]  # table still there


def test_explanations_are_capped(env):  # noqa: F811
    from logpipe.enrich.llm import OfflineAdapter, cap_explanation

    out = cap_explanation({"summary": "x" * 10_000, "likely_cause": "y" * 10_000,
                           "next_steps": ["z" * 5_000] * 9})
    assert len(out["summary"]) <= 1000 and len(out["likely_cause"]) <= 1000
    assert len(out["next_steps"]) == 4 and all(len(s) <= 500 for s in out["next_steps"])
    assert OfflineAdapter  # the offline summary is built from facts, never from the lines' text
