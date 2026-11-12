"""When the LLM fails or hangs, the plan says so and stops: it never spins forever."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest

from logpipe.ask.activity import LLMFailed, Recorder
from logpipe.settings import get_settings
from tests.test_ask_service import plan_of, ready  # noqa: F401  (fixtures)
from tests.test_server import (
    env,  # noqa: F401  (ready needs it)
    run_jobs,
)

QUESTION = "what is wrong with the database lately"


class Broken:
    """Reads the question, then breaks (or hangs) when asked for the answer."""

    can_ask = True

    def __init__(
        self,
        fail_answer: Exception | None = None,
        hang: threading.Event | None = None,
        fail_route: Exception | None = None,
    ):
        self.fail_answer, self.hang, self.fail_route = fail_answer, hang, fail_route

    def ask(self, schema, prompt):
        if schema.__name__ == "RouteOut":
            if self.fail_route:
                raise self.fail_route
            return schema(intent="rca", services=["db"], anchor_event="db down")
        if self.hang is not None:
            self.hang.wait(30)
            return None
        assert self.fail_answer is not None
        raise self.fail_answer


def answer_of(server, client, plan):
    for _ in range(8):
        p = client.get(f"/api/ask/{plan['id']}").json()
        if p["status"] != "waiting_approval":
            return p
        client.post(f"/api/ask/{plan['id']}/approve")
        run_jobs(server)
    raise AssertionError("still waiting")


def test_a_failing_llm_ends_the_run_with_the_ranking_and_says_why(ready):  # noqa: F811
    server, client = ready
    server._adapter = Broken(fail_answer=RuntimeError("Gemini: signed out"))
    done = answer_of(server, client, plan_of(client, QUESTION))
    answer = done["result"]["answer"]
    assert done["status"] == "done" and answer["by"] == "rules"
    assert "signed out" in answer["llm_error"] and "not a diagnosis" in answer["note"]
    step = done["llm_steps"][-1]
    assert step["status"] == "failed" and "signed out" in step["outcome"]


def test_a_hanging_llm_is_cut_off_at_the_deadline(ready, monkeypatch):  # noqa: F811
    server, client = ready
    monkeypatch.setenv("LOGPIPE_ASK_LLM_TIMEOUT_SECONDS", "0.5")
    get_settings.cache_clear()
    release = threading.Event()
    server._adapter = Broken(hang=release)
    try:
        done = answer_of(server, client, plan_of(client, QUESTION))
    finally:
        release.set()
    assert done["status"] == "done"
    assert "no answer after 0.5 s" in done["result"]["answer"]["llm_error"]
    assert (
        done["llm_steps"][-1]["status"] == "failed"
        and done["llm_steps"][-1]["timeout_seconds"] == 0.5
    )


def test_a_failing_llm_on_the_question_is_a_clear_error_not_a_crash(ready):  # noqa: F811
    server, client = ready
    server._adapter = Broken(fail_route=ConnectionError("browser closed"))
    r = client.post("/api/ask", json={"question": QUESTION})
    assert r.status_code == 400 and "browser closed" in r.json()["detail"]


def test_anything_else_breaking_marks_the_plan_failed(ready, monkeypatch):  # noqa: F811
    server, client = ready
    plan = plan_of(client, "errors from db in the last 15 minutes")

    def boom(*_args, **_kwargs):
        raise RuntimeError("store unreachable")

    monkeypatch.setattr("logpipe.ask.service.advance", boom)
    client.post(f"/api/ask/{plan['id']}/approve")
    job = server.run_job(server.jobs.get_nowait())
    assert job["status"] == "failed"  # the job fails too, and the plan says why
    p = client.get(f"/api/ask/{plan['id']}").json()
    assert p["status"] == "failed" and "store unreachable" in p["error"]


def test_a_plan_left_running_by_a_restart_is_marked_failed(ready):  # noqa: F811
    server, client = ready
    plan = plan_of(client, "errors from db in the last 15 minutes")
    stale = server.control.get("ask_plan", plan["id"])
    old = datetime.now(UTC) - timedelta(seconds=get_settings().ask.llm_timeout_seconds + 300)
    stale.update(status="running", updated_at=old.isoformat())
    server.control.put("ask_plan", plan["id"], stale)
    p = client.get(f"/api/ask/{plan['id']}").json()
    assert p["status"] == "failed" and "stopped responding" in p["error"]


def test_the_recorder_reports_failures_as_llm_failed():
    entries: list[dict] = []

    class Raises:
        can_ask = True

        def ask(self, schema, prompt):
            raise TimeoutError("slow")

    from logpipe.ask.synth import AnswerOut

    with pytest.raises(LLMFailed, match="slow"):
        Recorder(Raises(), entries, timeout=5).ask(AnswerOut, "p")
    assert entries[0]["status"] == "failed"
