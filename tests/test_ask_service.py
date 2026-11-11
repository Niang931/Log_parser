"""Ask on the server: nothing runs before approval, the toggle only loosens from Settings, plans
expire, edits re-plan, and answers come back with what ran."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tests.test_server import app_lines, calibrated, env, events, run_jobs, send  # noqa: F401  (env is a fixture)


@pytest.fixture
def ready(env):  # noqa: F811
    server, client, headers = env
    calibrated(server, client, headers, app_lines(300))
    now = datetime.now(UTC)
    send(
        server,
        client,
        headers,
        [
            e | {"ts": (now - timedelta(minutes=3)).isoformat()}
            for e in events(["ERROR db pool exhausted"] * 3 + ["WARN disk 91% full"], service="db")
        ],
    )
    return server, client


def plan_of(client, question, **kw):
    r = client.post("/api/ask", json={"question": question, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_nothing_runs_before_approval(ready):
    server, client = ready
    plan = plan_of(client, "errors from db in the last 15 minutes")
    assert (
        plan["status"] == "waiting_approval" and plan["via"] == "fast" and plan["require_approval"]
    )
    q = plan["rounds"][0]["queries"][0]
    assert (
        q["status"] == "pending"
        and "level IN ('error')" in q["compiled"]
        and "service IN ('db')" in q["compiled"]
    )
    assert server.jobs.empty()  # not even queued
    # a job for an unapproved round refuses to run
    server.enqueue("ask", plan=plan["id"], round=1)
    job = server.run_job(server.jobs.get_nowait())
    assert job["status"] == "failed" and "not approved" in job["error"]

    approved = client.post(f"/api/ask/{plan['id']}/approve").json()
    assert approved["status"] == "approved"
    run_jobs(server)
    done = client.get(f"/api/ask/{plan['id']}").json()
    assert done["status"] == "done" and done["result"]["kind"] == "lines"
    assert [l["message"] for l in done["result"]["lines"]] == ["ERROR db pool exhausted"] * 3
    assert done["rounds"][0]["queries"][0]["status"] == "done"
    assert client.post(f"/api/ask/{plan['id']}/approve").status_code == 400  # once only
    assert [p["id"] for p in client.get("/api/ask").json()["plans"]] == [plan["id"]]


def test_per_query_approval_and_reject(ready):
    server, client = ready
    plan = plan_of(client, "why did db go down?")  # rca: a discovery round of several reads
    ids = [q["id"] for q in plan["rounds"][0]["queries"]]
    assert len(ids) >= 4 and {q["engine"] for q in plan["rounds"][0]["queries"]} == {
        "sqlite",
        "local",
    }
    assert (
        client.post(f"/api/ask/{plan['id']}/approve", json={"queries": ["nope"]}).status_code == 400
    )
    assert client.post(f"/api/ask/{plan['id']}/approve", json={"queries": []}).status_code == 400
    client.post(f"/api/ask/{plan['id']}/approve", json={"queries": ids[:2]})
    run_jobs(server)
    after = client.get(f"/api/ask/{plan['id']}").json()
    statuses = [q["status"] for q in after["rounds"][0]["queries"]]
    assert statuses[:2] == ["done", "done"] and set(statuses[2:]) == {"rejected"}
    # the next round (or the answer) came from what was approved; any next round waits again
    assert after["status"] in ("waiting_approval", "done")
    if after["status"] == "waiting_approval":
        assert all(q["status"] == "pending" for q in after["rounds"][-1]["queries"])
        assert client.post(f"/api/ask/{plan['id']}/reject").json()["status"] == "rejected"
        assert client.post(f"/api/ask/{plan['id']}/approve").status_code == 400


def test_toggle_off_runs_rounds_but_a_request_can_still_ask_for_approval(ready):
    server, client = ready
    assert client.put("/api/settings", json={"ask_require_approval": "no"}).status_code == 400
    client.put("/api/settings", json={"ask_require_approval": False})
    plan = plan_of(client, "why did db go down?")
    assert plan["status"] == "approved" and not plan["require_approval"]
    run_jobs(server)
    done = client.get(f"/api/ask/{plan['id']}").json()
    assert done["status"] == "done" and len(done["rounds"]) >= 2  # every round ran in the one job
    assert all(q["status"] in ("done", "failed") for r in done["rounds"] for q in r["queries"])

    tight = plan_of(client, "why did db go down?", require_approval=True)
    assert tight["status"] == "waiting_approval"
    # and with approval on, a request can't switch it off
    client.put("/api/settings", json={"ask_require_approval": True})
    assert (
        plan_of(client, "why did db go down?", require_approval=False)["status"]
        == "waiting_approval"
    )
    assert (
        client.post("/api/ask", json={"question": "why", "require_approval": "false"}).status_code
        == 400
    )


def test_edit_replans_and_waits_again(ready):
    _, client = ready
    client.put("/api/settings", json={"ask_require_approval": False})
    plan = plan_of(client, "errors in the last hour", require_approval=True)
    spec = plan["spec"] | {"services": ["db"], "min_level": "warn"}
    edited = client.post(f"/api/ask/{plan['id']}/edit", json={"spec": spec}).json()
    # an edit is always re-approved
    assert edited["status"] == "waiting_approval" and edited["require_approval"]
    assert "level IN ('warn', 'error')" in edited["rounds"][0]["queries"][0]["compiled"]
    bad = client.post(f"/api/ask/{plan['id']}/edit", json={"spec": spec | {"services": ["nope"]}})
    assert bad.status_code == 400 and "unknown service" in bad.json()["detail"]
    assert (
        client.post(
            f"/api/ask/{plan['id']}/edit", json={"spec": spec | {"sql": "DROP"}}
        ).status_code
        == 400
    )


def test_plans_expire(ready):
    server, client = ready
    plan = plan_of(client, "errors in the last hour")
    doc = server.control.get("ask_plan", plan["id"])
    server.control.put(
        "ask_plan",
        plan["id"],
        doc | {"expires_at": (datetime.now(UTC) - timedelta(seconds=1)).isoformat()},
    )
    r = client.post(f"/api/ask/{plan['id']}/approve")
    assert r.status_code == 400 and "expired" in r.json()["detail"]
    assert client.get(f"/api/ask/{plan['id']}").json()["status"] == "expired"


def test_offline_needs_llm_for_hard_questions_and_general_needs_no_queries(ready):
    _, client = ready
    r = client.post("/api/ask", json={"question": "average response time by host"})
    assert r.status_code == 400 and "needs an LLM" in r.json()["detail"]
    assert client.post("/api/ask", json={"question": ""}).status_code == 400
    assert client.get("/api/ask/missing").status_code == 404


def test_general_tier_and_llm_spec(ready):
    server, client = ready

    class Fake:
        can_ask = True

        def ask(self, schema, prompt):
            if schema.__name__ == "RouteOut":
                return schema(intent="general", answer="SIGTERM asks a process to stop.")
            return None

    server._adapter = Fake()
    plan = plan_of(client, "what does SIGTERM mean?")
    assert plan["status"] == "done" and plan["result"]["kind"] == "general"
    assert "not from your logs" in plan["result"]["note"] and plan["rounds"] == []


def test_shutdown_set_is_editable(ready):
    _, client = ready
    view = client.get("/api/ask-shutdown").json()
    tid = view["templates"][0]["template_id"]
    on = client.put("/api/ask-shutdown", json={"template_id": tid, "shutdown": True}).json()
    assert next(t for t in on["templates"] if t["template_id"] == tid)["shutdown"]
    off = client.put("/api/ask-shutdown", json={"template_id": tid, "shutdown": None}).json()
    assert not next(t for t in off["templates"] if t["template_id"] == tid)["edited"]
    assert (
        client.put("/api/ask-shutdown", json={"template_id": 10**6, "shutdown": True}).status_code
        == 400
    )


def test_follow_up_reads_wait_for_approval_and_stop_at_four(ready):
    server, client = ready

    class Curious:
        """Always wants one more search."""

        can_ask = True

        def ask(self, schema, prompt):
            if schema.__name__ == "RouteOut":
                return schema(intent="rca", services=["db"], anchor_event="db down")
            return schema(
                summary="need more",
                hypotheses=[],
                follow_up=[{"kind": "lines", "keywords": ["pool"], "services": ["db"]}],
            )

    server._adapter = Curious()
    plan = plan_of(client, "what is wrong with the database lately")
    assert plan["via"] == "llm"
    purposes = []
    for _ in range(10):
        p = client.get(f"/api/ask/{plan['id']}").json()
        if p["status"] != "waiting_approval":
            break
        purposes.append(p["rounds"][-1]["purpose"])
        client.post(f"/api/ask/{plan['id']}/approve")
        run_jobs(server)
    done = client.get(f"/api/ask/{plan['id']}").json()
    follow = [x for x in purposes if x.startswith("follow-up")]
    assert follow == [f"follow-up {i} of 4" for i in range(1, 5)]
    assert done["status"] == "done" and done["result"]["answer"]["insufficient_evidence"]
    # the activity log the UI shows as it happens
    steps = [(e["step"], e["status"]) for e in done["llm_steps"]]
    assert (
        steps[0] == ("Reading your question", "done")
        and steps.count(("Writing the answer from the evidence", "done")) == 5
    )
    assert "asked for 1 more read" in done["llm_steps"][1]["outcome"] and done["llm_steps"][1][
        "prompt"
    ].startswith("You are helping")
    q = done["rounds"][-1]["queries"][0]
    assert "'%pool%'" in q["compiled"] or "lower('pool')" in q["compiled"]
