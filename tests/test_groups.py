"""Groups of streams: managed through the API, one group per stream, and named in an Ask question
they cover all their services (and the LLM is told which services belong together)."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tests.test_server import app_lines, calibrated, env, events, run_jobs, send  # noqa: F401  (env is a fixture)


@pytest.fixture
def mail(env):  # noqa: F811
    """Three services, each its own stream; ldap and proxy are the mail system."""
    server, client, headers = env
    calibrated(server, client, headers, app_lines(300))
    now = datetime.now(UTC)
    for svc in ("ldap", "proxy", "camera"):
        send(server, client, headers, [e | {"ts": (now - timedelta(minutes=3)).isoformat()}
                                       for e in events([f"ERROR {svc} backend timeout"] * 3, service=svc, host=f"{svc}-1")])
    streams = {s["name"]: s["id"] for s in client.get("/api/streams").json()["streams"]}
    return server, client, streams


def test_groups_hold_streams_and_a_stream_is_in_one_group(mail):
    _, client, streams = mail
    g = client.post("/api/groups", json={"name": "mail", "description": "ldap + proxy",
                                         "streams": [streams["ldap"], streams["proxy"]]}).json()
    assert g["id"] == "mail" and g["services"] == ["ldap", "proxy"] and len(g["streams"]) == 2
    iot = client.post("/api/groups", json={"name": "iot", "streams": [streams["camera"], streams["proxy"]]}).json()
    assert iot["services"] == ["camera", "proxy"]  # proxy moved out of mail
    assert client.get("/api/groups").json()["groups"][1]["services"] == ["ldap"]
    assert client.post("/api/groups", json={"name": "MAIL"}).status_code == 400  # names are unique
    assert client.post("/api/groups", json={"name": "x", "streams": ["nope"]}).status_code == 400
    # a stream's own form moves it too, or takes it out
    s = next(x for x in client.get("/api/streams").json()["streams"] if x["id"] == streams["camera"])
    client.put(f"/api/streams/{s['id']}", json={"name": s["name"], "picked": s["picked"], "group": None})
    assert next(g for g in client.get("/api/groups").json()["groups"] if g["id"] == "iot")["services"] == ["proxy"]
    assert client.put(f"/api/streams/{s['id']}", json={"name": s["name"], "picked": s["picked"], "group": "nope"}).status_code == 400
    client.delete("/api/groups/iot")
    assert all(x.get("group") != "iot" for x in client.get("/api/streams").json()["streams"])
    assert client.delete("/api/groups/iot").status_code == 404


def test_asking_about_a_group_covers_its_services(mail):
    server, client, streams = mail
    client.post("/api/groups", json={"name": "mail", "streams": [streams["ldap"], streams["proxy"]]})
    plan = client.post("/api/ask", json={"question": "errors in mail in the last 15 minutes"}).json()
    assert plan["via"] == "fast" and plan["spec"]["groups"] == ["mail"]
    q = plan["rounds"][0]["queries"][0]["compiled"]
    assert "service IN ('ldap', 'proxy')" in q and "camera" not in q
    client.post(f"/api/ask/{plan['id']}/approve")
    run_jobs(server)
    lines = client.get(f"/api/ask/{plan['id']}").json()["result"]["lines"]
    assert {line["service"] for line in lines} == {"ldap", "proxy"}

    rca = client.post("/api/ask", json={"question": "why did mail go down?"}).json()
    assert all("'camera'" not in q["compiled"] for q in rca["rounds"][0]["queries"] if q["engine"] != "local")
    empty = client.post("/api/groups", json={"name": "spare"}).json()
    r = client.post("/api/ask", json={"question": "errors", "spec": {"intent": "keyword", "groups": [empty["name"]]}})
    assert r.status_code == 400 and "no services yet" in r.json()["detail"]


def test_the_llm_is_told_about_groups(mail):
    server, client, streams = mail
    client.post("/api/groups", json={"name": "mail", "streams": [streams["ldap"], streams["proxy"]]})

    prompts: list[str] = []

    class Reader:
        can_ask = True

        def ask(self, schema, prompt):
            prompts.append(prompt)
            if schema.__name__ == "RouteOut":
                return schema(intent="rca", groups=["mail"], anchor_event="mail system down")
            return schema(summary="ldap timed out first", hypotheses=[])

    server._adapter = Reader()
    plan = client.post("/api/ask", json={"question": "is email ok? nobody can sign in"}).json()
    assert plan["via"] == "llm" and plan["spec"]["groups"] == ["mail"]
    assert "  mail: ldap, proxy" in prompts[0]  # the router sees the groups
    for _ in range(6):
        p = client.get(f"/api/ask/{plan['id']}").json()
        if p["status"] != "waiting_approval":
            break
        client.post(f"/api/ask/{plan['id']}/approve")
        run_jobs(server)
    final = prompts[-1]
    assert "Question: is email ok? nobody can sign in" in final and "mail: ldap, proxy" in final
