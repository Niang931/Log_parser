"""The RCAEval adapter on a hand-made case: no download and no LLM."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

from logpipe.eval.rcaeval_bench import Transcript, main, parse_reply, ranks

INJECT = datetime(2024, 1, 15, 21, 24, tzinfo=UTC)
SERVICES = ["frontend", "cartservice", "checkoutservice"]


def write_case(root):
    case = root / "re3-ob" / "cases" / "re3-ob-0001"
    case.mkdir(parents=True)
    task = {
        "case_id": "re3-ob-0001",
        "system": "Online Boutique",
        "inject_time": int(INJECT.timestamp()),
        "files": {"logs": "logs.csv"},
        "candidate_services": SERVICES,
    }
    (case / "task.json").write_text(json.dumps(task))
    rows = []
    t = INJECT - timedelta(minutes=10)
    i = 0
    while t < INJECT + timedelta(minutes=5):
        for s in SERVICES:
            rows.append((t, s, f"INFO request {i} served by {s} in {i % 7 + 3} ms"))
        if t >= INJECT:
            rows.append((t, "checkoutservice", f"ERROR panic: nil pointer in PlaceOrder order={i}"))
        t += timedelta(seconds=10)
        i += 1
    with (case / "logs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "timestamp", "container_name", "message"])
        for ts, s, m in rows:
            w.writerow([int(ts.timestamp()), int(ts.timestamp() * 1e9), s, m])
    return case.parent.parent


def test_ranks_the_failing_service_first(tmp_path):
    tasks = write_case(tmp_path)
    main([str(tasks), "--variants", "full", "--epochs", "1"])
    [pred] = [json.loads(line) for line in (tmp_path / "preds.re3-ob.full.jsonl").open()]
    assert pred["case_id"] == "re3-ob-0001"
    assert pred["ranks"][0] == "checkoutservice", pred
    assert pred["usage"]["wall_time_s"] > 0
    [kept] = (tmp_path / "transcripts" / "re3-ob.full.reports.jsonl").read_text().splitlines()
    assert json.loads(kept)["report"]["candidates"]

    # a second run resumes: the answered case isn't redone
    main([str(tasks), "--variants", "full", "--epochs", "1"])
    assert len((tmp_path / "preds.re3-ob.full.jsonl").read_text().splitlines()) == 1


def test_ranks_keep_rcaeval_names_in_report_order():
    report = {
        "candidates": [{"service": "CheckoutService"}, {"service": "redis"}],
        "anchor": {"service": "frontend"},
        "evidence": [{"service": "checkoutservice"}, {"service": "cartservice"}],
    }
    assert ranks(report, SERVICES) == ["checkoutservice", "frontend", "cartservice"]


def test_parse_reply_finds_the_json_in_prose():
    text = 'Looking at it... ```json\n{"ranks": ["cartservice"], "reasoning": "x {y}"}\n```'
    assert parse_reply(text)["ranks"] == ["cartservice"]
    assert parse_reply("no json here") == {}


def test_the_llm_cited_services_lead_the_ranks():
    report = {
        "candidates": [{"service": "redis"}, {"service": "frontend"}],
        "evidence": [
            {"ref": "L1", "service": "redis", "template_id": 3},
            {"ref": "L2", "service": "cartservice", "template_id": 9},
        ],
        "answer": {"by": "llm", "hypotheses": [{"cause": "cart", "cites": ["T9"]}]},
    }
    assert ranks(report, SERVICES + ["redis"])[:2] == ["cartservice", "redis"]
    # the rules' own answer doesn't reorder anything
    report["answer"]["by"] = "rules"
    assert ranks(report, SERVICES + ["redis"])[0] == "redis"


class Reply(BaseModel):
    text: str


class Echo:
    def ask(self, schema, prompt):
        if prompt == "boom":
            raise TimeoutError("no reply")
        return schema(text=prompt.upper())


def test_transcript_keeps_every_prompt_reply_and_error(tmp_path):
    log = tmp_path / "t.jsonl"
    llm = Transcript(Echo(), log)
    llm.case = "re3-ob-0001"
    assert llm.ask(Reply, "hi").text == "HI"
    try:
        llm.ask(Reply, "boom")
    except TimeoutError:
        pass
    ok, failed = [json.loads(line) for line in log.read_text().splitlines()]
    assert ok["case_id"] == "re3-ob-0001" and ok["prompt"] == "hi" and ok["reply"] == {"text": "HI"}
    assert failed["reply"] is None and failed["error"] == "TimeoutError: no reply"
