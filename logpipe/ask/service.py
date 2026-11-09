"""Ask on the server: plans that wait for approval, the job that runs approved rounds, and the
glue to the server's stores, models and settings.

A plan is a control doc (kind ``ask_plan``). Nothing in it runs until ``approve``: the server,
not the UI, enforces that. With approval switched off in Settings (``ask_require_approval``),
rounds are approved as they're planned; a request can still ask for approval (it can only
tighten the setting, never loosen it). A plan waiting for approval expires after 15 minutes.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..registry import UNK_TEMPLATE_ID
from .engine import MAX_ROUNDS, Ctx, Outcome, Round, advance, execute, first_round, overlap_search
from .router import route
from .spec import Catalog, QuerySpec, validate
from .store import LogStore, Query

PLAN_TTL = timedelta(minutes=15)
KIND = "ask_plan"
FINAL = ("done", "failed", "rejected", "expired")


def _now() -> datetime:
    return datetime.now(UTC)


class AskService:
    def __init__(self, server):
        self.server = server
        self.store = LogStore(server.reads)
        self.dir = Path(server.data_dir) / "ask"
        self.index = None  # set by logpipe.ask.index once built

    # --- context ------------------------------------------------------------------------
    def catalog(self) -> Catalog:
        s = self.server
        snap = s.registry.current
        from ..server.core import _template_text

        return Catalog(services={src["service"] for src in s.sources.values() if src["service"]},
                       hosts={src["host"] for src in s.sources.values()} | {a["hostname"] for a in s.control.all("agent")},
                       templates={t: _template_text(e.tokens, e.keys) for t, e in snap.templates.items()
                                  if e.alias_of is None})

    def ctx(self, now: datetime | None = None, adapter=None, **flags) -> Ctx:
        s = self.server
        cat = self.catalog()
        snap = s.registry.current

        def scorer_for(service: str):
            for src in s.sources.values():
                if src["service"] == service:
                    for stream in s.streams_for(src["id"]):
                        if (scorer := s.scorers.get(stream["id"])) is not None:
                            return scorer
            return None

        from .index import shutdown_set

        index = self.index_for(cat)
        return Ctx(store=self.store, templates=cat.templates, canonical=snap.canonical_id, scorer_for=scorer_for,
                   agents=s.agents, alerts=s.reads.alerts,
                   search_templates=index.search if index else overlap_search(cat.templates),
                   neighbours=index.neighbours if index else (lambda tid, k: []),
                   shutdown=shutdown_set(s.control, cat.templates, index), adapter=adapter or s.adapter,
                   now=now or _now(), window=s.window, **flags)

    def index_for(self, cat: Catalog):
        from .index import TemplateIndex

        version = self.server.registry.current.version
        if self.index is None or self.index.version != version or set(self.index.ids) != set(cat.templates):
            self.index = TemplateIndex.build(cat.templates, version, self.server.scorers, self.server.registry.current)
        return self.index

    def require_approval(self, asked: bool | None) -> bool:
        return self.server.settings.ask_require_approval or asked is True

    # --- plans --------------------------------------------------------------------------
    def ask(self, question: str, *, require_approval: bool | None = None, tz_offset: int | None = None,
            spec: dict | None = None, incident: str | None = None) -> dict:
        now = _now()
        cat = self.catalog()
        if spec is not None:
            routed_spec, via, notes = validate(spec, cat, now), "given", []
        else:
            r = route(question, cat, self.server.adapter, now, tz_offset)
            routed_spec, via, notes = r.spec, r.via, r.notes
        plan = {"id": uuid.uuid4().hex[:12], "question": question[:1000], "via": via, "notes": notes,
                "spec": routed_spec.model_dump(mode="json"), "require_approval": self.require_approval(require_approval),
                "created_at": now.isoformat(), "incident": incident, "rounds": [], "state": {}, "result": None}
        return self._plan(plan, first_round(routed_spec, self.ctx(now)), now)

    def _plan(self, plan: dict, out: Outcome, now: datetime, enqueue: bool = True) -> dict:
        plan["state"] = {**plan.get("state", {}), **out.state}
        if out.result is not None:
            plan.update(status="done", result=out.result, finished_at=now.isoformat())
        else:
            assert out.round is not None
            plan["rounds"].append({"n": out.round.n, "purpose": out.round.purpose,
                                   "queries": [asdict(q) for q in out.round.queries]})
            plan.update(status="waiting_approval", expires_at=(now + PLAN_TTL).isoformat())
            if not plan["require_approval"]:
                for q in plan["rounds"][-1]["queries"]:
                    q["status"] = "approved"
                plan["status"] = "approved"
        self.server.control.put(KIND, plan["id"], plan)
        if plan["status"] == "approved" and enqueue:  # the job runs auto-approved rounds itself
            self.server.enqueue("ask", plan=plan["id"], round=plan["rounds"][-1]["n"])
        self.server.events.notify()
        return plan

    def get(self, plan_id: str) -> dict:
        plan = self.server.control.get(KIND, plan_id)
        if plan is None:
            raise KeyError(plan_id)
        if plan["status"] == "waiting_approval" and plan["expires_at"] < _now().isoformat():
            plan["status"] = "expired"
            self.server.control.put(KIND, plan_id, plan)
        return plan

    def recent(self, limit: int = 20) -> list[dict]:
        plans = sorted(self.server.control.all(KIND), key=lambda p: p["created_at"], reverse=True)[:limit]
        return [{k: p.get(k) for k in ("id", "question", "status", "created_at", "via")}
                | {"intent": p["spec"]["intent"]} for p in plans]

    def _waiting(self, plan_id: str) -> dict:
        plan = self.get(plan_id)
        if plan["status"] == "expired":
            raise ValueError("this plan expired (15 minutes without approval); ask again")
        if plan["status"] != "waiting_approval":
            raise ValueError(f"this plan is {plan['status']}, not waiting for approval")
        return plan

    def approve(self, plan_id: str, approved: list[str] | None = None) -> dict:
        """Run the waiting round's queries in ``approved`` (all when None); the rest are skipped."""
        plan = self._waiting(plan_id)
        current = plan["rounds"][-1]
        ids = {q["id"] for q in current["queries"]}
        chosen = ids if approved is None else set(approved)
        if unknown := chosen - ids:
            raise ValueError(f"no query {min(unknown)} in this round")
        if not chosen:
            raise ValueError("approve at least one query, or reject the plan")
        for q in current["queries"]:
            q["status"] = "approved" if q["id"] in chosen else "rejected"
        plan.update(status="approved", approved_at=_now().isoformat())
        self.server.control.put(KIND, plan_id, plan)
        job = self.server.enqueue("ask", plan=plan_id, round=current["n"])
        self.server.events.notify()
        return plan | {"job": job["id"]}

    def edit(self, plan_id: str, spec: dict) -> dict:
        """A changed spec re-plans the first round; it waits for approval again (whatever the setting)."""
        plan = self._waiting(plan_id)
        if len(plan["rounds"]) > 1:
            raise ValueError("the spec can only be edited before the first round runs")
        now = _now()
        new = validate(spec, self.catalog(), now)
        plan.update(spec=new.model_dump(mode="json"), rounds=[], state={}, via=f"{plan['via']}, edited",
                    require_approval=True)
        return self._plan(plan, first_round(new, self.ctx(now)), now)

    def reject(self, plan_id: str) -> dict:
        plan = self._waiting(plan_id)
        for q in plan["rounds"][-1]["queries"]:
            q["status"] = "rejected"
        plan.update(status="rejected", finished_at=_now().isoformat())
        self.server.control.put(KIND, plan_id, plan)
        self.server.events.notify()
        return plan

    # --- the shutdown set ---------------------------------------------------------------
    def shutdown_view(self) -> dict:
        from .index import auto_shutdown

        cat = self.catalog()
        edits = self.server.control.get("ask_shutdown", "edits") or {"add": [], "remove": []}
        auto = auto_shutdown(cat.templates, self.index_for(cat))
        rows = []
        for tid, text in cat.templates.items():
            on = (tid in auto or tid in edits.get("add", [])) and tid not in edits.get("remove", [])
            rows.append({"template_id": tid, "text": text, "shutdown": on, "auto": tid in auto, "why": auto.get(tid),
                         "edited": tid in edits.get("add", []) or tid in edits.get("remove", [])})
        return {"templates": rows, "index": self.index.kind if self.index else None}

    def edit_shutdown(self, body: dict) -> dict:
        """``{"template_id": 12, "shutdown": true|false|null}``: null goes back to the automatic choice."""
        tid, on = body.get("template_id"), body.get("shutdown")
        if not isinstance(tid, int) or tid not in self.catalog().templates:
            raise ValueError("template_id must be an existing template")
        if on is not None and not isinstance(on, bool):
            raise ValueError("shutdown is true, false or null")
        edits = self.server.control.get("ask_shutdown", "edits") or {"add": [], "remove": []}
        add, remove = set(edits.get("add", [])) - {tid}, set(edits.get("remove", [])) - {tid}
        if on is True:
            add.add(tid)
        elif on is False:
            remove.add(tid)
        self.server.control.put("ask_shutdown", "edits", {"add": sorted(add), "remove": sorted(remove)})
        return self.shutdown_view()

    # --- the job ------------------------------------------------------------------------
    def _outputs_path(self, plan_id: str) -> Path:
        return self.dir / f"{plan_id}.json"

    def run(self, progress, plan_id: str, round_n: int, adapter=None) -> str:
        plan = self.get(plan_id)
        if plan["status"] != "approved" or plan["rounds"][-1]["n"] != round_n:
            raise ValueError(f"plan {plan_id} round {round_n} is not approved")
        spec = QuerySpec.model_validate(plan["spec"])
        path = self._outputs_path(plan_id)
        outputs: dict[str, list[dict]] = json.loads(path.read_text("utf-8")) if path.exists() else {}
        plan["status"] = "running"
        self.server.control.put(KIND, plan_id, plan)
        rounds = 0
        while True:
            rounds += 1
            current = plan["rounds"][-1]
            queries = [Query(**q) for q in current["queries"]]
            ctx = self.ctx(adapter=adapter)
            for i, q in enumerate(queries):
                if q.status != "approved":
                    continue
                progress(0.1 + 0.8 * i / max(len(queries), 1), f"round {current['n']}: {q.title}")
                try:
                    outputs[q.id] = execute(q, ctx)
                    q.status, q.result_summary = "done", f"{len(outputs[q.id])} rows"
                except (ValueError, PermissionError) as error:
                    q.status, q.error = "failed", str(error)
            current["queries"] = [asdict(q) for q in queries]
            self.dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(outputs, default=str), "utf-8")
            done = Round(current["n"], current["purpose"], queries)
            progress(0.9, "reading what came back")
            out = advance(spec, plan["state"], done, outputs, ctx)
            if out.round is not None and len(plan["rounds"]) >= MAX_ROUNDS:
                out = Outcome(result={"kind": "clarify", "message": "stopped after the maximum number of rounds"},
                              state=out.state)
            plan = self._plan(plan, out, _now(), enqueue=False)
            if plan["status"] != "approved" or rounds > MAX_ROUNDS:  # waiting for a person, or finished
                break
        if plan["status"] == "done" and plan.get("incident"):
            self._save_incident(plan)
        return {"done": "answer ready", "waiting_approval": "waiting for approval of the next round"}.get(
            plan["status"], plan["status"])

    def _save_incident(self, plan: dict) -> None:
        control = self.server.control
        inc = control.get("incident", plan["incident"]) or {"id": plan["incident"], "status": "open"}
        control.put("incident", plan["incident"], inc | {"explanation": plan["result"] | {"at": _now().isoformat(),
                                                                                           "plan": plan["id"]}})


def unparsed(tid: int) -> bool:
    return tid == UNK_TEMPLATE_ID
