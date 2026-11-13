"""Datadog REST: pull a customer's logs in; push incidents, root causes and grok pipelines out."""

from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote

import httpx

from .parsing.grok import _PART
from .settings import DatadogSettings

LOGGER = logging.getLogger(__name__)
PIPELINE_NAME = "logpipe (auto)"


class DatadogAPI:
    def __init__(self, settings: DatadogSettings, transport: httpx.BaseTransport | None = None):
        if not settings.api_key:
            raise ValueError("DD_API_KEY is not set")
        headers = {"DD-API-KEY": settings.api_key}
        if settings.app_key:
            headers["DD-APPLICATION-KEY"] = settings.app_key
        self.settings = settings
        self.http = httpx.Client(
            base_url=settings.api_url, headers=headers, timeout=10, transport=transport
        )

    def send(self, method: str, path: str, **kw) -> Any:
        r = self.http.request(method, path, **kw)
        r.raise_for_status()
        return r.json() if r.content else {}

    def search_logs(self, query: str, since: str, limit: int = 1000) -> list[dict]:
        """One page of logs newer than ``since``, oldest first, as ``ingest()`` events."""
        body = {
            "filter": {"query": query, "from": since, "to": "now"},
            "sort": "timestamp",
            "page": {"limit": limit},
        }
        data = self.send("POST", "/api/v2/logs/events/search", json=body).get("data") or []
        return [to_event(d.get("attributes") or {}) for d in data]

    def post_event(self, title: str, text: str, tags: list[str], key: str) -> dict:
        """v2 (the only events API newer orgs have), on its own intake host; the API key is enough."""
        attrs = {
            "category": "alert",
            "title": title[:100],
            "message": f"%%% \n{text}\n %%%"[:4000],
            "tags": tags,
            "aggregation_key": key[:100],
            "attributes": {"status": "warn", "priority": "2"},
        }
        url = f"https://event-management-intake.{self.settings.site}/api/v2/events"
        return self.send("POST", url, json={"data": {"type": "event", "attributes": attrs}})

    def upsert_pipeline(self, rules: list[str], query: str = "*") -> dict:
        body = {
            "name": PIPELINE_NAME,
            "is_enabled": True,
            "filter": {"query": query},
            "processors": [
                {
                    "type": "grok-parser",
                    "name": "logpipe templates",
                    "is_enabled": True,
                    "source": "message",
                    "samples": [],
                    "grok": {"support_rules": "", "match_rules": "\n".join(rules)},
                }
            ],
        }
        existing = next(
            (
                p
                for p in self.send("GET", "/api/v1/logs/config/pipelines")
                if p.get("name") == PIPELINE_NAME
            ),
            None,
        )
        if existing:
            # shortcut: overwrites edits made in Datadog.
            return self.send("PUT", f"/api/v1/logs/config/pipelines/{existing['id']}", json=body)
        return self.send("POST", "/api/v1/logs/config/pipelines", json=body)


def to_event(a: dict) -> dict:
    """A Datadog log's attributes -> the dict ``Server.ingest()`` takes."""
    tags = a.get("tags") or []
    source = next((t.split(":", 1)[1] for t in tags if t.startswith("source:")), "")
    return {
        "message": a.get("message") or "",
        "host": a.get("host"),
        "service": a.get("service"),
        "source": source,
        "tags": tags,
        "ts": a.get("timestamp", ""),
    }


_DD_MATCHER = {
    "notspace": "notSpace",
    "quoted": "quotedString",
    "path": "notSpace",
    "timestamp": 'regex("[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}[^ ]*")',
}
_SPECIAL = re.compile(r"([\\^$.|?*+()\[\]{}])")


def to_dd_rule(rule: str) -> str:
    """A logpipe rule in Datadog grok: literals regex-escaped, matchers renamed."""
    out, at = [], 0
    for m in _PART.finditer(rule):
        out.append(_SPECIAL.sub(r"\\\1", rule[at : m.start()]))
        matcher, arg, name = m.groups()
        part = f'regex("{arg}")' if arg is not None else _DD_MATCHER.get(matcher, matcher)
        out.append(f"%{{{part}{':' + name if name else ''}}}")
        at = m.end()
    out.append(_SPECIAL.sub(r"\\\1", rule[at:]))
    return "".join(out)


def dd_rules(templates: list[dict]) -> list[str]:
    return [f"t{t['template_id']} {to_dd_rule(t['rule'])}" for t in templates if t.get("rule")]


def _ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


def incident_event(inc: dict, settings: DatadogSettings) -> dict:
    q = " ".join(f"{k}:{inc[k]}" for k in ("host", "service") if inc.get(k))
    logs = (
        f"{settings.app_url}/logs?query={quote(q)}"
        f"&from_ts={_ms(inc['start']) - 300_000}&to_ts={_ms(inc['end']) + 60_000}"
    )
    lines = [
        (
            f"LogBERT flagged unusual log sequences on **{inc['key']}** "
            f"from {inc['start']} to {inc['end']}."
        ),
        f"- alerts: {inc['alerts']}, peak share of unexpected lines: {inc['peak_miss']:.0%}",
        f"- [Logs around it in Datadog]({logs})",
    ]
    if settings.public_url:
        lines.append(f"- [Open in logpipe]({settings.public_url.rstrip('/')}/alerts/{inc['id']})")
    lines.append("Root cause analysis is running; it follows as a second event.")
    return {
        "title": f"logpipe: anomaly on {inc['key']}",
        "text": "\n".join(lines),
        "tags": tags(inc),
        "key": inc["id"],
    }


def rca_event(inc: dict, explanation: dict) -> dict:
    ans = explanation.get("answer")
    lines = [ans if isinstance(ans, str) else explanation.get("message", "")]
    ans = ans if isinstance(ans, dict) else {}
    lines[0] = ans.get("summary") or lines[0]
    for h in (ans.get("hypotheses") or ans.get("leads") or [])[:3]:
        lines.append(f"- **{h['cause']}** ({h.get('confidence', 'low')}) {h.get('reasoning', '')}")
    if ans.get("next_steps"):
        lines.append("Next steps:")
        lines += [f"- {s}" for s in ans["next_steps"][:5]]
    return {
        "title": f"logpipe RCA: {inc['key']}",
        "text": "\n".join(str(x) for x in lines if x) or "No answer; see the incident in logpipe.",
        "tags": tags(inc),
        "key": inc["id"],
    }


def tags(inc: dict) -> list[str]:
    out = ["source:logpipe", f"logpipe_incident:{inc['id']}"]
    return out + [f"{k}:{inc[k]}" for k in ("host", "service") if inc.get(k)]


class DatadogSync:
    def __init__(self, settings: DatadogSettings, query: str = "", api: DatadogAPI | None = None):
        self.api = api or DatadogAPI(settings)
        self.settings = settings
        self.query = query or settings.pull_query
        self.seen: set[str] | None = None
        self.last = 0.0

    def tick(self, server) -> None:
        if time.monotonic() - self.last < self.settings.pull_every_s:
            return
        self.last = time.monotonic()
        for step in (self.pull, self.push):
            try:
                step(server)
            except Exception:
                LOGGER.exception("datadog %s failed", step.__name__)

    def pull(self, server) -> None:
        if not self.query:
            return
        doc = server.control.get("datadog", "pull") or {"id": "pull"}
        since = doc.get("since") or (datetime.now(UTC) - timedelta(minutes=15)).isoformat()
        # shortcut: one 1000-line page per tick (search limit ~300/h); dual-ship via the Agent for more.
        events = self.api.search_logs(self.query, since)
        if not events:
            return
        server.ingest(events)
        # shortcut: time cursor, late lines in the same millisecond are skipped.
        last = datetime.fromisoformat(events[-1]["ts"]) + timedelta(milliseconds=1)
        server.control.put("datadog", "pull", doc | {"since": last.isoformat()})

    def alert(self, server, body: dict) -> dict:
        """Datadog monitor webhook -> RCA anchored at the alert; answered as an event next tick."""
        title = str(body.get("title") or "Datadog alert")[:200]
        when = str(body.get("date") or "")
        anchor = (
            datetime.fromtimestamp(int(when) / 1000, UTC)
            if when.isdigit()
            else datetime.fromisoformat(when)
            if when
            else datetime.now(UTC)
        )
        tags = [t.strip() for t in str(body.get("tags") or "").split(",") if ":" in t]
        tag = dict(t.split(":", 1) for t in tags)
        host = str(body.get("host") or tag.get("host") or "")
        service = str(body.get("service") or tag.get("service") or "")
        spec = {"intent": "rca", "anchor_time": anchor.isoformat(), "anchor_event": title}
        question = f"Why did the Datadog alert '{title}' fire?"
        scoped = spec | {"services": [service] if service else [], "hosts": [host] if host else []}
        try:
            plan = server.ask.ask(question, spec=scoped, require_approval=False)
        except ValueError:  # a host or service logpipe hasn't seen
            plan = server.ask.ask(question, spec=spec, require_approval=False)
        key = str(body.get("id") or plan["id"])
        doc = {"id": f"alert-{plan['id']}", "plan": plan["id"], "title": title, "key": key}
        server.control.put("datadog", doc["id"], doc | {"host": host, "service": service})
        return plan

    def push(self, server) -> None:
        for doc in server.control.all("datadog"):
            if doc.get("plan") and not doc.get("posted"):
                try:
                    plan: dict = server.ask.get(doc["plan"])
                except KeyError:
                    plan = {"status": "deleted"}
                if plan["status"] in ("done", "failed", "rejected", "expired", "deleted"):
                    inc = {"id": doc["key"], "key": doc["title"]} | {
                        k: doc[k] for k in ("host", "service")
                    }
                    self.api.post_event(
                        **rca_event(inc, plan.get("result") or {"message": plan["status"]})
                    )
                    server.control.put("datadog", doc["id"], doc | {"posted": True})
        incidents = server.incidents()
        if self.seen is None:
            self.seen = {i["id"] for i in incidents}
            return
        for inc in incidents:
            state = server.control.get("incident", inc["id"]) or {"id": inc["id"], "status": "open"}
            if not state.get("dd_event") and inc["id"] not in self.seen:
                self.api.post_event(**incident_event(inc, self.settings))
                server.control.put("incident", inc["id"], state | {"dd_event": True})
                if not state.get("ask_plan"):
                    server.explain_incident(inc["id"], require_approval=False)
            elif state.get("dd_event") and not state.get("dd_rca") and state.get("explanation"):
                self.api.post_event(**rca_event(inc, state["explanation"]))
                server.control.put("incident", inc["id"], state | {"dd_rca": True})
