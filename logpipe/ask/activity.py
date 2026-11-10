"""What the LLM is doing for a plan, as it happens: one entry per call (reading the question,
writing the answer), with the adapter's own steps (opening the browser, prompt sent, answer in),
the prompt and the reply. Stored on the plan (``plan["ai"]``) so the UI can show it while it runs.
"""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime

from ..enrich import llm

STEPS = {"RouteOut": "Reading your question", "AnswerOut": "Writing the answer from the evidence"}
KEEP = 20_000  # characters of a prompt or reply kept for the UI


def _now() -> str:
    return datetime.now(UTC).isoformat()


def outcome(schema_name: str, out) -> str:
    if out is None:
        return "no usable answer"
    d = out.model_dump()
    if schema_name == "RouteOut":
        return f"read as: {d.get('intent')}"
    if schema_name == "AnswerOut":
        asked = len(d.get("follow_up") or [])
        return (f"{len(d.get('hypotheses') or [])} likely cause(s)" + (", not enough evidence" if d.get("insufficient_evidence") else "")
                + (f", asked for {asked} more read(s)" if asked else ""))
    return "answered"


class Recorder:
    """The LLM adapter, with each call written to ``entries`` (``save`` is called on every change)."""

    def __init__(self, inner, entries: list[dict], save=lambda: None):
        self._inner, self.entries, self._save = inner, entries, save

    @property
    def can_ask(self) -> bool:
        return bool(getattr(self._inner, "can_ask", False))

    def ask(self, schema, prompt: str):
        name = schema.__name__
        entry = {"step": STEPS.get(name, name), "schema": name, "status": "running", "started_at": _now(),
                 "messages": [], "prompt": prompt[:KEEP], "reply": None, "outcome": ""}
        self.entries.append(entry)
        self._save()
        started = time.monotonic()

        def listen(message: str, record: dict | None) -> None:
            if not message.startswith("answer:"):  # the reply itself is kept whole below
                entry["messages"] = [*entry["messages"], {"at": _now(), "text": message[:300]}][-20:]
                self._save()

        outer = llm.LISTEN.get()
        token = llm.LISTEN.set(listen if outer is None else lambda m, r: (listen(m, r), outer(m, r)))
        try:
            out = self._inner.ask(schema, prompt)
        except Exception as error:
            entry.update(status="failed", outcome=str(error)[:300], finished_at=_now(), seconds=round(time.monotonic() - started, 1))
            self._save()
            raise
        finally:
            llm.LISTEN.reset(token)
        entry.update(status="done" if out is not None else "failed", outcome=outcome(name, out), finished_at=_now(),
                     seconds=round(time.monotonic() - started, 1),
                     reply=None if out is None else json.dumps(out.model_dump(), indent=1)[:KEEP])
        self._save()
        return out

    def __getattr__(self, name: str):
        return getattr(self._inner, name)
