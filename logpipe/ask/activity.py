"""What the LLM is doing for a plan, as it happens: one entry per call (reading the question,
writing the answer), with the adapter's own steps (opening the browser, prompt sent, answer in),
the prompt and the reply. Stored on the plan (``plan["ai"]``) so the UI can show it while it runs.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, cast, get_args

from ..enrich import llm
from ..settings import get_settings

# the structured answers Ask asks the LLM for (router.RouteOut, synth.AnswerOut)
Schema = Literal["RouteOut", "AnswerOut"]
STEPS: dict[Schema, str] = {
    "RouteOut": "Reading your question",
    "AnswerOut": "Writing the answer from the evidence",
}


class StepStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


KEEP = 20_000  # characters of a prompt or reply kept for the UI


def _now() -> str:
    return datetime.now(UTC).isoformat()


def outcome(schema: Schema, out) -> str:
    if out is None:
        return "no usable answer"
    match schema:
        case "RouteOut":
            return f"read as: {out.intent}"
        case "AnswerOut":
            asked = len(out.follow_up)
            return (
                f"{len(out.hypotheses)} likely cause(s)"
                + (", not enough evidence" if out.insufficient_evidence else "")
                + (f", asked for {asked} more read(s)" if asked else "")
            )


class LLMFailed(RuntimeError):
    """The LLM raised, or gave no answer before the deadline. Only the server's Recorder raises
    it: Ask falls back to the rules on it, while any other adapter's error (an eval replay
    missing a reply) still fails loudly."""


class Recorder:
    """The LLM adapter, with each call written to ``entries`` (``save`` is called on every change)
    and cut off after ``timeout`` seconds: a web chat that hangs must not hold the plan forever."""

    def __init__(self, inner, entries: list[dict], save=lambda: None, timeout: float | None = None):
        self._inner, self.entries, self._save = inner, entries, save
        self.timeout = get_settings().ask.llm_timeout_seconds if timeout is None else timeout

    @property
    def can_ask(self) -> bool:
        return bool(getattr(self._inner, "can_ask", False))

    def ask(self, schema, prompt: str):
        if schema.__name__ not in get_args(Schema):
            raise TypeError(f"Ask only asks the LLM for {get_args(Schema)}, not {schema.__name__}")
        name = cast(Schema, schema.__name__)
        entry = {
            "step": STEPS[name],
            "schema": name,
            "status": StepStatus.RUNNING,
            "started_at": _now(),
            "timeout_seconds": self.timeout,
            "messages": [],
            "prompt": prompt[:KEEP],
            "reply": None,
            "outcome": "",
        }
        self.entries.append(entry)
        self._save()
        started = time.monotonic()

        def listen(message: str, record: dict | None) -> None:
            # a call cut off by the deadline may still talk: it no longer belongs to this entry
            if entry["status"] != StepStatus.RUNNING or message.startswith("answer:"):
                return
            said = {"at": _now(), "text": message[:300]}
            entry["messages"] = [*entry["messages"], said][-20:]
            self._save()

        def finish(status: StepStatus, outcome_text: str, reply: str | None = None) -> None:
            seconds = round(time.monotonic() - started, 1)
            entry.update(
                status=status,
                outcome=outcome_text,
                finished_at=_now(),
                seconds=seconds,
                reply=reply,
            )
            self._save()

        result: dict = {}

        def call() -> None:
            try:
                result["out"] = self._inner.ask(schema, prompt)
            except Exception as error:  # noqa: BLE001 - handed to the caller below
                result["error"] = error

        # the call runs on a thread of its own so it can be abandoned; it inherits the listener
        with llm.listening(listen):
            context = contextvars.copy_context()
        worker = threading.Thread(target=context.run, args=(call,), name="ask-llm", daemon=True)
        worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            finish(StepStatus.FAILED, f"no answer after {self.timeout:g} s: stopped waiting")
            raise LLMFailed(f"no answer after {self.timeout:g} s")
        if "error" in result:
            error = result["error"]
            finish(StepStatus.FAILED, f"failed: {error}"[:300])
            raise LLMFailed(str(error)[:300]) from error
        out = result.get("out")
        reply = None if out is None else json.dumps(out.model_dump(), indent=1)[:KEEP]
        finish(StepStatus.DONE if out is not None else StepStatus.FAILED, outcome(name, out), reply)
        return out

    def __getattr__(self, name: str):
        return getattr(self._inner, name)
