"""``logpipe server``'s engine: agent intake into the existing pipeline, the learning lifecycle
(setup -> drafted templates -> review -> per-stream LogBERT -> maintenance -> new drafts and
retraining) and the fleet's control plane.

One ingest thread owns the pipeline; one job thread runs calibration and training; a
housekeeping thread persists counters, runs enrichment and the retraining schedule.
"""
from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import logging
import math
import queue
import random
import re
import secrets
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from deepparse.masks_types import Mask
from deepparse.utils.sampling import entropy_greedy_sample

from ..anomaly.detector import DetectorConfig, StreamingDetector, sliding_windows
from ..anomaly.logbert import LogBERTConfig, TrainConfig, train_logbert
from ..anomaly.scorer import LogBERTScorer, ScorerSettings, Thresholds
from ..anomaly.vocab import TemplateVocab
from ..calibrate import append_drafts, calibrate
from ..enrich import llm
from ..enrich.llm import LLMAdapter
from ..enrich.validate import clean_keys
from ..parsing.frozen_parser import ParserHandle, Status
from ..parsing.grok import compile_rule, rule_from_template
from ..parsing.kv import SLOT_RE
from ..pipeline import Pipeline
from ..registry import UNK_TEMPLATE_ID, NewTemplate, SchemaRegistry
from ..settings import RuntimeSettings, get_settings
from ..sources.file_replay import LogEvent
from ..storage.memory import MemoryStore
from ..unresolved import UnresolvedQueue
from . import review
from .control import ControlStore
from .drift import Drift, histogram
from .groups import Groups
from .metrics import (
    ClickHouseSeries,
    CountedAdapter,
    MemorySeries,
    MetricRules,
    Metrics,
    MongoSeries,
)
from .stores import reads_for

LOGGER = logging.getLogger("logpipe.server")



class Notifier:
    """Wakes async waiters (agent long-polls, UI event streams) from any thread."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self._event = asyncio.Event()

    def bind(self) -> None:
        """Called from the event loop before waiting (uvicorn has one; test clients make several)."""
        loop = asyncio.get_running_loop()
        if self.loop is not loop:
            self.loop, self._event = loop, asyncio.Event()

    def notify(self, delay: float = 0) -> None:
        loop = self.loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(loop.call_later, delay, self._wake)

    def _wake(self) -> None:
        event, self._event = self._event, asyncio.Event()
        event.set()

    async def wait(self, timeout: float) -> bool:
        """True when notified within ``timeout`` seconds."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout)
            return True
        except TimeoutError:
            return False
SOURCE_TYPES = {"file", "journald", "docker", "tcp", "udp"}
RULE_TYPES = {"multi_line", "exclude_at_match", "include_at_match", "mask_sequences"}
JOB_KINDS = ("draft", "apply", "retrain", "reextract", "bench", "ask")
INCIDENT_STATES = ("open", "acknowledged", "resolved", "false_positive")


class NotEnoughData(ValueError):
    def __init__(self, stream: str, have: int, need: int):
        super().__init__(f"stream {stream}: {have} windows of parsed logs, need {need}")
        self.have, self.need = have, need


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def source_id(host: str, service: str, source: str) -> str:
    return f"{host}|{service}|{source}"


def remote_hash(text: str) -> str:
    """Same as the agent's config.HashBytes: the first 8 bytes of sha256, hex."""
    return hashlib.sha256(text.encode()).hexdigest()[:16] if text else ""


def validate_remote_yaml(text: str) -> None:
    """Reject a remote config the agent would refuse, before it is offered to the agent."""
    if not text.strip():
        return
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ValueError(f"not valid YAML: {error}") from error
    if not isinstance(doc, dict) or not isinstance(doc.get("logs"), list):
        raise ValueError("expected a mapping with a `logs:` list")  # noqa: TRY004 - shown as a 400
    for i, src in enumerate(doc["logs"]):
        if not isinstance(src, dict) or src.get("type") not in SOURCE_TYPES:
            raise ValueError(f"logs[{i}]: type must be one of {sorted(SOURCE_TYPES)}")
        if src["type"] == "file" and not src.get("path"):
            raise ValueError(f"logs[{i}]: file source needs path")
        if src["type"] in ("tcp", "udp") and not isinstance(src.get("port"), int):
            raise ValueError(f"logs[{i}]: {src['type']} source needs a port")
        for rule in src.get("log_processing_rules") or []:
            if rule.get("type") not in RULE_TYPES:
                raise ValueError(f"logs[{i}]: unknown rule type {rule.get('type')!r}")
            try:
                re.compile(rule.get("pattern", ""))
            except re.error as error:
                raise ValueError(f"logs[{i}]: bad pattern: {error}") from error


def _matches(selector: dict, host: str, service: str, source: str) -> bool:
    return all(fnmatch.fnmatchcase(value, selector.get(field) or "*")
               for field, value in (("host", host), ("service", service), ("source", source)))


def _when(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


def _template_text(tokens, keys) -> str:
    """``<*>`` slots shown with their field name: ``user <*:user_id> logged in``."""
    names = iter(keys)
    return SLOT_RE.sub(lambda m: f"{m.group(0)[:-1]}:{next(names, '?')}>", " ".join(tokens))


def _brief(value, cap: int = 20):
    """A call's argument or result in a JSON-safe, size-capped form."""
    if isinstance(value, Mask):
        return value.to_dict()
    if isinstance(value, list | tuple):
        return [_brief(v, cap) for v in value[:cap]] + ([f"… {len(value) - cap} more"] if len(value) > cap else [])
    return value if isinstance(value, str | int | float | bool | None) else str(value)


class _Traced:
    """The LLM adapter, reporting each call as a job step (what it asks, the chat steps, the
    answer) and recording its input, prompts and returned value for the detail panel."""

    def __init__(self, inner: LLMAdapter, progress, start: float, end: float):
        self._inner, self._progress = inner, progress
        self.fraction, self._end = start, end
        self.calls: list[dict] = []

    def _report(self, message: str) -> None:
        self.fraction += (self._end - self.fraction) * 0.05  # creeps on: the number of calls isn't known
        self._progress(self.fraction, message, {"LLM calls": self.calls[-50:]})

    def _traced(self, name: str, fn, args: tuple, shown: dict, asking: str, answered):
        prompts: list[dict] = []
        entry = {"call": name, "input": {k: _brief(v) for k, v in shown.items()}, "prompts": prompts,
                 "output": "(waiting)"}
        self.calls.append(entry)
        self._report(asking)

        def listen(message: str, record: dict | None) -> None:
            if record is not None:
                prompts.append(record)
            self._report(f"  {message}")
        token = llm.LISTEN.set(listen)
        try:
            out = fn(*args)
        finally:
            llm.LISTEN.reset(token)
        entry["output"] = _brief(out)
        self._report(answered(out))
        return out

    def synth_masks(self, lines, sample_size):
        return self._traced("synth_masks", self._inner.synth_masks, (lines, sample_size),
                            {"lines": len(lines), "sample size": sample_size, "first lines": list(lines[:10])},
                            f"synthesizing masks from {min(len(lines), sample_size)} sample lines",
                            lambda out: f"{len(out)} masks; Drain is mining templates")

    def ask(self, schema, prompt):
        return self._traced(f"ask:{schema.__name__}", self._inner.ask, (schema, prompt),
                            {"prompt": prompt}, f"asking the LLM ({schema.__name__})",
                            lambda out: "answered" if out is not None else "no usable answer")

    def name_keys(self, tokens, samples):
        return self._traced("name_keys", self._inner.name_keys, (tokens, samples),
                            {"template": " ".join(tokens), "example lines": list(samples[:5])},
                            f"naming the fields of: {' '.join(tokens)}",
                            lambda out: f"fields: {', '.join(out) or '(none)'}")

    def confirm_alias(self, new_tokens, old_tokens, new_samples):
        return self._traced("confirm_alias", self._inner.confirm_alias, (new_tokens, old_tokens, new_samples),
                            {"new": " ".join(new_tokens), "existing": " ".join(old_tokens),
                             "example lines": list(new_samples[:5])},
                            f"same event as an existing template? {' '.join(new_tokens)}",
                            lambda out: "yes, same event" if out else "no, a new event")

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class Busy(RuntimeError):
    """The ingest buffer is full; the agent keeps the batch and retries."""


class Server:
    def __init__(self, store, control: ControlStore, *, data_dir: Path,
                 adapter_factory: Callable[[], LLMAdapter],
                 queue_factory: Callable[[], UnresolvedQueue] | None = None,
                 bootstrap_key: str | None = None, epochs: int = 8, window: int = 20,
                 stride: int = 5, inbox_size: int = 50_000, enrich_every_s: float = 30,
                 schedule_every_s: float = 600, start_threads: bool = True,
                 min_train_steps: int | None = None):
        self.config = get_settings()
        self.store, self.control = store, control
        self.reads = reads_for(store)
        if type(store).__name__ == "MongoStore":
            self.metrics = Metrics(MongoSeries(store.db))
        else:
            self.metrics = Metrics(MemorySeries() if isinstance(store, MemoryStore) else ClickHouseSeries(store.client))
        self.metric_rules = MetricRules(control)
        self.drift = Drift()
        self._templates_at: tuple[float, int] | None = None
        self.data_dir = data_dir
        self.registry_path = data_dir / "registry.json"
        self.adapter_factory, self.queue_factory = adapter_factory, queue_factory
        self._adapter: LLMAdapter | None = None
        self.epochs, self.window, self.stride = epochs, window, stride
        self.min_train_steps = min_train_steps or self.config.server.min_train_steps
        self.enrich_every_s, self.schedule_every_s = enrich_every_s, schedule_every_s
        self.lock = threading.RLock()
        self.inbox: queue.Queue[LogEvent] = queue.Queue(maxsize=inbox_size)
        self.jobs: queue.Queue[str] = queue.Queue()
        self.received = 0
        self._rate: list[tuple[float, int]] = []
        self._dirty: set[tuple[str, str]] = set()
        self.events = Notifier()
        # Its own lock: connected() runs on the event loop, which must never wait on self.lock
        # (held across LLM calls and training).
        self._presence = threading.Lock()
        self._waiting: dict[str, int] = {}  # host -> open /intake/v1/wait requests
        self._gone: dict[str, float] = {}  # host -> monotonic time its last wait closed

        self.settings = RuntimeSettings.model_validate(control.get("settings", "server") or {})
        self.sources = {d["id"]: d for d in control.all("source")}
        if (legacy := control.get("stream", "all")) is not None and "picked" not in legacy:
            # The old catch-all stream: it keeps the sources it was catching, now as ticked ones.
            control.update("stream", "all", selectors=[], picked=sorted(
                sid for sid, src in self.sources.items() if not src.get("alias_of") and not self.streams_for(sid)))
        self._drop_empty_streams()
        self.tmeta = {int(d["id"]): d for d in control.all("template")}
        self.registry = (SchemaRegistry.load(self.registry_path) if self.registry_path.exists()
                         else SchemaRegistry())
        self.scorers: dict[str, LogBERTScorer] = {}
        for stream in control.all("stream"):
            model_dir = self._model_dir(stream["id"])
            if (model_dir / "config.json").exists():
                self.scorers[stream["id"]] = LogBERTScorer.load(model_dir)
        self._keys = {d["hash"]: d for d in control.all("key") if not d.get("revoked")}
        if bootstrap_key and hashlib.sha256(bootstrap_key.encode()).hexdigest() not in self._keys:
            self._store_key("bootstrap", bootstrap_key)
        self._build_pipeline()
        # Jobs left running by a stopped server will never finish.
        for job in control.all("job"):
            if job["status"] in ("queued", "running"):
                control.update("job", job["id"], status="failed", error="server restarted",
                               finished_at=now_iso())

        from ..ask.service import AskService

        self.groups = Groups(self)
        self.ask = AskService(self)
        self._stop = threading.Event()
        if start_threads:
            for target in (self._ingest_loop, self._job_loop, self._housekeeping_loop):
                threading.Thread(target=target, daemon=True, name=target.__name__).start()

    @property
    def adapter(self) -> LLMAdapter:
        if self._adapter is None:
            self._adapter = CountedAdapter(self.adapter_factory(), self.metrics)
        return self._adapter

    def _build_pipeline(self) -> None:
        """Before calibration: store everything as PENDING, learn nothing (the setup guide
        decides when calibration starts). After: online enrichment + per-stream detectors."""
        with self.lock:
            if self.settings.calibrated:
                self.pipeline = Pipeline.assemble(
                    self.registry, self.store, adapter=self.adapter,
                    queue=self.queue_factory() if self.queue_factory else None,
                    registry_path=self.registry_path, enrich_every=10**12, publish_lock=self.lock)
            else:
                self.pipeline = Pipeline(ParserHandle(self.registry.current), self.store)
            self.pipeline.hold = self._held
            self.detectors = {sid: self._detector(s) for sid, s in self.scorers.items()}

    def _held(self, event: LogEvent) -> bool:
        """A source that appeared after setup waits for its drafted templates to be reviewed,
        so the online learner doesn't learn its templates first."""
        if not self.settings.calibrated:
            return False
        src = self.sources.get(source_id(event.device_id, event.service, event.source))
        return src is None or not (src["calibrated"] or src.get("alias_of"))

    def _detector(self, scorer: LogBERTScorer) -> StreamingDetector:
        handle = self.pipeline.handle
        return StreamingDetector(
            scorer, DetectorConfig(window=self.window, stride=self.stride, record_windows=True),
            registry_version=lambda: handle.version,
            resolver=lambda line: self.registry.current.canonical_id(int(handle.parse(line).template_id)))

    def _model_dir(self, stream_id: str) -> Path:
        return self.data_dir / "models" / stream_id

    def _store_key(self, name: str, token: str) -> dict:
        digest = hashlib.sha256(token.encode()).hexdigest()
        doc = {"id": uuid.uuid4().hex[:12], "name": name, "hash": digest, "prefix": token[:7],
               "created_at": now_iso(), "revoked": False, "last_used_at": None}
        self.control.put("key", doc["id"], doc)
        self._keys[digest] = doc
        return doc

    def create_key(self, name: str) -> tuple[dict, str]:
        token = "lp_" + secrets.token_urlsafe(24)
        return self._store_key(name.strip() or "unnamed", token), token

    def revoke_key(self, key_id: str) -> None:
        doc = self.control.get("key", key_id)
        if doc is None:
            raise KeyError(key_id)
        self.control.update("key", key_id, revoked=True, revoked_at=now_iso())
        self._keys.pop(doc["hash"], None)

    def check_key(self, token: str | None) -> dict | None:
        if not token:
            return None
        doc = self._keys.get(hashlib.sha256(token.encode()).hexdigest())
        if doc is not None and (doc.get("last_used_at") or "")[:16] != now_iso()[:16]:
            doc["last_used_at"] = now_iso()  # minute resolution, persisted by housekeeping
            self._dirty.add(("key", doc["id"]))
        return doc

    def ingest(self, events: list[dict]) -> int:
        if self.inbox.qsize() + len(events) > self.inbox.maxsize:
            raise Busy(f"{self.inbox.qsize()} events waiting")
        for e in events:
            message = str(e.get("message", ""))
            if not message.strip():
                continue
            try:
                ts = datetime.fromisoformat(str(e.get("ts", "")))
            except ValueError:
                ts = None
            self.inbox.put_nowait(LogEvent(
                device_id=str(e.get("host") or "unknown")[:200], source=str(e.get("source") or "")[:200],
                line_no=0, line=message, service=str(e.get("service") or "")[:200],
                tags=tuple(str(t)[:200] for t in (e.get("tags") or [])[:50]), time=ts))
        return len(events)

    def _ingest_loop(self) -> None:
        while not self._stop.is_set():
            try:
                event = self.inbox.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.process(event)
            except Exception:
                self.metrics.count("errors")
                LOGGER.exception("failed to process a line from %s", event.device_id)

    def process(self, event: LogEvent) -> None:
        with self.lock:
            started = time.perf_counter_ns()
            result = self.pipeline.process(event)
            parse_ns, n_alerts = time.perf_counter_ns() - started, 0
            ts = self.pipeline._event_time or datetime.now(UTC)
            self.received += 1
            sid = source_id(event.device_id, event.service, event.source)
            src = self.sources.get(sid)
            if src is None:
                src = self.sources[sid] = {
                    "id": sid, "host": event.device_id, "service": event.service,
                    "source": event.source, "first_seen": now_iso(), "count": 0,
                    "calibrated": False, "alias_of": None}
                self._assign(sid)
                if self.settings.calibrated:
                    self._warn("new_source", sid, f"New log source {event.service or event.source} on {event.device_id}",
                               "Templates are drafted from a sample once it has collected enough lines; "
                               "you review them before they reach the parser.",
                               streams=[s["id"] for s in self.streams_for(sid)])
            src["count"] += 1
            src["last_seen"] = now_iso()
            self._dirty.add(("source", sid))

            tid = self.registry.current.canonical_id(int(result.template_id))
            if tid != UNK_TEMPLATE_ID:
                meta = self.tmeta.setdefault(tid, {"id": str(tid), "first_seen": now_iso(), "count": 0})
                meta["count"] += 1
                meta["last_seen"] = now_iso()
                self._dirty.add(("template", str(tid)))

            device = f"{event.device_id}/{event.service}"
            for stream in self.streams_for(sid):
                detector = self.detectors.get(stream["id"])
                if detector is None:
                    continue
                alerts = detector.observe(device, tid, ts, event.line)
                if alerts:
                    n_alerts += len(alerts)
                    self.store.insert_alerts(alerts)
                if detector.scored:  # drained every line, so it never grows
                    self.drift.observe(stream["id"], detector.scored)
                    detector.scored.clear()
                if tid != UNK_TEMPLATE_ID and tid not in detector.scorer.vocab:
                    self._novel(tid, stream["id"])
            self.metrics.parsed(parse_ns, tid == UNK_TEMPLATE_ID, n_alerts)

    def streams_for(self, sid: str) -> list[dict]:
        src = self.sources.get(sid)
        target = self.sources.get(src["alias_of"]) if src and src.get("alias_of") else src
        if target is None:
            return []
        # One stream per source, so it's always clear which model watches it: the stream it's
        # ticked in, else the oldest one whose pattern (for future sources) matches.
        streams = sorted(self.control.all("stream"), key=lambda s: s["created_at"])
        found = (next((s for s in streams if target["id"] in (s.get("picked") or [])), None)
                 or next((s for s in streams if any(
                     sel and _matches(sel, target["host"], target["service"], target["source"])
                     for sel in s.get("selectors") or [])), None))
        return [found] if found else []

    def _assign(self, sid: str, avoid: str | None = None) -> None:
        """A source without a stream joins the stream of a source like it (same service and type,
        any host; never ``avoid``, the one it was just taken out of), else gets one of its own."""
        if self.streams_for(sid):  # a pattern already takes it
            return
        src = self.sources[sid]
        for other_id, other in self.sources.items():
            if other_id != sid and (other["service"], other["source"]) == (src["service"], src["source"]) \
                    and (home := self.streams_for(other_id)) and home[0]["id"] != avoid:
                self.control.update("stream", home[0]["id"], picked=sorted({*(home[0].get("picked") or []), sid}))
                return
        name = src["service"] or src["source"] or src["host"]
        if any(s["name"] == name for s in self.control.all("stream")):
            name = f"{name} ({src['source'] or src['host']})"
        stream_id = self._stream_id(name)
        self.control.put("stream", stream_id, {"id": stream_id, "name": name, "picked": [sid], "selectors": [],
                                               "retrain_days": None, "created_at": now_iso()})
        self.events.notify()

    def _stream_id(self, name: str) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "stream"
        stream_id, n = base, 2
        while self.control.get("stream", stream_id) is not None:
            stream_id, n = f"{base}-{n}", n + 1
        return stream_id

    def _drop_empty_streams(self) -> None:
        """A stream no source is in (and no pattern waits for) has nothing to train: remove it."""
        homes = {s[0]["id"] for sid in self.sources if (s := self.streams_for(sid))}
        for stream in self.control.all("stream"):
            if stream["id"] not in homes and not stream.get("selectors"):
                self.delete_stream(stream["id"])

    def _warn(self, kind: str, subject: str, title: str, detail: str, streams: list[str]) -> dict:
        wid = f"{kind}:{subject}"
        doc = self.control.get("warning", wid)
        if doc is None:
            doc = {"id": wid, "kind": kind, "subject": subject, "title": title, "detail": detail,
                   "first_seen": now_iso(), "status": "open", "streams": streams}
            self.control.put("warning", wid, doc)
        return doc

    def _novel(self, tid: int, stream_id: str) -> None:
        """A template the stream's model has never seen: a novelty signal, not an anomaly."""
        wid = f"novel_template:{tid}"
        doc = self.control.get("warning", wid)
        if doc is None:
            entry = self.registry.current.templates.get(tid)
            text = " ".join(entry.tokens) if entry else f"#{tid}"
            self._warn("novel_template", str(tid), f"Template #{tid} is unseen by LogBERT",
                       text, streams=[stream_id])
        elif doc["status"] == "open" and stream_id not in doc["streams"]:
            self.control.update("warning", wid, streams=[*doc["streams"], stream_id])

    def warnings(self) -> list[dict]:
        out = []
        for w in self.control.all("warning"):
            if w["kind"] == "novel_template":
                meta = self.tmeta.get(int(w["subject"]), {})
                w |= {"count": meta.get("count", 0), "last_seen": meta.get("last_seen")}
            elif w["kind"] == "new_source":
                src = self.sources.get(w["subject"], {})
                w |= {"count": src.get("count", 0), "last_seen": src.get("last_seen")}
            out.append(w)
        return sorted(out, key=lambda w: (w["status"] != "open", w["first_seen"]), reverse=False)

    def resolve_warning(self, wid: str, note: str = "marked resolved") -> dict:
        if self.control.get("warning", wid) is None:
            raise KeyError(wid)
        return self.control.update("warning", wid, status="resolved", resolution=note,
                                   resolved_at=now_iso())

    def same_as(self, wid: str, target: str) -> dict:
        """The new source behaves like ``target`` (joins its streams), or the template is the
        same event as template ``target``."""
        w = self.control.get("warning", wid)
        if w is None:
            raise KeyError(wid)
        with self.lock:
            if w["kind"] == "new_source":
                if target not in self.sources or target == w["subject"]:
                    raise ValueError(f"unknown source {target!r}")
                self.sources[w["subject"]].update(alias_of=target, calibrated=True)
                self._dirty.add(("source", w["subject"]))
                for s in self.control.all("stream"):  # it now follows the target's stream
                    if w["subject"] in (s.get("picked") or []):
                        self.control.update("stream", s["id"], picked=[p for p in s["picked"] if p != w["subject"]])
                self._drop_empty_streams()
            else:
                snapshot = self.registry.alias(int(w["subject"]), int(target))
                self._linked(int(w["subject"]), "you")
                self.registry.save(self.registry_path)
                self.pipeline.handle.swap(snapshot)
        return self.resolve_warning(wid, f"same as {target}")

    def recalibrate_warning(self, wid: str) -> dict:
        w = self.control.get("warning", wid)
        if w is None:
            raise KeyError(wid)
        if w["kind"] == "new_source":
            src = self.sources.get(w["subject"], {})
            if src.get("count", 0) < self.config.server.min_draft_lines:
                raise ValueError(f"only {src.get('count', 0)} lines so far, need {self.config.server.min_draft_lines}")
            return self.enqueue("draft", sources=[w["subject"]])
        return self.train(w["streams"])

    def enqueue(self, kind: str, **params) -> dict:
        if kind not in JOB_KINDS:
            raise ValueError(f"unknown job kind {kind!r}")
        active = [j for j in self.control.all("job") if j["status"] in ("queued", "running")]
        for j in active:
            if j["kind"] == kind and j["params"] == params:
                return j
        doc = {"id": uuid.uuid4().hex[:12], "kind": kind, "params": params, "status": "queued",
               "progress": 0.0, "message": "waiting", "created_at": now_iso(), "log": []}
        self.control.put("job", doc["id"], doc)
        self.jobs.put(doc["id"])
        return doc

    def cancel_job(self, job_id: str) -> dict:
        job = self.control.get("job", job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] != "queued":
            raise ValueError(f"only queued jobs can be cancelled; this one is {job['status']}")
        return self.control.update("job", job_id, status="cancelled", finished_at=now_iso())

    def retry_job(self, job_id: str) -> dict:
        job = self.control.get("job", job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] not in ("failed", "cancelled"):
            raise ValueError(f"only failed or cancelled jobs can be retried; this one is {job['status']}")
        return self.enqueue(job["kind"], **job["params"])

    def _job_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self.jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            self.run_job(job_id)

    def run_job(self, job_id: str) -> dict:
        job = self.control.get("job", job_id)
        assert job is not None
        if job["status"] == "cancelled":
            return job
        self.control.update("job", job_id, status="running", started_at=now_iso())
        log: list[list[str]] = []
        detail: dict = {}  # section -> what went in / came out, for the UI's detail panel

        def progress(fraction: float, message: str, more: dict | None = None) -> None:
            if not log or log[-1][1] != message:
                log.append([now_iso(), message])
                del log[:-300]
            detail.update(more or {})
            self.control.update("job", job_id, progress=round(min(max(fraction, 0.0), 1.0), 3),
                                message=message, log=log, detail=detail)

        try:
            note = getattr(self, f"_job_{job['kind']}")(progress, **job["params"])
        except Exception as error:
            LOGGER.exception("job %s failed", job_id)
            return self.control.update("job", job_id, status="failed", error=str(error),
                                       finished_at=now_iso())
        message = note or "done"
        if message != (log[-1][1] if log else ""):
            log.append([now_iso(), message])
        return self.control.update("job", job_id, status="done", progress=1.0, message=message,
                                   log=log, finished_at=now_iso())

    def _sample(self, sids: list[str], progress, start: float, end: float) -> tuple[list[str], list[str]]:
        """(sample, recent): up to ``sample_cap`` lines per source picked by entropy-greedy
        priority sampling from the 4x most recent, so one chatty source can't crowd out the
        rest; and each source's ``sample_cap`` most recent lines, which reviewers' counts use."""
        cap = self.settings.sample_cap
        lines: list[str] = []
        pool: list[str] = []
        for i, sid in enumerate(sids):
            src = self.sources[sid]
            recent = self.reads.source_lines(src["host"], src["service"], src["source"], cap * 4)
            lines += [recent[j] for j in sorted(entropy_greedy_sample(recent, cap))]
            pool += recent[-cap:]
            progress(start + (end - start) * (i + 1) / len(sids), f"sampled {src['service'] or sid}")
        return lines, pool

    def _lines_path(self, batch_id: str) -> Path:
        return self.data_dir / "review" / f"{batch_id}.json"

    def batch_lines(self, batch_id: str) -> list[str]:
        path = self._lines_path(batch_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def _masks(self, batch: dict) -> list[Mask]:
        return [Mask.from_dict(m) for m in batch["masks"]] if batch["initial"] else list(self.registry.current.masks)

    def generate_drafts(self, sample_cap: int | None = None) -> dict:
        """First calibration: draft templates from every source for the setup guide's review."""
        if self.settings.calibrated:
            raise ValueError("already calibrated; new sources are drafted on their own")
        if sample_cap is not None:
            self.update_settings({"sample_cap": sample_cap})
        total = sum(s["count"] for s in self.sources.values())
        if total < self.config.server.min_draft_lines:
            raise ValueError(f"only {total} lines collected so far, need {self.config.server.min_draft_lines}")
        return self.enqueue("draft", sources=[])

    def _job_draft(self, progress, sources: list[str]) -> str:
        initial = not self.settings.calibrated
        sids = sources or [sid for sid, s in self.sources.items() if not s.get("alias_of")]
        if not sids:
            raise ValueError("no log sources yet: connect an agent and track some files first")
        lines, pool = self._sample(sids, progress, 0.0, 0.3)
        report: dict = {}
        adapter = _Traced(self.adapter, progress, 0.35, 0.88)
        progress(0.32, f"{len(lines)} lines sampled", {"Input": {
            "sources": sids, "sampled lines": len(lines), "sample (first 50)": lines[:50]}})
        if initial:
            progress(0.35, f"mining templates from {len(lines)} lines (masks, Drain, field names)")
            result = calibrate(lines, adapter)
            entries = sorted(result.snapshot.templates.values(), key=lambda e: e.template_id)
            templates = [NewTemplate(e.tokens, e.keys) for e in entries]
            masks = list(result.snapshot.masks)
            report = {"holdout_coverage": result.report.holdout_coverage}
        else:
            progress(0.35, f"mining templates for lines the parser can't match ({len(lines)} sampled)")
            templates = append_drafts(self.registry, lines, adapter)
            masks = list(self.registry.current.masks)
        progress(0.9, f"{len(templates)} templates drafted; counting matches", {
            "LLM calls": adapter.calls[-50:],
            "Output": {"templates": [_template_text(t.tokens, t.keys) for t in templates],
                       "masks": [m.to_dict() for m in masks], **report}})
        batch_id = uuid.uuid4().hex[:10]
        path = self._lines_path(batch_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pool), encoding="utf-8")
        drafts = review.draft_docs(batch_id, templates, masks, pool)
        for d in drafts:
            self.control.put("draft", d["id"], d)
        self.control.put("batch", batch_id, {
            "id": batch_id, "initial": initial, "sources": sids, "status": "review",
            "masks": [m.to_dict() for m in masks] if initial else [], "lines": len(pool),
            "sampled": len(lines),
            "report": report, "created_at": now_iso()})
        # A newer sample replaces an unreviewed one; only after the new one exists, so there is
        # never a moment without an open review.
        for old in self.control.all("batch"):
            if old["id"] != batch_id and old["status"] == "review" and (
                    old["initial"] if initial else set(old["sources"]) & set(sids)):
                self.control.update("batch", old["id"], status="replaced")
        if not drafts and not initial:  # the parser already matches every sampled line
            self._finish_batch(self.control.get("batch", batch_id) or {}, [],
                               "every sampled line already matches a template")
            return "no new templates needed"
        self.events.notify()
        return f"{len(drafts)} templates ready for review"

    def batches(self, status: str = "review") -> list[dict]:
        out = []
        for b in self.control.all("batch"):
            if b["status"] == status:
                drafts = self.drafts(b["id"])
                out.append(b | {"drafts": len(drafts), "pending": sum(d["status"] == "pending" for d in drafts),
                                "approved": sum(d["status"] == "approved" for d in drafts)})
        return sorted(out, key=lambda b: b["created_at"], reverse=True)

    def drafts(self, batch_id: str) -> list[dict]:
        return sorted((d for d in self.control.all("draft") if d["batch"] == batch_id),
                      key=lambda d: (-d["count"], d["id"]))

    def _open_batch(self, batch_id: str) -> dict:
        batch = self.control.get("batch", batch_id)
        if batch is None:
            raise KeyError(batch_id)
        if batch["status"] != "review":
            raise ValueError(f"this review is already {batch['status']}")
        return batch

    def check_draft(self, batch_id: str, tokens: list[str], keys: list[str]) -> dict:
        batch = self.control.get("batch", batch_id)
        if batch is None:
            raise KeyError(batch_id)
        return review.check(tokens, keys, self._masks(batch), self.batch_lines(batch_id))

    def check_draft_rule(self, draft_id: str, rule: str) -> dict:
        """A rule against the batch's recent lines, plus this draft's lines it would miss."""
        draft = self.control.get("draft", draft_id)
        if draft is None:
            raise KeyError(draft_id)
        batch = self.control.get("batch", draft["batch"]) or {}
        return review.check_rule(rule, self.batch_lines(draft["batch"]),
                                 review.mined_matcher(draft, self._masks(batch)))

    def update_draft(self, draft_id: str, body: dict) -> dict:
        draft = self.control.get("draft", draft_id)
        if draft is None:
            raise KeyError(draft_id)
        batch = self._open_batch(draft["batch"])
        changes: dict = {}
        if "rule" in body:
            checked = self.check_draft_rule(draft_id, str(body["rule"]))
            changes = {"rule": compile_rule(str(body["rule"])).text, "rule_edited": True, "edited": True,
                       "tokens": checked["tokens"], "keys": checked["keys"],
                       "count": checked["count"], "samples": checked["samples"]}
        elif "tokens" in body or "keys" in body:
            tokens = [str(t) for t in body.get("tokens", draft["tokens"])]
            keys = [str(k) for k in body.get("keys", draft["keys"])]
            if "tokens" in body and "keys" not in body:  # slots changed: keep names that still fit
                keys = clean_keys(draft["keys"], tokens)
            changes = review.check(tokens, keys, self._masks(batch), self.batch_lines(batch["id"]))
            changes |= {"tokens": tokens, "keys": keys, "edited": True,
                        "rule": rule_from_template(tokens, keys), "rule_edited": False}
            changes.pop("checked")
        if "status" in body:
            if body["status"] not in ("pending", "approved", "rejected"):
                raise ValueError("status must be pending, approved or rejected")
            changes["status"] = body["status"]
        return self.control.update("draft", draft_id, **changes)

    def decide_all(self, batch_id: str, status: str, ids: list[str] | None = None) -> int:
        """Every undecided draft, or with ``ids`` exactly those (back to "pending" too)."""
        self._open_batch(batch_id)
        if status not in ("approved", "rejected") and not (ids is not None and status == "pending"):
            raise ValueError("status must be approved or rejected")
        pending = [d for d in self.drafts(batch_id) if d["status"] != "merged"
                   and (d["id"] in ids if ids is not None else d["status"] == "pending")]
        for d in pending:
            self.control.update("draft", d["id"], status=status)
        return len(pending)

    def merge_drafts(self, target_id: str, others: list[str]) -> dict:
        target = self.control.get("draft", target_id)
        if target is None:
            raise KeyError(target_id)
        batch = self._open_batch(target["batch"])
        docs = [d for o in others if o != target_id
                if (d := self.control.get("draft", o)) is not None and d["batch"] == batch["id"]]
        if not docs or len(docs) != len(set(others) - {target_id}):
            raise ValueError("pick at least one other draft from the same review")
        tokens = target["tokens"]
        for d in docs:
            tokens = review.merge_tokens(tokens, d["tokens"])
        keys = clean_keys(target["keys"], tokens)
        checked = review.check(tokens, keys, self._masks(batch), self.batch_lines(batch["id"]))
        for d in docs:
            self.control.update("draft", d["id"], status="merged", merged_into=target_id)
        return self.control.update("draft", target_id, tokens=tokens, keys=keys, edited=True,
                                   rule=rule_from_template(tokens, keys), rule_edited=False,
                                   count=checked["count"], samples=checked["samples"],
                                   status="approved" if target["status"] == "pending" else target["status"])

    def apply_batch(self, batch_id: str) -> dict:
        batch = self._open_batch(batch_id)
        drafts = self.drafts(batch_id)
        pending = sum(d["status"] == "pending" for d in drafts)
        if pending:
            raise ValueError(f"{pending} templates still need approve or reject")
        if batch["initial"] and not any(d["status"] == "approved" for d in drafts):
            raise ValueError("approve at least one template before calibrating")
        return self.enqueue("apply", batch=batch_id)

    def discard_batch(self, batch_id: str) -> dict:
        """Add nothing; the sources are left to the online learner and stop being drafted."""
        batch = self._open_batch(batch_id)
        if batch["initial"]:
            raise ValueError("the first calibration can't be skipped; regenerate it instead")
        done = self._finish_batch(batch, [], "discarded: left to the online learner", status="discarded")
        if self.pipeline.worker is not None:
            self.pipeline.worker.rehydrate()
        return done

    def _finish_batch(self, batch: dict, ids: list[int], note: str, status: str = "applied") -> dict:
        with self.lock:
            for sid in batch["sources"]:
                if sid in self.sources:
                    self.sources[sid]["calibrated"] = True
                    self._dirty.add(("source", sid))
        done = self.control.update("batch", batch["id"], status=status, template_ids=ids, note=note,
                                   finished_at=now_iso())
        for sid in batch["sources"]:
            wid = f"new_source:{sid}"
            if (w := self.control.get("warning", wid)) and w["status"] == "open":
                self.resolve_warning(wid, note)
        self.events.notify()
        return done

    def _job_apply(self, progress, batch: str) -> str:
        doc = self._open_batch(batch)
        approved = [d for d in self.drafts(batch) if d["status"] == "approved"]
        templates = [NewTemplate(tuple(d["tokens"]), tuple(d["keys"]),
                                 rule=d["rule"] if d.get("rule_edited") else None) for d in approved]
        vocab = [k for t in templates for k in t.keys]
        progress(0.05, f"publishing {len(templates)} approved templates")
        with self.lock:
            if doc["initial"]:
                registry = SchemaRegistry()
                _, ids = registry.publish(templates, masks=self._masks(doc), key_vocab=vocab)
                self.registry = registry
                self.settings = self.settings.model_copy(update={"calibrated": True, "calibrated_at": now_iso(), "setup_done": True})
                self.control.put("settings", "server", self.settings.model_dump())
            else:
                ids = self.registry.publish(templates, key_vocab=vocab)[1] if templates else []
            self.registry.save(self.registry_path)
            for tid in ids:
                self.tmeta.setdefault(tid, {"id": str(tid), "first_seen": now_iso(), "count": 0})["reviewed"] = True
                self._dirty.add(("template", str(tid)))
            if doc["initial"]:
                self._build_pipeline()
            else:
                self.pipeline.handle.swap(self.registry.current)
        for d, tid in zip(approved, ids):
            self.control.update("draft", d["id"], template_id=tid)
        self._finish_batch(doc, ids, f"{len(ids)} templates added (registry v{self.registry.current.version})")
        progress(0.3, f"registry v{self.registry.current.version}; backfilling stored lines")
        self._backfill()
        if self.pipeline.worker is not None:
            # What the review didn't cover (incl. a new source's held lines) is learned online.
            self.pipeline.worker.rehydrate()
        streams = (sorted(s["id"] for s in self.control.all("stream")) if doc["initial"] else
                   sorted({s["id"] for sid in doc["sources"] for s in self.streams_for(sid)}))
        waiting = self._retrain(streams, progress, 0.5)
        return f"{len(ids)} templates added" + (f"; {waiting} waiting for data" if waiting else "")

    # Templates the online learner added without review: listed so a person can keep, correct
    # or merge them; they go live at once so parsing never waits on a person.
    def auto_templates(self) -> list[dict]:
        if not self.settings.calibrated:
            return []
        snap = self.registry.current
        return [{"template_id": t.template_id, "tokens": list(t.tokens), "keys": list(t.keys),
                 "rule": t.rule or rule_from_template(t.tokens, t.keys), "rule_edited": bool(t.rule),
                 "count": self.tmeta.get(t.template_id, {}).get("count", 0),
                 "first_seen": self.tmeta.get(t.template_id, {}).get("first_seen")}
                for t in sorted(snap.templates.values(), key=lambda t: t.template_id)
                if t.alias_of is None and not self.tmeta.get(t.template_id, {}).get("reviewed")]

    def review_template(self, tid: int, body: dict) -> dict:
        """``{"keep": true}``, ``{"rule": "..."}``, ``{"tokens": [...], "keys": [...]}`` or
        ``{"same_as": id}``. A new rule also re-extracts the template's stored lines (a job)."""
        snap = self.registry.current
        if tid not in snap.templates:
            raise KeyError(tid)
        job = None
        with self.lock:
            if "rule" in body:
                rule = compile_rule(str(body["rule"]))
                snapshot = self.registry.replace(tid, rule.tokens, rule.keys, rule.text)
                job = True
            elif "same_as" in body:
                snapshot = self.registry.alias(tid, int(body["same_as"]))
                self._linked(tid, "you")
            elif "tokens" in body or "keys" in body:
                entry = snap.templates[tid]
                tokens = [str(t) for t in body.get("tokens", entry.tokens)]
                keys = ([str(k) for k in body["keys"]] if "keys" in body
                        else clean_keys(list(entry.keys), tokens))
                review.validate(tokens, keys)
                snapshot = self.registry.replace(tid, tokens, keys)
            else:
                snapshot = None
            if snapshot is not None:
                self.registry.save(self.registry_path)
                self.pipeline.handle.swap(snapshot)
            meta = self.tmeta.setdefault(tid, {"id": str(tid), "first_seen": now_iso(), "count": 0})
            meta["reviewed"] = True
            self._dirty.add(("template", str(tid)))
        if job:
            job = self.enqueue("reextract", template=tid)
        return {"template_id": tid, "reviewed": True, "job": job}

    def enrich(self) -> None:
        """One online-learner cycle; remembers how each link it made was decided."""
        report = self.pipeline.enrich()
        if report is None:
            return
        with self.lock:
            for tid in report.aliased:
                reason = report.alias_decisions.get(tid, (0.0, ""))[1]
                meta = self.tmeta.setdefault(tid, {"id": str(tid), "first_seen": now_iso(), "count": 0})
                # only a reworded line needs the LLM; a changed variable is linked by the pipeline itself
                meta.update(linked_by="llm" if "LLM" in reason else "pipeline", link_reason=reason, linked_at=now_iso())
                self._dirty.add(("template", str(tid)))

    def _linked(self, tid: int, by: str | None) -> None:
        meta = self.tmeta.setdefault(tid, {"id": str(tid), "first_seen": now_iso(), "count": 0})
        meta.update(linked_by=by, linked_at=now_iso() if by else None)
        self._dirty.add(("template", str(tid)))

    def link_template(self, tid: int, target: int) -> dict:
        """``tid`` is the same event as ``target``: models count its lines as ``target``."""
        with self.lock:
            snapshot = self.registry.alias(tid, target)
            self.registry.save(self.registry_path)
            self.pipeline.handle.swap(snapshot)
            self._linked(tid, "you")
        return {"template_id": tid, "alias_of": target}

    def unlink_template(self, tid: int) -> dict:
        with self.lock:
            snapshot = self.registry.unalias(tid)
            self.registry.save(self.registry_path)
            self.pipeline.handle.swap(snapshot)
            self._linked(tid, None)
        return {"template_id": tid, "alias_of": None}

    def check_template_rule(self, tid: int, rule: str) -> dict:
        """A rule against this template's stored lines (what it extracts, what it misses)."""
        if tid not in self.registry.current.templates:
            raise KeyError(tid)
        lines = [r["message"] for r in self.reads.recent({"template_id": tid}, 500)]
        return review.check_rule(rule, lines, own=lambda _line: True)

    def _job_reextract(self, progress, template: int) -> str:
        entry = self.registry.current.templates.get(template)
        if entry is None or not entry.rule:
            raise ValueError(f"template {template} has no parsing rule")
        rule = compile_rule(entry.rule)
        progress(0.05, f"reading stored lines of template #{template}")
        rows = self.store.fetch_pending(1_000_000, template_id=template)
        out = []
        for i, row in enumerate(rows):
            fields = rule.extract(row.raw_message)
            if fields is not None:
                out.append(replace(row, payload=fields, version=row.version + 1))
            if i % 5000 == 0:
                progress(0.1 + 0.6 * i / max(len(rows), 1), f"re-extracted {i} of {len(rows)} lines")
        for i in range(0, len(out), 10_000):
            self.store.insert_logs(out[i:i + 10_000])
        self.store.flush()
        return f"{len(out)} of {len(rows)} stored lines re-extracted with the new rule"

    def train(self, streams: list[str]) -> dict:
        """Retrain now if there is enough data; otherwise the stream waits and trains by itself."""
        ready, waiting = [], []
        for sid in streams:
            have = self._window_count(sid)
            if have >= self.config.server.min_train_windows:
                ready.append(sid)
            else:
                self._wait(sid, have)
                waiting.append(sid)
        if not ready:
            return {"waiting": waiting}
        return self.enqueue("retrain", streams=ready)

    def _wait(self, stream_id: str, have: int) -> None:
        self.control.update("stream", stream_id, waiting=True, model_error=None,
                            data={"have": have, "need": self.config.server.min_train_windows, "at": now_iso()})
        self.events.notify()

    def _job_retrain(self, progress, streams: list[str], warning: str | None = None) -> str:
        waiting = self._retrain(streams, progress, 0.0)
        return f"{waiting} of {len(streams)} streams waiting for data" if waiting else "done"

    def _retrain(self, streams: list[str], progress, start: float) -> int:
        """Train each stream; one without enough data waits instead of failing. Returns how many wait."""
        waiting = 0
        for i, sid in enumerate(streams):
            lo = start + (1 - start) * i / max(len(streams), 1)
            hi = start + (1 - start) * (i + 1) / max(len(streams), 1)
            try:
                self._train(sid, lambda f, m, d=None, lo=lo, hi=hi: progress(lo + (hi - lo) * f, m, d))
            except NotEnoughData as short:
                self._wait(sid, short.have)
                progress(hi, f"{sid}: waiting for data ({short.have}/{short.need} windows); trains by itself")
                waiting += 1
        return waiting

    def _backfill(self, limit: int = 1_000_000) -> int:
        # shortcut: one page of PENDING rows; page by timestamp if a backlog ever exceeds a million rows.
        rows = self.store.fetch_pending(limit)
        handle = self.pipeline.handle
        resolved = []
        for row in rows:
            result = handle.parse(row.raw_message)
            if result.status == Status.FINAL:
                resolved.append(row.resolved(result))
        with self.lock:  # back-filled lines count toward their template like live ones
            for row in resolved:
                tid = self.registry.current.canonical_id(row.template_id)
                meta = self.tmeta.setdefault(tid, {"id": str(tid), "first_seen": row.timestamp.isoformat(), "count": 0})
                meta["count"] += 1
                meta["first_seen"] = min(meta["first_seen"], row.timestamp.isoformat())
                meta.setdefault("last_seen", now_iso())
                self._dirty.add(("template", str(tid)))
        for i in range(0, len(resolved), 10_000):
            self.store.insert_logs(resolved[i:i + 10_000])
        self.store.flush()
        return len(resolved)

    def _windows(self, stream_id: str) -> tuple[list[list[int]], int]:
        """(training windows, sources they come from): capped, per-source sequences of canonical
        template ids."""
        cap = self.settings.sample_cap
        snapshot = self.registry.current
        sequences: list[list[int]] = []
        sids = [sid for sid in self.sources
                if any(s["id"] == stream_id for s in self.streams_for(sid))]
        for sid in sids:
            src = self.sources[sid]
            ids = [snapshot.canonical_id(int(t)) for _, t in
                   self.reads.source_rows(src["host"], src["service"], src["source"], cap)]
            ids = [t for t in ids if t != UNK_TEMPLATE_ID]
            if ids:
                sequences.append(ids)
        windows = [w for seq in sequences for w in sliding_windows(seq, self.window, self.stride)]
        # Incidents marked false positive: normal after all, so the next model learns them.
        windows += [[snapshot.canonical_id(int(t)) for t in w] for d in self.control.all("incident")
                    if d.get("status") == "false_positive" and d.get("stream") == stream_id
                    for w in d.get("windows") or []]
        return windows, len(sequences)

    def _window_count(self, stream_id: str) -> int:
        return len(self._windows(stream_id)[0]) if self.settings.calibrated else 0

    def _train(self, stream_id: str, progress) -> None:
        """Train this stream's LogBERT."""
        stream = self.control.get("stream", stream_id)
        if stream is None:
            raise KeyError(stream_id)
        windows, n_sources = self._windows(stream_id)
        if len(windows) < self.config.server.min_train_windows:
            raise NotEnoughData(stream["name"], len(windows), self.config.server.min_train_windows)
        random.Random(7).shuffle(windows)
        n_valid = max(1, len(windows) // 10)
        vocab = TemplateVocab.build(windows)
        encoded = [vocab.encode(w) for w in windows]
        # Enough optimizer steps to learn the order of events whatever the amount of data: a few
        # minutes of logs are ~2 batches an epoch, and a handful of epochs left LogBERT at chance
        # (alerting on normal traffic). Patience scales so a tiny validation set can't stop it early.
        n_train = len(encoded) - n_valid
        batch = 16 if n_train < 2000 else 64
        epochs = min(300, max(self.epochs, math.ceil(self.min_train_steps / math.ceil(n_train / batch))))
        train_cfg = TrainConfig(epochs=epochs, batch_size=batch, lr=3e-3 if batch == 16 else 1e-3,
                                patience=max(5, epochs // 4))
        config = LogBERTConfig(len(vocab), 32, 1, 2, max_len=self.window + 1)
        name = stream["name"]
        progress(0.05, f"{name}: {len(windows)} windows from {n_sources} sources", {f"{name}: input": {
            "windows (train / valid)": f"{len(windows) - n_valid} / {n_valid}", "sources": n_sources,
            "templates in vocab": len(vocab), "window / stride": f"{self.window} / {self.stride}",
            "LogBERT config": config.to_dict(),
            "epochs / batch / lr": f"{epochs} / {batch} / {train_cfg.lr}", "example windows (template ids)": [" ".join(map(str, w)) for w in windows[:5]]}})
        epoch_log: list[str] = []

        def log(message: str) -> None:
            if message.startswith("epoch "):
                epoch = int(message.split()[1].rstrip(":"))
                epoch_log.append(message)
                progress(0.1 + 0.8 * (epoch + 1) / epochs, f"{name}: {message}", {f"{name}: training": epoch_log})

        trained = train_logbert(encoded[n_valid:], encoded[:n_valid], config, train_cfg, log=log)
        version = f"{stream_id}-{datetime.now(UTC):%Y%m%d%H%M%S}"
        scorer = LogBERTScorer.from_training(trained, vocab, settings=ScorerSettings(num_candidates=3),
                                             version=version)
        # No labelled anomalies here: flag sequences that miss more than 99% of held-out normal
        # windows do, and never alert on novelty alone (unseen templates become warnings).
        misses = [s.miss_ratio for s in scorer.score(windows[:n_valid])]
        # shortcut: a percentile threshold without labels; tune with labelled incidents when there are some.
        miss = max(0.3, float(np.quantile(misses, 0.99))) if misses else 0.5
        scorer.settings.thresholds = Thresholds(miss_ratio=miss, novelty_ratio=float("inf"))
        scorer.save(self._model_dir(stream_id))
        progress(0.95, f"{name}: threshold set", {f"{name}: output": {
            "model version": version, "miss-ratio threshold": round(miss, 3),
            "held-out miss ratio p50 / p99": (f"{np.quantile(misses, 0.5):.3f} / {np.quantile(misses, 0.99):.3f}"
                                              if misses else "–")}})
        with self.lock:
            self.scorers[stream_id] = scorer
            self.detectors[stream_id] = self._detector(scorer)
            self.drift.reset(stream_id)
        self.control.update("stream", stream_id, trained_at=now_iso(), model_version=version,
                            vocab=sorted(vocab.to_json()), windows=len(windows),
                            miss_threshold=round(miss, 3), model_error=None, waiting=False,
                            miss_baseline=histogram(misses) if misses else None,
                            data={"have": len(windows), "need": self.config.server.min_train_windows, "at": now_iso()})
        for w in self.control.all("warning"):
            if (w["kind"] == "novel_template" and w["status"] == "open"
                    and int(w["subject"]) in vocab and stream_id in w["streams"]):
                rest = [s for s in w["streams"] if s != stream_id]
                if rest:
                    self.control.update("warning", w["id"], streams=rest)
                else:
                    self.resolve_warning(w["id"], f"learned by {version}")
        progress(1.0, f"{stream['name']}: model {version} ready")

    def update_settings(self, body: dict) -> dict:
        self.settings = self.settings.changed(body)
        self.control.put("settings", "server", self.settings.model_dump())
        return self.settings.model_dump()

    def due_drafts(self) -> list[str]:
        """New sources that collected enough lines (or waited long enough) for a template sample."""
        if not self.settings.calibrated:
            return []
        busy = {sid for b in self.control.all("batch") if b["status"] == "review" for sid in b["sources"]}
        busy |= {sid for j in self.control.all("job") if j["kind"] == "draft"
                 and j["status"] in ("queued", "running") for sid in j["params"]["sources"]}
        cutoff = (datetime.now(UTC) - timedelta(minutes=self.settings.draft_after_minutes)).isoformat()
        return sorted(sid for sid, s in self.sources.items()
                      if not s["calibrated"] and not s.get("alias_of") and sid not in busy
                      and (s["count"] >= self.settings.draft_lines
                           or (s["count"] >= self.config.server.min_draft_lines and s["first_seen"] <= cutoff)))

    def collecting(self) -> list[dict]:
        """New sources still gathering lines before their templates are drafted."""
        if not self.settings.calibrated:
            return []
        busy = {sid for b in self.control.all("batch") if b["status"] == "review" for sid in b["sources"]}
        need = self.settings.draft_lines
        return [{"id": sid, "host": s["host"], "service": s["service"], "source": s["source"],
                 "have": s["count"], "need": need, "first_seen": s["first_seen"],
                 "draft_by": (datetime.fromisoformat(s["first_seen"])
                              + timedelta(minutes=self.settings.draft_after_minutes)).isoformat()}
                for sid, s in sorted(self.sources.items())
                if not s["calibrated"] and not s.get("alias_of") and sid not in busy]

    def refresh_readiness(self) -> None:
        """Count each waiting stream's training windows; start training the ones that have enough."""
        for stream in self.control.all("stream"):
            if not stream.get("waiting"):
                continue
            have = self._window_count(stream["id"])
            self.control.update("stream", stream["id"],
                                data={"have": have, "need": self.config.server.min_train_windows, "at": now_iso()})
            if have >= self.config.server.min_train_windows:
                self.enqueue("retrain", streams=[stream["id"]])

    def maintenance(self) -> dict:
        warnings = [w for w in self.warnings() if w["status"] == "open"]
        streams = {s["id"]: s for s in self.control.all("stream")}
        for w in warnings:
            if w["kind"] == "novel_template":
                w["readiness"] = [{"stream": sid, "name": streams[sid]["name"], **(streams[sid].get("data") or {})}
                                  for sid in w["streams"] if sid in streams]
        return {
            "review": self.batches(),
            "auto_templates": len(self.auto_templates()),
            "collecting": self.collecting(),
            "waiting": [s for s in streams.values() if s.get("waiting")],
            "warnings": warnings,
        }

    def incidents(self, limit: int = 2000) -> list[dict]:
        """Alerts grouped: one service's alerts less than ``incident_gap_minutes`` apart are one incident,
        named by its first alert, newest first, with what a person decided about it."""
        groups: dict[str, dict] = {}
        out: list[dict] = []
        for a in sorted(self.reads.alerts(limit), key=lambda a: a["window_start"]):
            cur = groups.get(a["key"])
            if cur is None or _when(a["window_start"]) - _when(cur["end"]) > self.config.server.incident_gap:
                host, _, service = a["key"].partition("/")
                cur = groups[a["key"]] = {
                    "id": a["alert_id"], "key": a["key"], "host": host, "service": service,
                    "start": a["window_start"], "end": a["window_end"], "alerts": 0, "peak_miss": 0.0,
                    "model_version": a["model_version"], "stream": a["model_version"].rsplit("-", 1)[0],
                    "windows": []}
                out.append(cur)
            cur["alerts"] += 1
            cur["end"] = max(cur["end"], a["window_end"], key=_when)
            cur["peak_miss"] = max(cur["peak_miss"], a["miss_ratio"])
            cur["windows"].append(a["template_ids"])
        for inc in out:
            state = self.control.get("incident", inc["id"]) or {}
            inc.update(status=state.get("status", "open"), note=state.get("note", ""),
                       updated_at=state.get("updated_at"), explanation=state.get("explanation"))
        return sorted(out, key=lambda i: _when(i["end"]), reverse=True)

    def incident(self, incident_id: str) -> dict:
        """One incident with its lines in order, each marked as the model expected it or not
        (and what it expected instead), plus warnings and errors elsewhere at the same time."""
        inc = next((i for i in self.incidents() if i["id"] == incident_id), None)
        if inc is None:
            raise KeyError(incident_id)
        start, end = _when(inc["start"]), _when(inc["end"])
        lines = self.reads.between(start, end, 400, inc["host"], inc["service"])
        snap = self.registry.current
        ids = [snap.canonical_id(int(line["template_id"])) for line in lines]
        scorer = self.scorers.get(inc["stream"])
        verdicts = ([v for i in range(0, len(ids), self.window) for v in scorer.explain(ids[i:i + self.window])]
                    if scorer else [{"status": "expected", "expected": []} for _ in ids])

        def text(tid: int) -> str:
            entry = snap.templates.get(tid)
            return _template_text(entry.tokens, entry.keys) if entry else ("not parsed yet" if tid == UNK_TEMPLATE_ID else f"#{tid}")
        def words(tid: int) -> str:  # what a person recognises: the template's fixed words, no slots
            entry = snap.templates.get(tid)
            return " ".join(t for t in entry.tokens if not SLOT_RE.search(t)) if entry else text(tid)
        events = [line | {"template_id": tid, "template": text(tid), "status": v["status"],
                          "expected": [{"template_id": t, "template": text(t), "words": words(t)} for t in v["expected"]]}
                  for line, tid, v in zip(lines, ids, verdicts)]
        around = [r for r in self.reads.between(start - timedelta(minutes=1), end + timedelta(minutes=1), 2000)
                  if r["level"] in ("error", "warn") and (r["host"], r["service"]) != (inc["host"], inc["service"])][:50]
        plan_id = (self.control.get("incident", incident_id) or {}).get("ask_plan")
        plan = self.control.get("ask_plan", plan_id) if plan_id else None
        return inc | {"events": events, "around": around,
                      "model_current": bool(scorer) and scorer.version == inc["model_version"],
                      "ask_plan": plan}

    def set_incident(self, incident_id: str, status: str, note: str = "") -> dict:
        if status not in INCIDENT_STATES:
            raise ValueError(f"status must be one of {', '.join(INCIDENT_STATES)}")
        inc = next((i for i in self.incidents() if i["id"] == incident_id), None)
        if inc is None:
            raise KeyError(incident_id)
        doc = (self.control.get("incident", incident_id) or {"id": incident_id}) | {
            "status": status, "note": note[:2000], "updated_at": now_iso(), "stream": inc["stream"],
            # what the next retrain learns as normal if this was a false positive
            "windows": inc["windows"][:200] if status == "false_positive" else []}
        self.control.put("incident", incident_id, doc)
        self.events.notify()
        return doc

    def explain_incident(self, incident_id: str, require_approval: bool | None = None) -> dict:
        """A root-cause Ask anchored at the incident's start, on its service: its queries wait for
        approval like any other question."""
        inc = self.incident(incident_id)  # 404 before planning
        spec = {"intent": "rca", "anchor_time": inc["start"], "anchor_event": f"anomaly incident on {inc['key']}",
                "services": [inc["service"]] if inc["service"] else [], "hosts": [inc["host"]] if inc["host"] else []}
        plan = self.ask.ask(f"Why did {inc['key']} misbehave from {inc['start']}?", spec=spec,
                            require_approval=require_approval, incident=incident_id)
        self.control.put("incident", incident_id, (self.control.get("incident", incident_id) or {"id": incident_id, "status": "open"})
                         | {"ask_plan": plan["id"], "stream": inc["stream"]})
        return plan

    def _job_ask(self, progress, plan: str, round: int) -> str:
        return self.ask.run(progress, plan, round, adapter=_Traced(self.adapter, progress, 0.5, 0.95))

    def _job_bench(self, progress, store: str = "memory", max_rate: int = 0, step_seconds: float = 10) -> str:
        """``logpipe bench`` in its own process (so it neither slows this server nor is skewed
        by it); its progress lines become the job's, its result a ``bench`` doc."""
        import subprocess
        import sys

        out = self.data_dir / "bench" / f"{uuid.uuid4().hex[:8]}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "logpipe.bench", store, str(int(max_rate)), str(step_seconds), str(out)]
        done = 0.05
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line.startswith(("step ", "flood", "calibrating", "generating", "warming")):
                    done = min(done + 0.1, 0.95)
                if line.strip():
                    progress(done, line.strip())
        if proc.returncode or not out.exists():
            raise RuntimeError(f"logpipe bench exited with {proc.returncode}")
        result = json.loads(out.read_text())
        self.control.put("bench", result["id"], result)
        fall = f", falls behind at {result['falls_behind_at']:,}/s" if result["falls_behind_at"] else ""
        return f"max sustained {result['max_sustained_per_s']:,} lines/s{fall}"

    def save_stream(self, doc: dict, stream_id: str | None = None) -> dict:
        name = str(doc.get("name", "")).strip()
        if not name:
            raise ValueError("a stream needs a name")
        selectors = [cleaned for sel in doc.get("selectors") or []
                     if (cleaned := {k: str(v) for k, v in sel.items() if k in ("host", "service", "source") and v})]
        picked = sorted({str(p) for p in doc.get("picked") or []})
        if unknown := [p for p in picked if p not in self.sources]:
            raise ValueError(f"unknown source {unknown[0]!r}")
        days = doc.get("retrain_days")
        if days is not None and (not isinstance(days, int | float) or days <= 0):
            raise ValueError("retrain_days must be a positive number of days")
        stream_id = stream_id or self._stream_id(name)
        existing = self.control.get("stream", stream_id) or {"id": stream_id, "created_at": now_iso(), "group": None}
        if "group" in doc:  # null takes it out of its group
            if doc["group"] is not None and self.control.get("group", str(doc["group"])) is None:
                raise ValueError(f"unknown group {doc['group']!r}")
            existing = existing | {"group": doc["group"]}
        for other in self.control.all("stream"):  # a ticked source moves here from wherever it was
            if other["id"] != stream_id and set(other.get("picked") or []) & set(picked):
                self.control.update("stream", other["id"], picked=sorted(set(other["picked"]) - set(picked)))
        saved = self.control.put("stream", stream_id, existing | {
            "name": name, "picked": picked, "selectors": selectors, "retrain_days": days})
        for sid in sorted(self.sources):  # unticked, or a pattern removed: never left without a model
            if not self.sources[sid].get("alias_of") and not self.streams_for(sid):
                self._assign(sid, avoid=stream_id)
        self._drop_empty_streams()  # the ones whose last source just moved here
        self.events.notify()
        return self.control.get("stream", stream_id) or saved

    def delete_stream(self, stream_id: str) -> None:
        if any((s := self.streams_for(sid)) and s[0]["id"] == stream_id for sid in self.sources):
            raise ValueError("move this stream's sources to another stream first")
        self.control.delete("stream", stream_id)
        with self.lock:
            self.scorers.pop(stream_id, None)
            self.detectors.pop(stream_id, None)
        shutil.rmtree(self._model_dir(stream_id), ignore_errors=True)  # a later stream may reuse the id
        self.events.notify()

    def heartbeat(self, status: dict, key: dict) -> dict:
        host = str(status.get("hostname") or "unknown")[:200]
        prev = self.control.get("agent", host) or {"first_seen": now_iso()}
        self.control.put("agent", host, prev | {
            "hostname": host, "status": status, "last_seen": now_iso(), "key_name": key["name"]})
        self.events.notify()
        return {"remote_hash": self.remote_hash_of(host)}

    def remote_hash_of(self, host: str) -> str:
        return (self.control.get("remote", host) or {}).get("hash", "")

    def set_remote(self, host: str, text: str) -> dict:
        validate_remote_yaml(text)
        doc = self.control.put("remote", host, {"id": host, "yaml": text, "hash": remote_hash(text),
                                                "updated_at": now_iso()})
        self.events.notify()  # the agent's open wait returns at once
        return doc

    def connected(self, host: str, delta: int) -> None:
        """An agent opened (+1) or closed (-1) its long-poll; presence follows it instantly."""
        with self._presence:
            self._waiting[host] = self._waiting.get(host, 0) + delta
            if self._waiting[host] <= 0:
                del self._waiting[host]
                self._gone[host] = time.monotonic()
            else:
                self._gone.pop(host, None)
        # agents reconnect within milliseconds; only a lasting disconnect turns them offline
        self.events.notify(self.config.server.gone_seconds + 0.2 if delta < 0 else 0)

    def _online(self, a: dict, cutoff: str) -> bool:
        host = a["hostname"]
        if host in self._waiting:
            return True
        gone = self._gone.get(host)
        if gone is not None and time.monotonic() - gone > self.config.server.gone_seconds:
            return False
        return a["last_seen"] >= cutoff

    def agents(self) -> list[dict]:
        out = []
        cutoff = (datetime.now(UTC) - timedelta(seconds=self.config.server.online_seconds)).isoformat()
        for a in self.control.all("agent"):
            remote = self.control.get("remote", a["hostname"]) or {}
            st = a.get("status", {})
            out.append(a | {"online": self._online(a, cutoff), "remote": remote,
                            "remote_applied": st.get("remote_hash", "") == remote.get("hash", "")})
        return sorted(out, key=lambda a: a["hostname"])

    def rate(self) -> float:
        """Lines per second over the last minute."""
        now = time.monotonic()
        self._rate.append((now, self.received))
        self._rate = [(t, n) for t, n in self._rate if now - t <= 60]
        (t0, n0), (t1, n1) = self._rate[0], self._rate[-1]
        return (n1 - n0) / (t1 - t0) if t1 > t0 else 0.0

    def persist(self) -> None:
        with self.lock:
            dirty, self._dirty = self._dirty, set()
            docs = []
            for kind, doc_id in dirty:
                if kind == "source":
                    docs.append((kind, doc_id, dict(self.sources[doc_id])))
                elif kind == "template":
                    docs.append((kind, doc_id, dict(self.tmeta[int(doc_id)])))
                else:
                    docs.append((kind, doc_id, next(
                        (d for d in self._keys.values() if d["id"] == doc_id), None)))
        for kind, doc_id, doc in docs:
            if doc is not None:
                self.control.put(kind, doc_id, doc)

    def due_retrains(self) -> list[str]:
        """Streams whose model is older than their retrain period and that have open
        unseen-template warnings."""
        open_streams = {s for w in self.control.all("warning")
                        if w["kind"] == "novel_template" and w["status"] == "open" for s in w["streams"]}
        due = []
        for stream in self.control.all("stream"):
            days = stream.get("retrain_days") or self.settings.retrain_days
            trained = stream.get("trained_at")
            if (stream["id"] in open_streams and trained and
                    datetime.fromisoformat(trained) + timedelta(days=days) <= datetime.now(UTC)):
                due.append(stream["id"])
        return due

    def store_usage(self) -> dict[str, float | None]:
        """The log store's size on disk and how full its disk is (ClickHouse only)."""
        client = getattr(self.store, "client", None)
        if client is None:
            return {}
        size = client.query("SELECT sum(bytes_on_disk) FROM system.parts WHERE active AND "
                            "database = currentDatabase()").result_rows[0][0] or 0
        free, total = client.query("SELECT sum(free_space), sum(total_space) FROM system.disks").result_rows[0]
        return {"store_mb": size / 2**20, "disk_used_pct": 100 * (1 - free / total) if total else None}

    def sample_metrics(self) -> dict[str, float]:
        gauges: dict[str, float | None] = {"inbox": self.inbox.qsize()}
        try:
            gauges["pending_rows"] = self.store.count_by_status().get("PENDING", 0)
            gauges |= self.store_usage()
        except Exception:  # a store hiccup must not stop the process metrics
            LOGGER.warning("store metrics unavailable", exc_info=True)
        drift = self.drift.report(self.control.all("stream"))
        gauges["score_psi_max"] = max((d["psi"] for d in drift if d["psi"] is not None), default=None)
        gauges["alert_window_pct"] = max((d["alert_pct"] for d in drift if d["alert_pct"] is not None), default=None)
        now, count = time.time(), len(self.registry.current.templates)
        if self._templates_at is not None and now > self._templates_at[0]:
            gauges["new_templates_per_hour"] = max(0, count - self._templates_at[1]) * 3600 / (now - self._templates_at[0])
        self._templates_at = (now, count)
        q = getattr(self.pipeline, "queue", None)
        if q is not None and hasattr(q, "stats"):
            gauges["queue_lag"] = q.stats().lag
        values = self.metrics.sample(gauges)
        for alert in self.metric_rules.evaluate(values):
            LOGGER.warning("metric alert: %s (%s = %g)", alert["label"], alert["metric"], alert["value"])
        return values

    def _housekeeping_loop(self) -> None:
        last_enrich = last_schedule = last_sample = time.monotonic()
        last_ready = 0.0
        while not self._stop.wait(5):
            try:
                self.persist()
                now = time.monotonic()
                if now - last_sample >= self.metrics.every - 0.5:
                    last_sample = now
                    self.sample_metrics()
                if now - last_enrich >= self.enrich_every_s and self.pipeline.worker is not None:
                    last_enrich = now
                    # Not under self.lock: the worker takes it only to publish, so lines that
                    # parse keep flowing while the LLM looks at the ones that don't.
                    self.enrich()
                if now - last_schedule >= self.schedule_every_s:
                    last_schedule = now
                    for sid in self.due_retrains():
                        self.train([sid])
                due = self.due_drafts()
                if due:
                    self.enqueue("draft", sources=due)
                if now - last_ready >= 60:
                    last_ready = now
                    self.refresh_readiness()
            except Exception:
                LOGGER.exception("housekeeping failed")

    def close(self) -> None:
        self._stop.set()
        self.persist()
        with self.lock:
            self.pipeline.finish()
            if self.pipeline.queue is not None:
                self.pipeline.queue.close()
        self.store.close()
