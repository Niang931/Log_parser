"""``logpipe server``'s engine: agent intake into the existing pipeline, the learning lifecycle
(setup -> calibration -> per-stream LogBERT -> warnings -> recalibration/retraining) and the
fleet's control plane.

One ingest thread owns the pipeline; one job thread runs calibration and training; a
housekeeping thread persists counters, runs enrichment and the retraining schedule.
"""
from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import logging
import queue
import random
import re
import secrets
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from deepparse.utils.sampling import entropy_greedy_sample

from ..anomaly.detector import DetectorConfig, StreamingDetector, sliding_windows
from ..anomaly.logbert import LogBERTConfig, TrainConfig, train_logbert
from ..anomaly.scorer import LogBERTScorer, ScorerSettings, Thresholds
from ..anomaly.vocab import TemplateVocab
from ..calibrate import append_calibrate, calibrate
from ..enrich.llm import LLMAdapter
from ..parsing.frozen_parser import ParserHandle, Status
from ..pipeline import Pipeline
from ..registry import UNK_TEMPLATE_ID, SchemaRegistry
from ..sources.file_replay import LogEvent
from ..unresolved import UnresolvedQueue
from .control import ControlStore
from .reads import reads_for

LOGGER = logging.getLogger("logpipe.server")

ONLINE_SECONDS = 45  # agents that don't hold /intake/v1/wait open (older builds)
GONE_SECONDS = 1  # an agent whose wait connection closed and didn't come back is offline


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
DEFAULT_SETTINGS = {"sample_cap": 5000, "retrain_days": 7, "calibrated": False,
                    "calibrated_at": None, "setup_done": False}
MIN_TRAIN_WINDOWS = 8


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


class Busy(RuntimeError):
    """The ingest buffer is full; the agent keeps the batch and retries."""


class Server:
    def __init__(self, store, control: ControlStore, *, data_dir: Path,
                 adapter_factory: Callable[[], LLMAdapter],
                 queue_factory: Callable[[], UnresolvedQueue] | None = None,
                 bootstrap_key: str | None = None, epochs: int = 8, window: int = 20,
                 stride: int = 5, inbox_size: int = 50_000, enrich_every_s: float = 30,
                 schedule_every_s: float = 600, start_threads: bool = True):
        self.store, self.control = store, control
        self.reads = reads_for(store)
        self.data_dir = data_dir
        self.registry_path = data_dir / "registry.json"
        self.adapter_factory, self.queue_factory = adapter_factory, queue_factory
        self._adapter: LLMAdapter | None = None
        self.epochs, self.window, self.stride = epochs, window, stride
        self.enrich_every_s, self.schedule_every_s = enrich_every_s, schedule_every_s
        self.lock = threading.RLock()
        self.inbox: queue.Queue[LogEvent] = queue.Queue(maxsize=inbox_size)
        self.jobs: queue.Queue[str] = queue.Queue()
        self.received = 0
        self._rate: list[tuple[float, int]] = []
        self._dirty: set[tuple[str, str]] = set()
        self.events = Notifier()
        self._waiting: dict[str, int] = {}  # host -> open /intake/v1/wait requests
        self._gone: dict[str, float] = {}  # host -> monotonic time its last wait closed

        self.settings = {**DEFAULT_SETTINGS, **(control.get("settings", "server") or {})}
        if not control.all("stream"):
            control.put("stream", "all", {"id": "all", "name": "All logs", "selectors": [{}],
                                          "retrain_days": None, "created_at": now_iso()})
        self.sources = {d["id"]: d for d in control.all("source")}
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
        self._stop = threading.Event()
        if start_threads:
            for target in (self._ingest_loop, self._job_loop, self._housekeeping_loop):
                threading.Thread(target=target, daemon=True, name=target.__name__).start()

    # --- pipeline -----------------------------------------------------------------------
    @property
    def adapter(self) -> LLMAdapter:
        if self._adapter is None:
            self._adapter = self.adapter_factory()
        return self._adapter

    def _build_pipeline(self) -> None:
        """Before calibration: store everything as PENDING, learn nothing (the setup guide
        decides when calibration starts). After: online enrichment + per-stream detectors."""
        with self.lock:
            if self.settings["calibrated"]:
                self.pipeline = Pipeline.assemble(
                    self.registry, self.store, adapter=self.adapter,
                    queue=self.queue_factory() if self.queue_factory else None,
                    registry_path=self.registry_path, enrich_every=10**12)
            else:
                self.pipeline = Pipeline(ParserHandle(self.registry.current), self.store)
            self.detectors = {sid: self._detector(s) for sid, s in self.scorers.items()}

    def _detector(self, scorer: LogBERTScorer) -> StreamingDetector:
        handle = self.pipeline.handle
        return StreamingDetector(
            scorer, DetectorConfig(window=self.window, stride=self.stride),
            registry_version=lambda: handle.version,
            resolver=lambda line: self.registry.current.canonical_id(int(handle.parse(line).template_id)))

    def _model_dir(self, stream_id: str) -> Path:
        return self.data_dir / "models" / stream_id

    # --- API keys -----------------------------------------------------------------------
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

    # --- intake -------------------------------------------------------------------------
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
                LOGGER.exception("failed to process a line from %s", event.device_id)

    def process(self, event: LogEvent) -> None:
        with self.lock:
            result = self.pipeline.process(event)
            ts = self.pipeline._event_time or datetime.now(UTC)
            self.received += 1
            sid = source_id(event.device_id, event.service, event.source)
            src = self.sources.get(sid)
            if src is None:
                src = self.sources[sid] = {
                    "id": sid, "host": event.device_id, "service": event.service,
                    "source": event.source, "first_seen": now_iso(), "count": 0,
                    "calibrated": False, "alias_of": None}
                if self.settings["calibrated"]:
                    self._warn("new_source", sid, f"New log source {event.service or event.source} on {event.device_id}",
                               "Collected after calibration: no template sample and no LogBERT model has seen it yet.",
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
                    self.store.insert_alerts(alerts)
                if tid != UNK_TEMPLATE_ID and tid not in detector.scorer.vocab:
                    self._novel(tid, stream["id"])

    def streams_for(self, sid: str) -> list[dict]:
        src = self.sources.get(sid)
        target = self.sources.get(src["alias_of"]) if src and src.get("alias_of") else src
        if target is None:
            return []
        return [s for s in self.control.all("stream")
                if any(_matches(sel, target["host"], target["service"], target["source"])
                       for sel in s["selectors"])]

    # --- warnings -----------------------------------------------------------------------
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
            else:
                snapshot = self.registry.alias(int(w["subject"]), int(target))
                self.registry.save(self.registry_path)
                self.pipeline.handle.swap(snapshot)
        return self.resolve_warning(wid, f"same as {target}")

    def recalibrate_warning(self, wid: str) -> dict:
        w = self.control.get("warning", wid)
        if w is None:
            raise KeyError(wid)
        if w["kind"] == "new_source":
            return self.enqueue("recalibrate", sources=[w["subject"]], warning=wid)
        return self.enqueue("retrain", streams=w["streams"], warning=wid)

    # --- jobs ---------------------------------------------------------------------------
    def enqueue(self, kind: str, **params) -> dict:
        if kind == "calibrate" and self.settings["calibrated"]:
            raise ValueError("already calibrated; add sources with recalibration instead")
        active = [j for j in self.control.all("job") if j["status"] in ("queued", "running")]
        for j in active:
            if j["kind"] == kind and j["params"] == params:
                return j
        doc = {"id": uuid.uuid4().hex[:12], "kind": kind, "params": params, "status": "queued",
               "progress": 0.0, "message": "waiting", "created_at": now_iso()}
        self.control.put("job", doc["id"], doc)
        self.jobs.put(doc["id"])
        return doc

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
        self.control.update("job", job_id, status="running", started_at=now_iso())

        def progress(fraction: float, message: str) -> None:
            self.control.update("job", job_id, progress=round(min(max(fraction, 0.0), 1.0), 3),
                                message=message)

        try:
            getattr(self, f"_job_{job['kind']}")(progress, **job["params"])
        except Exception as error:
            LOGGER.exception("job %s failed", job_id)
            return self.control.update("job", job_id, status="failed", error=str(error),
                                       finished_at=now_iso())
        return self.control.update("job", job_id, status="done", progress=1.0, message="done",
                                   finished_at=now_iso())

    def _sample(self, sids: list[str], progress, start: float, end: float) -> list[str]:
        """Up to ``sample_cap`` lines per source, picked by entropy-greedy priority sampling
        from the 4x most recent, so one chatty source can't crowd out the rest."""
        cap = int(self.settings["sample_cap"])
        lines: list[str] = []
        for i, sid in enumerate(sids):
            src = self.sources[sid]
            recent = self.reads.source_lines(src["host"], src["service"], src["source"], cap * 4)
            lines += [recent[j] for j in sorted(entropy_greedy_sample(recent, cap))]
            progress(start + (end - start) * (i + 1) / len(sids), f"sampled {src['service'] or sid}")
        return lines

    def _job_calibrate(self, progress) -> None:
        sids = [sid for sid, s in self.sources.items() if not s.get("alias_of")]
        if not sids:
            raise ValueError("no log sources yet: connect an agent and track some files first")
        lines = self._sample(sids, progress, 0.0, 0.15)
        progress(0.2, f"calibrating on {len(lines)} lines (masks, Drain, key names)")
        result = calibrate(lines, self.adapter)
        with self.lock:
            self.registry = SchemaRegistry(result.snapshot)
            self.registry.save(self.registry_path)
            self.settings.update(calibrated=True, calibrated_at=now_iso(), setup_done=True)
            self.control.put("settings", "server", self.settings)
            for sid in sids:
                self.sources[sid]["calibrated"] = True
                self._dirty.add(("source", sid))
            self._build_pipeline()
        progress(0.5, f"registry v{result.snapshot.version}: {len(result.snapshot.templates)} templates; backfilling")
        self._backfill()
        if self.pipeline.worker is not None:
            self.pipeline.worker.rehydrate()  # what calibration didn't cover is learned online
        streams = self.control.all("stream")
        for i, stream in enumerate(streams):
            lo, hi = 0.6 + 0.4 * i / len(streams), 0.6 + 0.4 * (i + 1) / len(streams)
            try:
                self._train(stream["id"], lambda f, m, lo=lo, hi=hi: progress(lo + (hi - lo) * f, m))
            except ValueError as error:  # too little data: the stream stays without a model
                self.control.update("stream", stream["id"], model_error=str(error))

    def _job_recalibrate(self, progress, sources: list[str], warning: str | None = None) -> None:
        """Append templates for new sources (existing ids never change), then retrain the
        streams those sources feed (their models are replaced)."""
        lines = self._sample(sources, progress, 0.0, 0.15)
        progress(0.2, f"adding templates from {len(lines)} lines")
        with self.lock:
            snapshot_before = self.registry.current.version
        new_ids = append_calibrate(self.registry, lines, self.adapter)
        with self.lock:
            self.registry.save(self.registry_path)
            self.pipeline.handle.swap(self.registry.current)
            for sid in sources:
                self.sources[sid]["calibrated"] = True
                self._dirty.add(("source", sid))
        progress(0.4, f"{len(new_ids)} new templates (registry v{snapshot_before} -> "
                      f"v{self.registry.current.version}); backfilling")
        self._backfill()
        streams = sorted({s["id"] for sid in sources for s in self.streams_for(sid)})
        self._retrain(streams, progress, 0.5)
        for sid in sources:
            wid = f"new_source:{sid}"
            if self.control.get("warning", wid):
                self.resolve_warning(wid, "recalibrated")

    def _job_retrain(self, progress, streams: list[str], warning: str | None = None) -> None:
        self._retrain(streams, progress, 0.0)

    def _retrain(self, streams: list[str], progress, start: float) -> None:
        for i, sid in enumerate(streams):
            lo = start + (1 - start) * i / max(len(streams), 1)
            hi = start + (1 - start) * (i + 1) / max(len(streams), 1)
            self._train(sid, lambda f, m, lo=lo, hi=hi: progress(lo + (hi - lo) * f, m))

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

    def _train(self, stream_id: str, progress) -> None:
        """Train this stream's LogBERT on capped, per-source sequences of canonical template ids."""
        stream = self.control.get("stream", stream_id)
        if stream is None:
            raise KeyError(stream_id)
        cap = int(self.settings["sample_cap"])
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
        if len(windows) < MIN_TRAIN_WINDOWS:
            raise ValueError(f"stream {stream['name']}: {len(windows)} windows of parsed logs, "
                             f"need {MIN_TRAIN_WINDOWS} (let the agents collect more)")
        progress(0.05, f"{stream['name']}: {len(windows)} windows from {len(sequences)} sources")
        random.Random(7).shuffle(windows)
        n_valid = max(1, len(windows) // 10)
        vocab = TemplateVocab.build(windows)
        encoded = [vocab.encode(w) for w in windows]
        epochs = self.epochs

        def log(message: str) -> None:
            if message.startswith("epoch "):
                epoch = int(message.split()[1].rstrip(":"))
                progress(0.1 + 0.8 * (epoch + 1) / epochs, f"{stream['name']}: {message}")

        trained = train_logbert(encoded[n_valid:], encoded[:n_valid],
                                LogBERTConfig(len(vocab), 32, 1, 2, max_len=self.window + 1),
                                TrainConfig(epochs=epochs), log=log)
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
        with self.lock:
            self.scorers[stream_id] = scorer
            self.detectors[stream_id] = self._detector(scorer)
        self.control.update("stream", stream_id, trained_at=now_iso(), model_version=version,
                            vocab=sorted(vocab.to_json()), windows=len(windows),
                            miss_threshold=round(miss, 3), model_error=None)
        for w in self.control.all("warning"):
            if (w["kind"] == "novel_template" and w["status"] == "open"
                    and int(w["subject"]) in vocab and stream_id in w["streams"]):
                rest = [s for s in w["streams"] if s != stream_id]
                if rest:
                    self.control.update("warning", w["id"], streams=rest)
                else:
                    self.resolve_warning(w["id"], f"learned by {version}")
        progress(1.0, f"{stream['name']}: model {version} ready")

    # --- streams ------------------------------------------------------------------------
    def save_stream(self, doc: dict, stream_id: str | None = None) -> dict:
        name = str(doc.get("name", "")).strip()
        if not name:
            raise ValueError("a stream needs a name")
        selectors = [{k: str(v) for k, v in sel.items() if k in ("host", "service", "source") and v}
                     for sel in doc.get("selectors") or [{}]]
        days = doc.get("retrain_days")
        if days is not None and (not isinstance(days, int | float) or days <= 0):
            raise ValueError("retrain_days must be a positive number of days")
        stream_id = stream_id or re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or uuid.uuid4().hex[:8]
        existing = self.control.get("stream", stream_id) or {"id": stream_id, "created_at": now_iso()}
        return self.control.put("stream", stream_id, existing | {
            "name": name, "selectors": selectors or [{}], "retrain_days": days})

    def delete_stream(self, stream_id: str) -> None:
        if stream_id == "all":
            raise ValueError("the default stream can't be deleted")
        self.control.delete("stream", stream_id)
        with self.lock:
            self.scorers.pop(stream_id, None)
            self.detectors.pop(stream_id, None)

    # --- agents -------------------------------------------------------------------------
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
        with self.lock:
            self._waiting[host] = self._waiting.get(host, 0) + delta
            if self._waiting[host] <= 0:
                del self._waiting[host]
                self._gone[host] = time.monotonic()
            else:
                self._gone.pop(host, None)
        # agents reconnect within milliseconds; only a lasting disconnect turns them offline
        self.events.notify(GONE_SECONDS + 0.2 if delta < 0 else 0)

    def _online(self, a: dict, cutoff: str) -> bool:
        host = a["hostname"]
        if host in self._waiting:
            return True
        gone = self._gone.get(host)
        if gone is not None and time.monotonic() - gone > GONE_SECONDS:
            return False
        return a["last_seen"] >= cutoff

    def agents(self) -> list[dict]:
        out = []
        cutoff = (datetime.now(UTC) - timedelta(seconds=ONLINE_SECONDS)).isoformat()
        for a in self.control.all("agent"):
            remote = self.control.get("remote", a["hostname"]) or {}
            st = a.get("status", {})
            out.append(a | {"online": self._online(a, cutoff), "remote": remote,
                            "remote_applied": st.get("remote_hash", "") == remote.get("hash", "")})
        return sorted(out, key=lambda a: a["hostname"])

    # --- housekeeping -------------------------------------------------------------------
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
            days = stream.get("retrain_days") or self.settings["retrain_days"]
            trained = stream.get("trained_at")
            if (stream["id"] in open_streams and trained and
                    datetime.fromisoformat(trained) + timedelta(days=days) <= datetime.now(UTC)):
                due.append(stream["id"])
        return due

    def _housekeeping_loop(self) -> None:
        last_enrich = last_schedule = time.monotonic()
        while not self._stop.wait(5):
            try:
                self.persist()
                now = time.monotonic()
                if now - last_enrich >= self.enrich_every_s and self.pipeline.worker is not None:
                    last_enrich = now
                    with self.lock:
                        self.pipeline.enrich()
                if now - last_schedule >= self.schedule_every_s:
                    last_schedule = now
                    for sid in self.due_retrains():
                        self.enqueue("retrain", streams=[sid], warning="scheduled")
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
