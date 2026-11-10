"""``logpipe demo``: the whole system on a live synthetic IoT fleet, in a browser.

Setup follows ``eval/iot_eval.py`` at toy size: calibrate on the normal history (or load a
registry cached on Colab with the fine-tuned mask model), train a small LogBERT, tune its
thresholds on a validation stream, then generate the test stream live (so a firmware rollout
or an anomaly can be triggered from the page) and feed it through ``Pipeline`` one line at a
time. A stdlib HTTP server exposes each stage as JSON for ``index.html``.
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
import threading
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

from deepparse.drain.masks_application import MaskApplier
from deepparse.tokenize import tokenize

from ..anomaly.detector import DetectorConfig, sliding_windows
from ..anomaly.logbert import LogBERTConfig, TrainConfig, train_logbert
from ..anomaly.scorer import LogBERTScorer, ScorerSettings, tune_thresholds
from ..anomaly.vocab import TemplateVocab
from ..calibrate import calibrate
from ..enrich.llm import get_adapter
from ..enrich.worker import CycleReport
from ..eval.iot_eval import _Labels
from ..eval.synth_iot import LiveFleet, as_log_events, generate
from ..parsing.frozen_parser import FrozenParser
from ..pipeline import Pipeline
from ..registry import UNK_TEMPLATE_ID, RegistrySnapshot, SchemaRegistry, TemplateEntry
from ..storage.base import AnomalyAlert, LogRow, LogStore
from ..storage.memory import MemoryStore
from ..unresolved import open_queue

if TYPE_CHECKING:
    from deepparse.ai_scraper.session import Browser

INDEX = Path(__file__).with_name("index.html")
EVAL_REPORT = Path("artifacts/eval/REPORT.md")
ANOMALY_RATE = 0.02
# The demo's own ClickHouse database, emptied at every session start.
DEMO_DB = "logpipe_demo"


def template_sql(entry: TemplateEntry, engine: str = "sqlite") -> str:
    """A SELECT that turns one template's rows into a table, one column per slot key."""
    # A payload ``timestamp`` repeats the row's column, and its alias would shadow it in ORDER BY.
    keys = [k for k in entry.keys if k not in ("timestamp", "device_id")]
    if engine == "clickhouse":
        cols = "".join(f",\n  payload.`{k}` AS `{k}`" for k in keys)
        return (f"SELECT timestamp, device_id{cols}\nFROM iot_logs FINAL\n"
                f"WHERE template_id = {entry.template_id} AND log_status = 'FINAL'\n"
                "ORDER BY timestamp DESC\nLIMIT 50")
    cols = "".join(f",\n  json_extract(payload, '$.\"{k}\"') AS \"{k}\"" for k in keys)
    return (f"SELECT timestamp, device_id{cols}\nFROM logs\n"
            f"WHERE template_id = {entry.template_id} AND status = 'FINAL'\n"
            "ORDER BY timestamp DESC\nLIMIT 50;")


class _Mirror(MemoryStore):
    """The page reads the in-memory rows; every write also goes to ClickHouse."""

    def __init__(self, sink: LogStore):
        super().__init__()
        self.sink = sink

    def insert_logs(self, rows: Sequence[LogRow]) -> None:
        super().insert_logs(rows)
        self.sink.insert_logs(rows)

    def insert_alerts(self, alerts: Sequence[AnomalyAlert]) -> None:
        super().insert_alerts(alerts)
        self.sink.insert_alerts(alerts)

    def flush(self) -> None:
        self.sink.flush()

    def close(self) -> None:
        self.sink.close()


def llm_choices() -> list[str]:
    from deepparse.ai_scraper.sites import SITES
    from deepparse.synth.chat_provider import API_KEY_ENV

    return ["offline", *(f"llm:web:{s}" for s in sorted(SITES)), *(f"llm:{p}" for p in API_KEY_ENV)]


def _meta_path(registry_path: Path) -> Path:
    return registry_path.with_suffix(".demo.json")


def cache_path(cache_dir: Path, llm: str, devices: int, history_hours: float, seed: int) -> Path:
    """Where a calibration of this fleet with this mask backend is kept between runs."""
    backend = re.sub(r"[^a-z0-9]+", "-", llm.lower()).strip("-")
    return cache_dir / f"registry_{backend}_d{devices}_h{history_hours:g}_s{seed}.json"


def save_registry(path: Path, *, devices: int = 2, history_hours: float = 2, seed: int = 7,
                  llm: str = "offline", browser: Browser = "camoufox", log=print,
                  adapter=None) -> None:
    """Calibrate the demo's history with ``llm`` and save the registry (the Colab step for ``hf``)."""
    fleet = generate(devices, history_hours, 0, 0, anomaly_rate=0.0, seed=seed)
    log(f"[demo] calibrating {len(fleet.history)} history lines with {llm}")
    cal = calibrate([e.line for e in fleet.history], adapter or get_adapter(llm, browser=browser))
    path.parent.mkdir(parents=True, exist_ok=True)
    SchemaRegistry(cal.snapshot).save(path)
    _meta_path(path).write_text(json.dumps({"mask_backend": llm, "devices": devices,
                                            "history_hours": history_hours, "seed": seed}))
    log(f"[demo] saved registry v{cal.snapshot.version} ({len(cal.snapshot.masks)} masks, "
        f"{len(cal.snapshot.templates)} templates) -> {path}")


class Demo:
    def __init__(self, devices: int = 2, history_hours: float = 2, test_hours: float = 1,
                 validation_hours: float = 1, epochs: int = 8, window: int = 20, stride: int = 5,
                 enrich_every: int = 300, seed: int = 7, llm: str = "offline", queue: str = "memory",
                 store: str = "memory", registry_path: Path | None = None,
                 browser: Browser = "camoufox", cache_dir: Path | None = None,
                 recalibrate: bool = False, log=print):
        """``cache_dir`` keeps the calibration between runs (reused unless ``recalibrate``);
        ``registry_path`` loads a given registry instead, e.g. fine-tuned masks from Colab."""
        fleet = generate(devices, history_hours, validation_hours, 0, anomaly_rate=ANOMALY_RATE,
                         seed=seed)
        adapter = get_adapter(llm, browser=browser)
        if registry_path is None and cache_dir is not None:
            cached = cache_path(cache_dir, llm, devices, history_hours, seed)
            if recalibrate or not cached.exists():
                save_registry(cached, devices=devices, history_hours=history_hours, seed=seed,
                              llm=llm, log=log, adapter=adapter)
            else:
                log(f"[demo] reusing the calibration in {cached} (--recalibrate to redo it)")
            registry_path = cached
        if registry_path is not None:
            snapshot = SchemaRegistry.load(registry_path).current
            meta_file = _meta_path(registry_path)
            meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
            self.mask_backend = meta.get("mask_backend", f"cached {registry_path.name}")
            if meta and (meta.get("devices"), meta.get("history_hours"), meta.get("seed")) != (
                    devices, history_hours, seed):
                log(f"[demo] warning: registry was calibrated for {meta}, this run uses "
                    f"devices={devices} history_hours={history_hours} seed={seed}")
            log(f"[demo] loaded registry v{snapshot.version} from {registry_path} "
                f"(masks by {self.mask_backend})")
        else:
            log(f"[demo] calibrating on {len(fleet.history)} history lines with {llm}")
            snapshot = calibrate([e.line for e in fleet.history], adapter).snapshot
            self.mask_backend = llm

        parser, seqs = FrozenParser(snapshot), defaultdict(list)
        for e in fleet.history:
            seqs[e.device_id].append(int(parser.parse(e.line).template_id))
        windows = [w for s in seqs.values() for w in sliding_windows(s, window, stride)]
        vocab = TemplateVocab.build(windows)
        encoded = [vocab.encode(w) for w in windows]
        random.Random(seed).shuffle(encoded)
        n_valid = max(1, len(encoded) // 10)
        log(f"[demo] training LogBERT on {len(windows)} windows, {epochs} epochs")
        trained = train_logbert(encoded[n_valid:], encoded[:n_valid],
                                LogBERTConfig(len(vocab), 32, 1, 2, max_len=window + 1),
                                TrainConfig(epochs=epochs), log=log)
        # Small vocabulary: a low top-g keeps the sequence signal meaningful (see iot_eval).
        self.scorer = LogBERTScorer.from_training(
            trained, vocab, settings=ScorerSettings(num_candidates=3), version="logbert-demo")
        detector_config = DetectorConfig(window=window, stride=stride, record_windows=True)

        log(f"[demo] tuning thresholds on {len(fleet.validation)} validation lines")
        val = Pipeline.assemble(SchemaRegistry(snapshot), MemoryStore(), scorer=self.scorer,
                                detector_config=detector_config)
        val.run(as_log_events(fleet.validation))
        assert val.detector is not None
        val_labels = _Labels(fleet.validation)
        self.scorer.settings.thresholds, _ = tune_thresholds(
            self.scorer, [w.score for w in val.detector.scored if not val_labels.kinds_in(w)],
            [w.score for w in val.detector.scored if val_labels.kinds_in(w)])

        self.initial: RegistrySnapshot = snapshot
        self.browser: Browser = browser
        self.adapter, self.detector_config = adapter, detector_config
        self.enrich_every = enrich_every
        self._fleet_args = (devices, history_hours, validation_hours, test_hours, seed)
        self.paused = False
        self.rate = 25.0
        self.lock = threading.RLock()
        self._appliers: dict[int, MaskApplier] = {}
        self._stats_lock = threading.Lock()
        self._stats_cache: tuple | None = None
        self.session = 0
        self._reset(snapshot, seed + 1, "session 1 started", (queue, store, llm),
                    self._open(queue, store, llm, adapter))

    def _open(self, queue: str, store: str, llm: str, adapter=None) -> tuple:
        """Queue, store and adapter for a session, starting their Docker services first.

        Done outside the lock: Docker and a web LLM can take a while, and a failure here
        leaves the running session untouched.
        """
        if queue not in ("memory", "kafka") or store not in ("memory", "clickhouse"):
            raise ValueError(f"queue {queue!r} / store {store!r}")
        services = [name for name, on in (("kafka", queue == "kafka"),
                                          ("clickhouse", store == "clickhouse")) if on]
        if services:
            from ..start import docker_up
            docker_up(services)
        # A fresh topic per session, so a broker kept up between demos never replays old rows.
        run = uuid.uuid4().hex[:8]
        q = (open_queue("kafka", topic=f"logpipe-demo-{run}", group=f"logpipe-demo-{run}")
             if queue == "kafka" else open_queue("memory"))
        st: MemoryStore = MemoryStore()
        if store == "clickhouse":
            from ..storage.clickhouse import ClickHouseStore

            sink = ClickHouseStore.from_env(database=DEMO_DB, batch_size=200)
            for table in ("iot_logs", "anomaly_alerts"):
                sink.client.command(f"TRUNCATE TABLE {table}")
            st = _Mirror(sink)
        return q, st, adapter or get_adapter(llm, browser=self.browser)

    def _reset(self, snapshot: RegistrySnapshot, stream_seed: int, note: str,
               kinds: tuple[str, str, str], opened: tuple) -> None:
        """Start a session: empty store and queue, ``snapshot`` as the registry, a fresh stream.

        Calibration, LogBERT and its thresholds are kept; they are the slow part.
        """
        devices, history_hours, validation_hours, test_hours, seed = self._fleet_args
        if self.session:
            self.queue.close()
            self.store.close()
            if opened[2] is not self.adapter and (close := getattr(self.adapter, "close", None)):
                close()  # a web LLM's browser
        self.queue_kind, self.store_kind, self.llm = kinds
        self.queue, self.store, self.adapter = opened
        self.registry = SchemaRegistry(snapshot)
        # The demo runs cycles itself (see step) so it can map each cycle back to its lines.
        self.pipeline = Pipeline.assemble(
            self.registry, self.store, adapter=self.adapter, scorer=self.scorer,
            detector_config=self.detector_config, enrich_every=10**12, queue=self.queue,
        )
        # Regenerated (cheap) so the devices start from their pre-rollout state again.
        fleet = generate(devices, history_hours, validation_hours, 0, anomaly_rate=ANOMALY_RATE,
                         seed=seed)
        start = fleet.validation[-1].timestamp if fleet.validation else fleet.history[-1].timestamp
        self.stream_seed = stream_seed
        self.fleet = LiveFleet(fleet.devices, start, test_hours, random.Random(stream_seed),
                               anomaly_rate=ANOMALY_RATE)
        self.labels = _Labels([])
        self.lines: list[dict] = []
        self.log_ids: list[str] = []
        self.line_of: dict[str, int] = {}
        self.trail: dict[int, list[str]] = defaultdict(list)
        self.events: list[dict] = []
        self._fw2: set[str] = set()
        self._in_anomaly: dict[str, str | None] = {}
        self._windows_seen = 0
        self._last_outcome: dict[str, dict] = {}
        self.finished = False
        self.session += 1
        self._event("control", note)

    def restart(self, new_stream: bool = False, keep_learned: bool = False,
                queue: str | None = None, store: str | None = None, llm: str | None = None) -> None:
        """Replay from the start: the same stream (or a new one), from calibration or keeping
        what enrichment learned, with this or another queue, store and keys/alias LLM."""
        kinds = (queue or self.queue_kind, store or self.store_kind, llm or self.llm)
        try:
            opened = self._open(*kinds, self.adapter if kinds[2] == self.llm else None)
        except Exception as error:
            raise RuntimeError(f"session not restarted: {error}") from error
        with self.lock:
            snapshot = self.registry.current if keep_learned else self.initial
            seed = random.randrange(1 << 30) if new_stream else self.stream_seed
            learned = len(snapshot.templates) - len(self.initial.templates)
            self._reset(snapshot, seed, f"session {self.session + 1} started: "
                        f"{'new' if new_stream else 'same'} stream, registry v{snapshot.version}"
                        + (f" keeping {learned} learned templates" if keep_learned else " from calibration")
                        + f"; {kinds[0]} queue, {kinds[1]} store, keys/alias by {kinds[2]}",
                        kinds, opened)
            self.paused = False

    def _event(self, kind: str, text: str, *, i: int | None = None, **extra) -> None:
        """``i`` is the line the event points at; default the line just ingested."""
        i = len(self.lines) - 1 if i is None else i
        self.events.append({"n": len(self.events), "i": i, "kind": kind,
                            "time": self.lines[-1]["time"] if self.lines else None, "text": text} | extra)

    def step(self) -> bool:
        """Feed the next live line through the pipeline; False once the stream is done."""
        with self.lock:
            if self.finished:
                return False
            e = self.fleet.next()
            if e is None:
                n = len(self.pipeline.stats.cycles)
                self.pipeline.finish()  # one last cycle and the detector's buffered windows
                for report in self.pipeline.stats.cycles[n:]:
                    self._cycle(report)
                self._windows()
                self.finished = True
                self._event("end", "stream finished")
                return False
            i = len(self.lines)
            self.labels.add(e)
            r = self.pipeline.process(e.to_log_event(i))
            # process() inserts exactly one row and runs no cycle (enrich_every is huge),
            # so the newest key is this line's row.
            log_id = str(next(reversed(self.store.rows)))
            self.log_ids.append(log_id)
            self.line_of[log_id] = i
            self.lines.append({
                "i": i, "device_id": e.device_id, "time": e.timestamp.isoformat(), "raw": e.line,
                "status": str(r.status), "template_id": r.template_id, "payload": r.payload,
                "registry_version": r.registry_version, "anomaly": e.anomaly, "firmware": e.firmware,
            })
            hit = str(r.status) == "FINAL"
            self.trail[i].append(f"v{r.registry_version} parser: " + (
                f"hit template #{r.template_id}" if hit else f"miss -> PENDING, published to {self.queue_kind} queue"))
            if e.firmware == 2 and e.device_id not in self._fw2:
                self._fw2.add(e.device_id)
                self._event("rollout", f"{e.device_id} now logs firmware 2.x", device=e.device_id)
            if e.anomaly and self._in_anomaly.get(e.device_id) != e.anomaly:
                self._event("truth", f"{e.anomaly} anomaly starts on {e.device_id}", device=e.device_id)
            self._in_anomaly[e.device_id] = e.anomaly
            if (i + 1) % self.enrich_every == 0:
                self._cycle(self.pipeline.enrich())
            self._windows()
            return True

    def _cycle(self, report: CycleReport | None) -> None:
        if report is None:
            return
        v = report.registry_version
        snap = self.registry.current
        first_line = {}  # template id -> first line it resolved, so a click shows that line
        for log_id, tid in report.resolved_rows:
            if log_id in self.line_of:
                first_line.setdefault(tid, self.line_of[log_id])
        for tid in report.published:
            t = snap.templates[tid]
            score, why = report.alias_decisions.get(tid, (0.0, ""))
            if tid in report.aliased:
                self._event("alias", f"v{v}: template #{tid} re-associated with #{report.aliased[tid]} "
                            f"(similarity {score:.2f}, {why})", template_id=tid, i=first_line.get(tid),
                            alias_of=report.aliased[tid], tokens=" ".join(t.tokens))
            else:
                self._event("new", f"v{v}: new template #{tid} (similarity {score:.2f}: {why})",
                            template_id=tid, i=first_line.get(tid), tokens=" ".join(t.tokens))
        for label, why in report.rejected.items():
            self._event("rejected", f"v{v}: cluster rejected ({why}): {label}")
        for log_id, tid in report.resolved_rows:
            if (i := self.line_of.get(log_id)) is not None:
                canon = snap.canonical_id(tid)
                self.trail[i].append(f"cycle -> v{v}: resolved as #{tid}"
                                     + (f" (alias of #{canon})" if canon != tid else "") + ", row backfilled")
        for log_id, attempts, why in report.requeued_rows:
            if (i := self.line_of.get(log_id)) is not None:
                self.trail[i].append(f"cycle -> v{v}: still PENDING ({why}), requeued attempt {attempts}")
        for log_id, why in report.dead_letter_rows:
            if (i := self.line_of.get(log_id)) is not None:
                self.trail[i].append(f"cycle -> v{v}: dead-lettered ({why})")
        if report.resolved:
            self._event("backfill", f"v{v}: {report.resolved} PENDING rows backfilled")
        if report.requeued_rows:
            self._event("requeue", f"v{v}: {len(report.requeued_rows)} rows requeued "
                        f"(max attempt {max(a for _, a, _ in report.requeued_rows)})")
        if report.dead_letter_rows:
            self._event("dead", f"v{v}: {len(report.dead_letter_rows)} rows dead-lettered")

    def _windows(self) -> None:
        """Turn newly scored windows into TP / FP / FN events, merging overlaps per device."""
        detector = self.pipeline.detector
        assert detector is not None
        for w in detector.scored[self._windows_seen:]:
            truth, flagged = self.labels.kinds_in(w), bool(w.reasons)
            outcome = ("tp" if flagged else "fn") if truth else ("fp" if flagged else None)
            last = self._last_outcome.get(w.device_id)
            if outcome is None:
                self._last_outcome.pop(w.device_id, None)
                continue
            if last and last["kind"] == outcome and w.start <= last["end"]:
                last["end"] = w.end  # same episode, already on the timeline
                continue
            what = {"tp": "ALERT, true positive", "fp": "ALERT, false positive",
                    "fn": "missed anomaly (false negative)"}[outcome]
            detail = ", ".join(sorted(truth)) if truth else ", ".join(w.reasons)
            self._event(outcome, f"{what} on {w.device_id}: {detail}", device=w.device_id)
            self._last_outcome[w.device_id] = {"kind": outcome, "end": w.end}
        self._windows_seen = len(detector.scored)

    def replay(self) -> None:
        """Background feeder; idles once the stream is done so a restart can start it again."""
        while True:
            if self.paused or self.finished:
                time.sleep(0.1)
                continue
            self.step()
            time.sleep(1 / self.rate)

    def control(self, action: str, value: str = "") -> dict:
        if action == "restart":  # outside the lock: it may wait for Docker
            flags = {x for x in value.split(",") if "=" not in x}
            opts = dict(x.split("=", 1) for x in value.split(",") if "=" in x)
            self.restart(new_stream="new" in flags, keep_learned="keep" in flags,
                         queue=opts.get("queue"), store=opts.get("store"), llm=opts.get("llm"))
            return {"paused": self.paused, "rate": self.rate, "session": self.session}
        with self.lock:
            if action == "pause":
                self.paused = True
            elif action == "resume":
                self.paused = False
            elif action == "step":
                self.paused = True
                self.step()
            elif action == "speed":
                self.rate = min(500.0, max(0.5, float(value)))
            elif action == "rollout":
                ids = self.fleet.rollout()
                self._event("control", f"rollout triggered: {', '.join(ids)} move to firmware 2.x")
            elif action == "inject":
                device_id, kind = value.split(":", 1)
                if device_id not in self.fleet.devices:
                    raise KeyError(device_id)
                self.fleet.inject(device_id, kind)
                self._event("control", f"injected {kind} into the next episode of {device_id}", device=device_id)
            elif action == "enrich":
                self._cycle(self.pipeline.enrich())
            else:
                raise ValueError(f"unknown action {action!r}")
            return {"paused": self.paused, "rate": self.rate, "session": self.session}

    def stream(self, after: int) -> dict:
        with self.lock:
            return {"lines": self.lines[after:after + 5000], "done": len(self.lines),
                    "end": self.fleet.end.isoformat(), "finished": self.finished,
                    "paused": self.paused, "rate": self.rate,
                    "registry_version": self.registry.current.version,
                    "alerts": len(self.pipeline.stats.alerts), "session": self.session}

    def _queue_stats(self) -> dict:
        """Kafka's lag asks the broker (~1.5 s); the page polls every second, so it is read
        outside the demo lock and reused for a few seconds."""
        queue, now = self.queue, time.monotonic()
        if self.queue_kind == "memory":
            return queue.stats().to_json()
        with self._stats_lock:
            cached = self._stats_cache
            if cached is None or cached[0] is not queue or now - cached[1] > 3:
                cached = self._stats_cache = (queue, now, queue.stats().to_json())
        return cached[2]

    def system(self) -> dict:
        """Counters for every node and edge of the flow graph."""
        queue_stats = self._queue_stats()
        with self.lock:
            stats, cycles = self.pipeline.stats, self.pipeline.stats.cycles
            outcomes = Counter(e["kind"] for e in self.events)
            detector = self.pipeline.detector
            assert detector is not None
            return {
                "lines": stats.lines, "hit": stats.final, "miss": stats.pending,
                "registry_version": self.registry.current.version,
                "templates": len(self.registry.current.templates),
                "masks": len(self.registry.current.masks), "mask_backend": self.mask_backend,
                "llm": self.llm, "queue": queue_stats, "store": self.store_kind,
                "choices": {"queue": ["memory", "kafka"], "store": ["memory", "clickhouse"],
                            "llm": llm_choices()},
                "cycles": len(cycles), "clusters": sum(c.clusters for c in cycles),
                "published": sum(len(c.published) for c in cycles),
                "aliased": sum(len(c.aliased) for c in cycles),
                "rejected": sum(len(c.rejected) for c in cycles),
                "resolved": sum(c.resolved for c in cycles),
                "requeued": sum(len(c.requeued_rows) for c in cycles),
                "dead_lettered": sum(c.dead_lettered for c in cycles),
                "llm_calls": sum(c.llm_calls for c in cycles),
                "pending_now": Counter(r.template_id for r in self.store.rows.values()).get(UNK_TEMPLATE_ID, 0),
                "windows": len(detector.scored), "alerts": len(stats.alerts),
                "tp": outcomes["tp"], "fp": outcomes["fp"], "fn": outcomes["fn"],
                "devices": {d: self.fleet.kinds(d) for d in sorted(self.fleet.devices)},
                "paused": self.paused, "rate": self.rate, "finished": self.finished,
                "session": self.session,
                "learned": len(self.registry.current.templates) - len(self.initial.templates),
            }

    def timeline(self, after: int) -> dict:
        with self.lock:
            return {"events": self.events[after:]}

    def templates(self) -> dict:
        with self.lock:
            snap = self.registry.current
            counts = Counter(r.template_id for r in self.store.rows.values())
            cycles = [{"registry_version": c.registry_version, "buffered": c.buffered,
                       "published": c.published, "aliased": c.aliased, "resolved": c.resolved,
                       "still_pending": c.still_pending, "dead_lettered": c.dead_lettered}
                      for c in self.pipeline.stats.cycles]
            return {
                "version": snap.version,
                "masks": [m.to_dict() for m in snap.masks],
                "mask_backend": self.mask_backend,
                "pending_rows": counts.get(UNK_TEMPLATE_ID, 0),
                "templates": [asdict(t) | {
                    "count": counts.get(t.template_id, 0),
                    "learned": t.template_id not in self.initial.templates,
                    "sql": template_sql(t, self._engine()),
                } for t in sorted(snap.templates.values(), key=lambda t: t.template_id)],
                "cycles": cycles, "engine": self._engine(),
            }

    def _engine(self) -> str:
        return "clickhouse" if isinstance(self.store, _Mirror) else "sqlite"

    def query(self, sql: str) -> dict:
        with self.lock:
            store = self.store
            rows = list(store.rows.values())
        if isinstance(store, _Mirror):
            store.flush()
            client = store.sink.client  # type: ignore[attr-defined]
            try:
                result = client.query(sql, settings={"readonly": 2})
            except Exception as error:  # the server's message, shown under the editor
                raise ValueError(str(error).strip().splitlines()[0]) from error
            return {"columns": list(result.column_names), "rows": result.result_rows[:200],
                    "table_rows": len(rows), "engine": "clickhouse"}
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE logs (log_id TEXT, timestamp TEXT, device_id TEXT, "
                   "template_id INTEGER, status TEXT, registry_version INTEGER, "
                   "payload TEXT, raw_message TEXT)")
        db.executemany("INSERT INTO logs VALUES (?,?,?,?,?,?,?,?)", [
            (str(r.log_id), r.timestamp.isoformat(), r.device_id, r.template_id, str(r.status),
             r.registry_version, json.dumps(r.payload), r.raw_message) for r in rows])
        db.execute("PRAGMA query_only = ON")
        cur = db.execute(sql)
        return {"columns": [d[0] for d in cur.description or ()], "rows": cur.fetchmany(200),
                "table_rows": len(rows), "engine": "sqlite"}

    def metrics(self) -> dict:
        with self.lock:
            buckets: dict[str, Counter] = defaultdict(Counter)
            for line in self.lines:
                buckets[line["time"][:16]][line["status"]] += 1
            detector = self.pipeline.detector
            assert detector is not None
            windows = [self._window(w) for w in detector.scored]
        t = self.scorer.settings.thresholds
        return {
            "minutes": [{"minute": m, "FINAL": c["FINAL"], "PENDING": c["PENDING"]}
                        for m, c in sorted(buckets.items())],
            "windows": windows,
            "thresholds": asdict(t) | {"radius": self.scorer.radius},
            "flags_by_reason": dict(Counter(r for w in windows for r in w["reasons"])),
        }

    def _window(self, w) -> dict:
        s = w.score
        return {"device_id": w.device_id, "start": w.start.isoformat(), "end": w.end.isoformat(),
                "template_ids": list(w.template_ids),
                "in_vocab": [t in self.scorer.vocab for t in w.template_ids],
                "reasons": list(w.reasons), "truth": sorted(self.labels.kinds_in(w)),
                "score": asdict(s) | {"miss_ratio": s.miss_ratio, "novelty_ratio": s.novelty_ratio}}

    def trace(self, i: int) -> dict:
        with self.lock:
            line = self.lines[i]
            row = self.store.rows[uuid.UUID(self.log_ids[i])]
            snap = self.registry.current
            applier = self._appliers.get(snap.version)
            if applier is None:
                applier = self._appliers[snap.version] = MaskApplier(snap.masks)
            masked = applier.apply(line["raw"])
            ingest_t = snap.templates.get(line["template_id"])
            now_t = snap.templates.get(row.template_id)
            detector = self.pipeline.detector
            assert detector is not None
            windows = [self._window(w) for w in detector.scored
                       if w.device_id == row.device_id and w.start <= row.timestamp <= w.end]
            trail = list(self.trail[i])
        return {
            "line": line, "masked": masked, "tokens": tokenize(masked),
            "ingest_template": ingest_t and asdict(ingest_t),
            "current": {"status": str(row.status), "template_id": row.template_id,
                        "payload": row.payload, "registry_version": row.registry_version,
                        "template": now_t and asdict(now_t)},
            "windows": windows, "trail": trail,
            "path": self._path(line, trail, windows),
        }

    @staticmethod
    def _path(line: dict, trail: list[str], windows: list[dict]) -> list[str]:
        """Graph nodes this line went through; ``requeue`` marks the alias -> queue edge."""
        path = ["source", "parser", "store", "detector"]
        if any(w["reasons"] for w in windows):
            path.append("alert")
        if line["status"] == "FINAL":
            return path
        path.append("queue")
        if len(trail) > 1:  # at least one cycle looked at it
            path += ["shadow", "llm", "validate", "alias"]
        if any("resolved" in x for x in trail):
            path += ["registry", "backfill"]
        if any("requeued" in x for x in trail):
            path.append("requeue")
        if any("dead-lettered" in x for x in trail):
            path.append("dead")
        return path

    @staticmethod
    def eval_report() -> dict:
        return {"report": EVAL_REPORT.read_text(encoding="utf-8") if EVAL_REPORT.exists() else None}


def make_handler(demo: Demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args) -> None:  # keep the terminal quiet while polling
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the page reloaded or the client gave up; nothing to answer

        def _route(self, routes: dict) -> None:
            url = urlparse(self.path)
            if url.path not in routes:
                return self._send(404, b'{"error": "not found"}')
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                payload, code = routes[url.path](q), 200
            except (KeyError, ValueError, IndexError, RuntimeError, sqlite3.Error) as exc:
                payload, code = {"error": f"{type(exc).__name__}: {exc}"}, 400
            self._send(code, json.dumps(payload, default=str).encode())

        def do_GET(self) -> None:
            if urlparse(self.path).path in ("/", "/index.html"):
                return self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            self._route({
                "/api/stream": lambda q: demo.stream(int(q.get("after", 0))),
                "/api/system": lambda q: demo.system(),
                "/api/events": lambda q: demo.timeline(int(q.get("after", 0))),
                "/api/templates": lambda q: demo.templates(),
                "/api/query": lambda q: demo.query(q["sql"]),
                "/api/metrics": lambda q: demo.metrics(),
                "/api/trace": lambda q: demo.trace(int(q["i"])),
                "/api/eval": lambda q: demo.eval_report(),
            })

        def do_POST(self) -> None:
            self._route({"/api/control": lambda q: demo.control(q["action"], q.get("value", ""))})

    return Handler


def serve(demo: Demo, host: str = "127.0.0.1", port: int = 8000, rate: float = 25) -> None:
    demo.rate = rate
    threading.Thread(target=demo.replay, daemon=True).start()
    server = ThreadingHTTPServer((host, port), make_handler(demo))
    print(f"[demo] live stream until {demo.fleet.end:%H:%M} event time at {rate} lines/s, "
          f"{demo.queue_kind} queue, {demo.store_kind} store, keys/alias by {demo.llm} -> http://{host}:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[demo] stopping")
    finally:
        server.server_close()
        demo.queue.close()
