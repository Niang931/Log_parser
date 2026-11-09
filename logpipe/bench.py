"""``logpipe bench``: how many log lines one server keeps up with, and what that costs.

A real :class:`~logpipe.server.core.Server` (calibrated on the synthetic IoT fleet's history,
offline LLM, no anomaly models) is fed the fleet's live phase through its intake at rising rates.
Each step reports what it was offered and what it processed, parse latency, CPU and memory; the
first step it can't keep up with ends the ramp. A last *flood* step offers as much as the intake
takes, which is the raw ceiling. The load generator runs in the same process, so its CPU counts.
"""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

DEFAULT_RATES = (500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000)
KEEP_UP = 0.95  # processed / offered for a step to count as kept up
LATENCY_CAP = 50_000


def _rss_mb() -> float:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError, IndexError):
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _pcts(values: list[int], scale: float = 1.0) -> dict[str, float]:
    if not values:
        return {}
    a = np.asarray(values) / scale
    return {f"p{q}": round(float(np.percentile(a, q)), 2) for q in (50, 95, 99)}


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


class _Recorder:
    """Latencies from wrapped calls, a bounded uniform sample per step."""

    def __init__(self):
        self.lock = threading.Lock()
        self.values: list[int] = []
        self.seen = 0
        self.rng = random.Random(0)

    def add(self, ns: int) -> None:
        with self.lock:
            self.seen += 1
            if len(self.values) < LATENCY_CAP:
                self.values.append(ns)
            elif (j := self.rng.randrange(self.seen)) < LATENCY_CAP:
                self.values[j] = ns

    def take(self) -> list[int]:
        with self.lock:
            out, self.values, self.seen = self.values, [], 0
            return out

    def wrap(self, fn):
        def timed(*args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                self.add(time.perf_counter_ns() - start)
        return timed


def _setup(store: str, data_dir: Path, say: Callable[[str], None]):
    from .calibrate import calibrate
    from .enrich.llm import OfflineAdapter, get_adapter
    from .eval.synth_iot import generate
    from .registry import SchemaRegistry
    from .server.control import ControlStore
    from .server.core import Server, now_iso, source_id

    say("generating the synthetic IoT fleet")
    fleet = generate(devices_per_kind=3, history_hours=2, validation_hours=0.01, test_hours=3, seed=11)
    say(f"calibrating on {len(fleet.history)} history lines")
    SchemaRegistry(calibrate([e.line for e in fleet.history], OfflineAdapter()).snapshot).save(
        data_dir / "registry.json")
    control = ControlStore()
    control.put("settings", "server", {"calibrated": True, "setup_done": True, "calibrated_at": now_iso()})
    for d in fleet.devices:  # sources known and reviewed, as on a server past its setup
        sid = source_id(d.id, "firmware", "synth")
        control.put("source", sid, {"id": sid, "host": d.id, "service": "firmware", "source": "synth",
                                    "first_seen": now_iso(), "count": 0, "calibrated": True, "alias_of": None})
    db = None
    if store == "clickhouse":
        from .storage.clickhouse import ClickHouseStore

        db = f"logpipe_bench_{uuid.uuid4().hex[:8]}"
        log_store = ClickHouseStore.from_env(database=db)
    elif store == "mongodb":
        from .storage.mongo import MongoStore

        db = f"logpipe_bench_{uuid.uuid4().hex[:8]}"
        log_store = MongoStore.from_env(database=db)
    else:
        from .storage.memory import MemoryStore

        log_store = MemoryStore()
    server = Server(log_store, control, data_dir=data_dir, adapter_factory=lambda: get_adapter("offline"),
                    schedule_every_s=10**9)
    events = [{"host": e.device_id, "service": "firmware", "source": "synth", "message": e.line}
              for e in fleet.test]
    return server, log_store, db, events


def run(store: str = "memory", rates: tuple[int, ...] = DEFAULT_RATES, step_seconds: float = 10,
        max_rate: int | None = None, flood: bool = True,
        say: Callable[[str], None] = lambda m: print(m, flush=True)) -> dict:
    """The benchmark; returns its result (see the module docstring)."""
    from .server.core import Busy

    rates = tuple(r for r in rates if max_rate is None or r <= max_rate) or (max_rate or 1000,)
    started = datetime.now(UTC)
    with tempfile.TemporaryDirectory(prefix="logpipe-bench-") as tmp:
        server, log_store, db, events = _setup(store, Path(tmp), say)
        parse, inserts = _Recorder(), _Recorder()
        server.process = parse.wrap(server.process)  # the ingest thread looks it up per line
        if (client := getattr(log_store, "client", None)) is not None:
            client.insert = inserts.wrap(client.insert)
        elif store == "mongodb":
            log_store.insert_logs = inserts.wrap(log_store.insert_logs)
        cursor = 0

        def batch(n: int) -> list[dict]:
            nonlocal cursor
            ts = datetime.now(UTC).isoformat()
            out = []
            for _ in range(n):
                out.append(events[cursor % len(events)] | {"ts": ts})
                cursor += 1
            return out

        def step(rate: int | None) -> dict:
            """``rate`` lines/s for ``step_seconds`` (None: as fast as the intake takes them)."""
            parse.take()
            inserts.take()
            offered = rejected = 0
            received0, cpu0, t0 = server.received, sum(os.times()[:2]), time.monotonic()
            peak = _rss_mb()
            tick = 0.05
            while time.monotonic() - t0 < step_seconds:
                if rate is None and server.inbox.qsize() > 20_000:
                    time.sleep(0.01)  # flood keeps the buffer full without starving the server of the GIL
                    continue
                n = 2000 if rate is None else max(1, round(rate * tick))
                try:
                    server.ingest(batch(n))
                    offered += n
                except Busy:
                    rejected += n
                    time.sleep(0.005)
                    continue
                if rate is not None:
                    due = t0 + offered / rate
                    time.sleep(max(0.0, due - time.monotonic()))
                peak = max(peak, _rss_mb())
            dt = time.monotonic() - t0
            processed = server.received - received0
            cpu = 100 * (sum(os.times()[:2]) - cpu0) / dt
            backlog = server.inbox.qsize()
            ins = inserts.take()
            res = {
                "target": rate, "offered_per_s": round(offered / dt), "processed_per_s": round(processed / dt),
                "rejected": rejected, "backlog": backlog, "cpu_pct": round(cpu, 1),
                "rss_mb": round(peak, 1), "parse_us": _pcts(parse.take(), 1000),
                "inserts": len(ins), "insert_ms": _pcts(ins, 1e6),
            }
            res["kept_up"] = bool(rate is not None and not rejected and processed >= KEEP_UP * offered
                                  and backlog <= max(rate * 0.5, 100))
            # let the backlog drain so the next step starts clean
            deadline = time.monotonic() + 30
            while server.inbox.qsize() and time.monotonic() < deadline:
                time.sleep(0.05)
            return res

        steps: list[dict] = []
        disk = None
        try:
            say("warming up")
            step_seconds, real = min(2, step_seconds), step_seconds
            step(rates[0])
            step_seconds = real
            for rate in rates:
                say(f"step {len(steps) + 1}: {rate:,} lines/s for {step_seconds:g} s")
                steps.append(step(rate))
                s = steps[-1]
                say(f"  processed {s['processed_per_s']:,}/s, CPU {s['cpu_pct']}%, "
                    f"parse p99 {s['parse_us'].get('p99', 0)} µs" + ("" if s["kept_up"] else " (fell behind)"))
                if not s["kept_up"]:
                    break
            if flood:
                say(f"flood: as fast as the intake takes lines for {step_seconds:g} s")
                steps.append(step(None))
                say(f"  processed {steps[-1]['processed_per_s']:,}/s, CPU {steps[-1]['cpu_pct']}%")
        finally:
            server.close()
            if store == "mongodb":  # bytes on disk for the rows written, measured before the database goes
                from .storage.mongo import MongoStore

                again = MongoStore.from_env(database=db)  # server.close() closed the first client
                st = again.db.command("dbStats")
                disk = {"lines": again.db["logs"].count_documents({}),
                        "data_bytes": st["storageSize"], "index_bytes": st["indexSize"]}
                again.db.client.drop_database(db)
                again.close()
            elif db:
                log_store.client.command(f"DROP DATABASE IF EXISTS `{db}`")  # type: ignore[attr-defined]
    kept = [s for s in steps if s["kept_up"]]
    behind = next((s for s in steps if s["target"] and not s["kept_up"]), None)
    return {
        "id": uuid.uuid4().hex[:12], "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"), "store": store,
        "step_seconds": step_seconds,
        "host": {"cpus": os.cpu_count(), "machine": platform.machine(), "system": platform.system(),
                 "python": platform.python_version(), "commit": _commit(), "hostname": platform.node()},
        "max_sustained_per_s": max((s["processed_per_s"] for s in kept), default=0),
        "falls_behind_at": behind["target"] if behind else None,
        "flood_per_s": steps[-1]["processed_per_s"] if flood and steps and steps[-1]["target"] is None else None,
        "peak_rss_mb": max((s["rss_mb"] for s in steps), default=0),
        "steps": steps,
    } | ({"disk": disk} if disk else {})


def table(result: dict) -> str:
    rows = [(f"{'target/s':>10} {'offered/s':>10} {'processed/s':>12} {'CPU %':>7} {'RSS MB':>7} "
             f"{'p50 µs':>8} {'p99 µs':>8} {'insert p99 ms':>13}  ok")]
    for s in result["steps"]:
        rows.append(f"{s['target'] or 'flood':>10} {s['offered_per_s']:>10,} {s['processed_per_s']:>12,} "
                    f"{s['cpu_pct']:>7} {s['rss_mb']:>7} {s['parse_us'].get('p50', '-'):>8} "
                    f"{s['parse_us'].get('p99', '-'):>8} {s['insert_ms'].get('p99', '-'):>13}  "
                    f"{'yes' if s['kept_up'] else ('-' if s['target'] is None else 'NO')}")
    rows.append(f"\nmax sustained: {result['max_sustained_per_s']:,} lines/s"
                + (f"; falls behind at {result['falls_behind_at']:,}/s" if result["falls_behind_at"] else "")
                + (f"; flood ceiling {result['flood_per_s']:,}/s" if result["flood_per_s"] else "")
                + f"; peak RSS {result['peak_rss_mb']} MB; {result['host']['cpus']} CPUs; store {result['store']}")
    return "\n".join(rows)


def main(store: str, max_rate: int | None, step_seconds: float, out: Path) -> dict:
    result = run(store=store, max_rate=max_rate, step_seconds=step_seconds)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("\n" + table(result), flush=True)
    print(f"\nwrote {out}", flush=True)
    return result


if __name__ == "__main__":  # the server's bench job runs it like this, in its own process
    main(sys.argv[1], int(sys.argv[2]) or None, float(sys.argv[3]), Path(sys.argv[4]))
