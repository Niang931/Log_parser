"""Demo incident for Datadog: healthy -> slow store -> failing store -> recovered.

Usage: python examples/datadog_incident.py [--logs DIR] [--registry FILE] [--phase SECONDS]
Always sends to Datadog (`make demo-datadog`). Expect `stage:store` p99 to jump, then `stage.errors`.
"""

from __future__ import annotations

import argparse
import itertools
import random
import time
from pathlib import Path

from logpipe.observability import Datadog
from logpipe.pipeline import Pipeline
from logpipe.registry import SchemaRegistry
from logpipe.settings import get_settings
from logpipe.sources.file_replay import replay
from logpipe.storage.memory import MemoryStore


class FaultyStore(MemoryStore):
    delay = 0.0  # seconds added to every insert
    fail = 0.0  # share of inserts that raise

    def insert_logs(self, rows) -> None:
        time.sleep(self.delay)
        if random.random() < self.fail:
            raise ConnectionError("injected: store unavailable")
        super().insert_logs(rows)


PHASES = [
    ("healthy", 0.0, 0.0),
    ("slow store", 0.003, 0.0),
    ("failing store", 0.003, 0.2),
    ("recovered", 0.0, 0.0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, default=Path("artifacts/fleet/test"))
    ap.add_argument("--registry", type=Path, default=Path("artifacts/registry.json"))
    ap.add_argument("--phase", type=float, default=60, help="seconds per phase")
    args = ap.parse_args()
    dd = Datadog(get_settings().datadog)
    if not dd.enabled:
        raise SystemExit(
            "Datadog could not start: install the `datadog` extra, check DD_AGENT_HOST"
        )
    store = FaultyStore()
    pipe = Pipeline.assemble(SchemaRegistry.load(args.registry), store, datadog=dd)
    events = itertools.cycle(list(replay(args.logs)))
    for name, store.delay, store.fail in PHASES:
        print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)
        end = time.monotonic() + args.phase
        while time.monotonic() < end:
            try:
                pipe.process(next(events))
            except ConnectionError:
                pass  # the stage timer already counted it; keep going like Server.process
    pipe.finish()


if __name__ == "__main__":
    main()
