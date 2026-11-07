"""Datadog hooks. Off unless DD_ENABLED=1. Any failure turns it off; the pipeline never breaks."""
from __future__ import annotations

import atexit
import json
import os
import socket
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path

_statsd = None
_log = None  # JSON-lines file the Datadog Agent tails
_llm_on = False
_OFF = nullcontext()  # reusable: disabled `stage()` allocates nothing


def init() -> bool:
    """Call once per process (a DogStatsD client must not cross processes)."""
    global _statsd, _log
    _statsd = None
    if _log is not None:
        _log.close()
        _log = None
    if os.environ.get("DD_ENABLED") != "1":
        return False
    try:
        from datadog import DogStatsd

        # Resolve once: a bad hostname blocks every send ~3.8 s on DNS.
        ip = socket.gethostbyname(os.environ.get("DD_AGENT_HOST", "127.0.0.1"))
        # Buffered: ~2 us per send, unbuffered ~14 us (measured), too much beside a us-scale parse.
        _statsd = DogStatsd(host=ip, port=int(os.environ.get("DD_DOGSTATSD_PORT", "8125")),
                            disable_buffering=False)
        path = Path(os.environ.get("DD_LOG_FILE", ".logpipe/dd-logs.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        # ponytail: line-buffered append, no rotation. Truncate by hand; add logrotate if it grows.
        _log = path.open("a", encoding="utf-8", buffering=1)
        atexit.register(_log.close)
    except Exception:  # noqa: BLE001  ImportError, gaierror, bad port, unwritable path: all mean "off"
        _statsd = None
        return False
    return True


def init_llm() -> bool:
    """Trace LLM calls. No Agent and no DD_API_KEY makes LLMObs.enable raise, so that means off."""
    global _llm_on
    _llm_on = False
    if os.environ.get("DD_LLMOBS_ENABLED") != "1":
        return False
    try:
        from ddtrace.llmobs import LLMObs

        LLMObs.enable(ml_app=os.environ.get("DD_LLMOBS_ML_APP", "logpipe"),
                      site=os.environ.get("DD_SITE", "datadoghq.com"), agentless_enabled=False,
                      # Off: the default patches fastapi/httpx/requests/openai/anthropic (a span per
                      # intake request, and duplicates of the spans `traced()` already makes).
                      integrations_enabled=False)
    except Exception:  # noqa: BLE001  telemetry must never break the app
        return False
    _llm_on = True
    return True


class _Traced:
    """Wraps an LLM adapter: every public call becomes one LLMObs span (same shape as CountedAdapter)."""

    def __init__(self, inner: object, backend: str) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_backend", backend)

    def __setattr__(self, name: str, value: object) -> None:  # e.g. the worker's key_vocab
        setattr(self._inner, name, value)

    def __getattr__(self, name: str):
        attr = getattr(self._inner, name)
        if not callable(attr) or name.startswith("_"):
            return attr
        parts = self._backend.split(":")

        def call(*args, **kwargs):
            from ddtrace.llmobs import LLMObs

            with LLMObs.llm(model_name=self._backend, model_provider=parts[1] if len(parts) > 1 else parts[0],
                            name=name):
                result = attr(*args, **kwargs)
                try:
                    # ponytail: raw args/result clipped to 2000 chars; they may hold log lines.
                    LLMObs.annotate(input_data=repr((args, kwargs))[:2000], output_data=repr(result)[:2000])
                except Exception:  # noqa: BLE001, S110  telemetry must never break the app
                    pass
                return result

        return call


def traced(adapter, backend: str):
    """The adapter as is when LLM tracing is off."""
    return _Traced(adapter, backend) if _llm_on else adapter


def stage(name: str) -> AbstractContextManager[None]:
    """`with stage("parse"):` times the block and counts an exception, then re-raises it."""
    return _OFF if _statsd is None else _timed(name)


@contextmanager
def _timed(name: str) -> Iterator[None]:
    tags = [f"stage:{name}"]
    start = time.perf_counter_ns()
    try:
        yield
    except Exception:
        _statsd.increment("logpipe.stage.errors", tags=tags)
        raise
    finally:
        _statsd.distribution("logpipe.stage.duration", (time.perf_counter_ns() - start) / 1e6, tags=tags)


def incr(name: str, n: int = 1, *tags: str) -> None:
    if _statsd is not None:
        _statsd.increment(f"logpipe.{name}", n, tags=list(tags))


def gauge(name: str, value: float, *tags: str) -> None:
    if _statsd is not None:
        _statsd.gauge(f"logpipe.{name}", value, tags=list(tags))


def logging_on() -> bool:
    return _log is not None


def log(**fields: object) -> None:
    """One JSON line for the Agent. `message` is the raw log; no `timestamp`, so Datadog stamps arrival
    (it rejects replayed events older than its intake window)."""
    if _log is not None:
        _log.write(json.dumps({"service": "logpipe", "ddsource": "logpipe", **fields}, default=str) + "\n")


def log_alerts(alerts) -> None:
    for a in alerts:
        log(message=f"anomaly on {a.device_id}: {','.join(a.reasons)}", status="warn", is_anomaly=True,
            anomaly_score=a.distance, device_id=a.device_id, reasons=list(a.reasons),
            miss_ratio=a.miss_ratio, novelty_ratio=a.novelty_ratio, template_ids=list(a.template_ids),
            window_start=a.window_start.isoformat(), window_end=a.window_end.isoformat())
