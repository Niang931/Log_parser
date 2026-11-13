"""Datadog hooks: metrics, JSON-line logs and LLM spans. ``Datadog()`` is off and every call is a
no-op; ``Datadog(settings)`` turns it on. Any failure means off: the pipeline never breaks."""

from __future__ import annotations

import atexit
import json
import socket
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext

from .settings import DatadogSettings

_OFF = nullcontext()  # reusable: a disabled `stage()` allocates nothing


class Datadog:
    """One per process: a DogStatsD client must not cross processes."""

    def __init__(self, settings: DatadogSettings | None = None) -> None:
        self._statsd = None
        self._log = None  # JSON-lines file the Datadog Agent tails
        self.llm_on = False
        self.service = settings.service if settings else "logpipe"
        if settings is None:
            return
        try:
            from datadog import DogStatsd

            # Resolve once: a bad hostname blocks every send ~3.8 s on DNS.
            ip = socket.gethostbyname(settings.agent_host)
            # Buffered: ~2 us per send, unbuffered ~14 us (measured), too much beside a us-scale parse.
            self._statsd = DogStatsd(host=ip, port=settings.dogstatsd_port, disable_buffering=False)
            settings.log_file.parent.mkdir(parents=True, exist_ok=True)
            # ponytail: line-buffered append, no rotation. Truncate by hand; add logrotate if it grows.
            self._log = settings.log_file.open("a", encoding="utf-8", buffering=1)
            atexit.register(self._log.close)
        except Exception:  # noqa: BLE001  ImportError, gaierror, unwritable path: all mean "off"
            self._statsd = None
            return
        try:
            from ddtrace.llmobs import LLMObs

            # No Agent and no DD_API_KEY makes this raise, so that means off.
            LLMObs.enable(
                ml_app=settings.llmobs_ml_app,
                site=settings.site,
                agentless_enabled=False,
                # Off: the default patches fastapi/httpx/requests/openai/anthropic (a span
                # per intake request, and duplicates of the spans `traced()` makes).
                integrations_enabled=False,
            )
            self.llm_on = True
        except Exception:  # noqa: BLE001, S110  telemetry must never break the app
            pass

    @property
    def enabled(self) -> bool:
        return self._statsd is not None

    def stage(self, name: str) -> AbstractContextManager[None]:
        """`with dd.stage("parse"):` times the block and counts an exception, then re-raises it."""
        return _OFF if self._statsd is None else self._timed(name)

    @contextmanager
    def _timed(self, name: str) -> Iterator[None]:
        tags = [f"stage:{name}"]
        start = time.perf_counter_ns()
        try:
            yield
        except Exception:
            self._statsd.increment("logpipe.stage.errors", tags=tags)
            raise
        finally:
            self._statsd.distribution(
                "logpipe.stage.duration", (time.perf_counter_ns() - start) / 1e6, tags=tags
            )

    def incr(self, name: str, n: int = 1, *tags: str) -> None:
        if self._statsd is not None:
            self._statsd.increment(f"logpipe.{name}", n, tags=list(tags))

    def gauge(self, name: str, value: float, *tags: str) -> None:
        if self._statsd is not None:
            self._statsd.gauge(f"logpipe.{name}", value, tags=list(tags))

    def logging_on(self) -> bool:
        return self._log is not None

    def log(self, **fields: object) -> None:
        """One JSON line for the Agent. `message` is the raw log; no `timestamp`, so Datadog stamps
        arrival (it rejects replayed events older than its intake window)."""
        if self._log is not None:
            line = {"service": self.service, "ddsource": "logpipe", **fields}
            self._log.write(json.dumps(line, default=str) + "\n")

    def log_alerts(self, alerts) -> None:
        for a in alerts:
            self.log(
                message=f"anomaly on {a.device_id}: {','.join(a.reasons)}",
                status="warn",
                is_anomaly=True,
                anomaly_score=a.distance,
                device_id=a.device_id,
                reasons=list(a.reasons),
                miss_ratio=a.miss_ratio,
                novelty_ratio=a.novelty_ratio,
                template_ids=list(a.template_ids),
                window_start=a.window_start.isoformat(),
                window_end=a.window_end.isoformat(),
            )

    def traced(self, adapter, backend: str):
        """The adapter as is when LLM tracing is off."""
        return _Traced(adapter, backend) if self.llm_on else adapter


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

            with LLMObs.llm(
                model_name=self._backend,
                model_provider=parts[1] if len(parts) > 1 else parts[0],
                name=name,
            ):
                result = attr(*args, **kwargs)
                try:
                    # ponytail: raw args/result clipped to 2000 chars; they may hold log lines.
                    LLMObs.annotate(
                        input_data=repr((args, kwargs))[:2000], output_data=repr(result)[:2000]
                    )
                except Exception:  # noqa: BLE001, S110  telemetry must never break the app
                    pass
                return result

        return call
