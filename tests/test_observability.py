import json
import socket
from contextlib import nullcontext

import pytest

from logpipe.observability import Datadog
from logpipe.settings import DatadogSettings


@pytest.fixture
def settings(tmp_path):
    return DatadogSettings(log_file=tmp_path / "dd.jsonl", service="shop")


@pytest.fixture(autouse=True)
def _no_llmobs(monkeypatch):
    """LLMObs.enable is process-wide; never really turn it on in tests."""
    monkeypatch.setattr("ddtrace.llmobs.LLMObs.enable", lambda **_: None)


def test_off_by_default_and_calls_are_noops():
    dd = Datadog()
    assert not dd.enabled and not dd.logging_on() and not dd.llm_on
    with dd.stage("parse"):
        pass
    dd.incr("lines", 1, "status:final")
    dd.gauge("enrich.backlog", 3)
    dd.log(message="x")


def test_bad_host_turns_off_not_crash(monkeypatch, settings):
    def boom(_host):
        raise socket.gaierror

    monkeypatch.setattr(socket, "gethostbyname", boom)
    dd = Datadog(settings)
    assert not dd.enabled and not dd.logging_on()
    with dd.stage("parse"):
        pass


def test_llmobs_enable_error_means_llm_off(monkeypatch, settings):
    def no_key(**_):
        raise ValueError("no key")

    monkeypatch.setattr("ddtrace.llmobs.LLMObs.enable", no_key)
    dd = Datadog(settings)
    assert dd.enabled and not dd.llm_on


def test_on_times_stage_and_counts_error_then_reraises(monkeypatch, settings):
    dd = Datadog(settings)  # UDP to a closed port: no Agent needed
    assert dd.enabled
    sent = []
    monkeypatch.setattr(dd._statsd, "distribution", lambda m, v, tags: sent.append(("d", m, tags)))
    monkeypatch.setattr(dd._statsd, "increment", lambda m, *a, tags: sent.append(("i", m, tags)))
    with pytest.raises(RuntimeError), dd.stage("store"):
        raise RuntimeError
    assert sent == [
        ("i", "logpipe.stage.errors", ["stage:store"]),
        ("d", "logpipe.stage.duration", ["stage:store"]),
    ]


def test_log_uses_configured_service(settings):
    dd = Datadog(settings)
    dd.log(message="hello")
    assert json.loads(settings.log_file.read_text())["service"] == "shop"


def test_traced_is_identity_when_off_and_wraps_when_on(monkeypatch, settings):
    class Fake:
        def __init__(self):
            self.key_vocab: list[str] = []

        def name_keys(self, tokens):
            return list(tokens)

    fake = Fake()
    assert Datadog().traced(fake, "offline") is fake

    spans = []
    monkeypatch.setattr("ddtrace.llmobs.LLMObs.llm", lambda **kw: spans.append(kw) or nullcontext())
    monkeypatch.setattr("ddtrace.llmobs.LLMObs.annotate", lambda **kw: None)
    wrapped = Datadog(settings).traced(fake, "llm:anthropic:m")
    assert wrapped.name_keys(["a"]) == ["a"]
    assert spans == [
        {"model_name": "llm:anthropic:m", "model_provider": "anthropic", "name": "name_keys"}
    ]
    wrapped.key_vocab = ["x"]  # writes through, like CountedAdapter
    assert fake.key_vocab == ["x"]


def test_cli_turns_datadog_on_only_with_the_flag(monkeypatch, tmp_path):
    from logpipe import cli

    monkeypatch.setenv("DD_LOG_FILE", str(tmp_path / "dd.jsonl"))
    seen = []
    monkeypatch.setattr(cli, "_run", lambda args: seen.append(cli._datadog(args).enabled))
    cli.main(["run"])
    cli.main(["run", "--datadog"])
    assert seen == [False, True]


def test_apm_patches_only_when_asked(monkeypatch, tmp_path):
    calls = []
    for name in ("PEER_SERVICE_DEFAULTS_ENABLED", "REMOVE_INTEGRATION_SERVICE_NAMES_ENABLED"):
        monkeypatch.setenv(f"DD_TRACE_{name}", "true")  # restored after the test
    monkeypatch.setenv("DD_SERVICE", "x")
    monkeypatch.setattr("ddtrace.patch", lambda **kw: calls.append(kw))
    Datadog(DatadogSettings(log_file=tmp_path / "a.jsonl"))
    assert calls == []
    Datadog(DatadogSettings(log_file=tmp_path / "b.jsonl", apm=True))
    assert calls and calls[0]["fastapi"] and calls[0]["pymongo"]
