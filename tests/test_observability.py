import socket

import pytest

from logpipe import observability as obs


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("DD_ENABLED", "DD_LLMOBS_ENABLED", "DD_AGENT_HOST"):
        monkeypatch.delenv(k, raising=False)
    yield
    obs._statsd = None


def test_off_by_default_and_calls_are_noops():
    assert obs.init() is False
    with obs.stage("parse"):
        pass
    obs.incr("lines", 1, "status:final")
    obs.gauge("enrich.backlog", 3)


def test_bad_host_turns_off_not_crash(monkeypatch):
    monkeypatch.setenv("DD_ENABLED", "1")

    def boom(_host):
        raise socket.gaierror

    monkeypatch.setattr(socket, "gethostbyname", boom)
    assert obs.init() is False
    with obs.stage("parse"):
        pass


def test_llmobs_enable_error_means_off(monkeypatch):
    monkeypatch.setenv("DD_LLMOBS_ENABLED", "1")
    monkeypatch.setattr("ddtrace.llmobs.LLMObs.enable", lambda **_: (_ for _ in ()).throw(ValueError("no key")))
    assert obs.init_llm() is False


def test_on_times_stage_and_counts_error_then_reraises(monkeypatch):
    monkeypatch.setenv("DD_ENABLED", "1")
    assert obs.init() is True  # UDP to a closed port: no Agent needed
    sent = []
    monkeypatch.setattr(obs._statsd, "distribution", lambda m, v, tags: sent.append(("d", m, tags)))
    monkeypatch.setattr(obs._statsd, "increment", lambda m, *a, tags: sent.append(("i", m, tags)))
    with pytest.raises(RuntimeError), obs.stage("store"):
        raise RuntimeError
    assert sent == [("i", "logpipe.stage.errors", ["stage:store"]),
                    ("d", "logpipe.stage.duration", ["stage:store"])]


def test_traced_is_identity_when_off_and_wraps_when_on(monkeypatch):
    from contextlib import nullcontext

    class Fake:
        def __init__(self):
            self.key_vocab: list[str] = []

        def name_keys(self, tokens):
            return list(tokens)

    fake = Fake()
    assert obs.traced(fake, "offline") is fake

    spans = []
    monkeypatch.setattr(obs, "_llm_on", True)
    monkeypatch.setattr("ddtrace.llmobs.LLMObs.llm", lambda **kw: spans.append(kw) or nullcontext())
    monkeypatch.setattr("ddtrace.llmobs.LLMObs.annotate", lambda **kw: None)
    wrapped = obs.traced(fake, "llm:anthropic:m")
    assert wrapped.name_keys(["a"]) == ["a"]
    assert spans == [{"model_name": "llm:anthropic:m", "model_provider": "anthropic", "name": "name_keys"}]
    wrapped.key_vocab = ["x"]  # writes through, like CountedAdapter
    assert fake.key_vocab == ["x"]
