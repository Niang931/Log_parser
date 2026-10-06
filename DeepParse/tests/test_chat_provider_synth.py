"""Tests for mask synthesis through chat providers (synth mode "llm"), with no network."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

from deepparse import synth_masks  # noqa: E402
from deepparse.ai_scraper import Reply  # noqa: E402
from deepparse.providers.web import WebProvider  # noqa: E402
from deepparse.synth import chat_provider  # noqa: E402
from deepparse.synth.chat_provider import (  # noqa: E402
    LLMSynthConfig,
    MaskResponse,
    MissingApiKeyError,
    synthesize_llm,
)

LOGS = [
    "2024-01-15 10:30:45 INFO Received block blk_-1608999687919862906 from 10.250.19.102",
    "2024-01-15 10:30:46 WARN Received block blk_7503483334202473044 from 10.250.10.6",
]

REPLY = {
    "masks": [
        {"label": "blk", "pattern": r"blk_-?\d+", "justification": "HDFS block id"},
        {"label": "IPV4", "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "justification": "address"},
        {"label": "dup", "pattern": r"blk_-?\d+", "justification": "same pattern again"},
        {"label": "BROKEN", "pattern": r"(unclosed", "justification": "does not compile"},
    ]
}


class FakeProvider:
    """A ChatProvider whose structured runnable replays canned replies."""

    name = "fake"
    default_model = "fake-model"

    def __init__(self, replies: list[MaskResponse | None]):
        self.replies = list(replies)
        self.calls: list[object] = []
        self.built: dict = {}
        self.closed = False

    def build(self, **kwargs):
        self.built = kwargs
        return object()

    def structured(self, model, schema):
        def run(messages):
            self.calls.append(messages)
            parsed = self.replies.pop(0)
            raw = AIMessage("", usage_metadata={
                "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
            })
            error = None if parsed else ValueError("no JSON")
            return {"raw": raw, "parsed": parsed, "parsing_error": error}

        return RunnableLambda(run)

    def price(self, model):
        return None

    def close(self):
        self.closed = True


@pytest.fixture
def fake(monkeypatch):
    def install(*replies):
        provider = FakeProvider(list(replies))
        monkeypatch.setattr(chat_provider, "_load_provider", lambda config: (provider, "key"))
        return provider

    return install


def test_masks_come_from_the_structured_reply(fake) -> None:
    provider = fake(MaskResponse.model_validate(REPLY))

    masks = synthesize_llm(LOGS, LLMSynthConfig(provider="fake", effort="high", seed=7))

    by_label = {mask.label: mask for mask in masks}
    assert by_label["BLK"].pattern == r"blk_-?\d+"
    assert by_label["BLK"].justification == "HDFS block id"
    assert "DUP" not in by_label, "a repeated pattern is kept once"
    assert "BROKEN" not in by_label, "a pattern that does not compile is dropped"
    # The paper's safety net still backfills the core classes the model left out.
    assert {"TIMESTAMP", "LOGLEVEL", "NUMBER", "IPV4"} <= set(by_label)
    assert provider.built["model"] == "fake-model"
    assert provider.built["effort"] == "high"
    assert provider.built["seed"] == 7
    assert provider.closed


def test_reprompts_when_a_reply_does_not_fit(fake) -> None:
    provider = fake(None, MaskResponse.model_validate(REPLY))

    masks = synthesize_llm(LOGS, LLMSynthConfig(provider="fake"))

    assert len(provider.calls) == 2
    assert any(mask.label == "BLK" for mask in masks)


def test_gives_up_on_core_classes_after_the_attempts(fake) -> None:
    provider = fake(None, None)

    masks = synthesize_llm(LOGS, LLMSynthConfig(provider="fake"))

    assert len(provider.calls) == 2
    assert {mask.label for mask in masks} == {"TIMESTAMP", "LOGLEVEL", "NUMBER", "IPV4"}


def test_the_sampled_logs_are_in_the_prompt(fake) -> None:
    provider = fake(MaskResponse.model_validate(REPLY))

    synthesize_llm(LOGS, LLMSynthConfig(provider="fake"))

    system, human = provider.calls[0]
    assert "regular expressions" in system.content
    assert all(line in human.content for line in LOGS)


def test_public_api_llm_mode(fake) -> None:
    fake(MaskResponse.model_validate(REPLY))

    patterns = synth_masks(LOGS, sample_size=2, mode="llm", provider="fake")

    assert {"label", "pattern", "justification"} <= set(patterns[0])
    assert any(entry["label"] == "BLK" for entry in patterns)


def test_a_missing_key_names_the_variable(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(MissingApiKeyError, match="GROQ_API_KEY"):
        synthesize_llm(LOGS, LLMSynthConfig(provider="groq"))


def test_gemini_accepts_either_key_variable(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")

    assert chat_provider.api_key_for("gemini") == "g-key"


def test_web_provider_parses_and_repairs_the_reply(monkeypatch) -> None:
    prompts: list[tuple[str, bool]] = []
    replies = [
        Reply(text="Sure! Here are some masks, I hope they help."),
        Reply(text="", code_blocks=[json.dumps(REPLY)]),
    ]

    def ask(self, site, prompt, *, new_chat=True, timeout=300):
        prompts.append((prompt, new_chat))
        return replies.pop(0)

    monkeypatch.setattr(WebProvider, "ask", ask)

    masks = synthesize_llm(LOGS, LLMSynthConfig(provider="web", model_name="claude"))

    assert any(mask.label == "BLK" for mask in masks)
    first, repair = prompts
    assert first[1] is True and "<json_schema>" in first[0] and LOGS[0] in first[0]
    assert repair[1] is False, "the repair turn stays in the same conversation"
