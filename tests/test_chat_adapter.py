import deepparse.synth.chat_provider as chat_provider
from deepparse.masks_types import Mask

from logpipe.enrich.llm import ChatAdapter, get_adapter


class FakeRunnable:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, _messages):
        fields = {"keys": ["state"]} if "keys" in self.schema.model_fields else {"same_event": True}
        return {"parsed": self.schema(**fields)}


class FakeProvider:
    default_model = "claude"

    def __init__(self, events: list[str]):
        self.events = events

    def build(self, **_kwargs):
        return object()

    def structured(self, _chat, schema):
        return FakeRunnable(schema)

    def close(self):
        self.events.append("close")


def test_web_adapter_shares_one_browser_and_frees_it_for_mask_synthesis(monkeypatch):
    events: list[str] = []
    configs = []

    def load(config):
        configs.append(config)
        events.append("open")
        return FakeProvider(events), ""

    def synthesize(_sample, config):
        configs.append(config)
        events.append("synth")
        return [Mask("STATE", r"\bopen\b", "door state")]

    monkeypatch.setattr(chat_provider, "_load_provider", load)
    monkeypatch.setattr(chat_provider, "synthesize_llm", synthesize)
    adapter = get_adapter("llm:web:gemini", browser="chrome")
    assert isinstance(adapter, ChatAdapter)

    assert adapter.name_keys(["Door", "<*>"], ["Door open"]) == ["state"]
    assert adapter.confirm_alias(["Door", "<*>"], ["Door", "is", "<*>"], ["Door open"])
    masks = adapter.synth_masks(["Door open", "Door closed"], 10)

    # One provider for both schemas, closed before synthesis opens its own.
    assert events == ["open", "close", "synth"]
    assert [m.label for m in masks] == ["STATE"]
    assert {(c.provider, c.model_name, c.browser) for c in configs} == {("web", "gemini", "chrome")}
