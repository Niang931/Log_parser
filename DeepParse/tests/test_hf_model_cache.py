import sys
import types

from deepparse.synth import hf_deepseek_r1 as hf


def test_model_loads_once_across_systems(monkeypatch):
    loads = []

    class Auto:
        @staticmethod
        def from_pretrained(name, **kwargs):
            loads.append(name)
            return types.SimpleNamespace(pad_token="x", eval=lambda: None, to=lambda d: None)

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False), float32="fp32",
        bfloat16="bf16")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=Auto, AutoModelForCausalLM=Auto))
    hf._load_model.cache_clear()
    first = hf._load_model("base", None, False, None, "cpu")
    second = hf._load_model("base", None, False, None, "cpu")
    assert first is second and loads == ["base", "base"]  # tokenizer + model, loaded once
    hf._load_model.cache_clear()
