from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from deepparse.api import synth_masks
from deepparse.masks_types import Mask

from ..parsing.kv import default_keys


class LLMAdapter(Protocol):
    def synth_masks(self, lines: Sequence[str], sample_size: int) -> list[Mask]: ...

    def name_keys(self, tokens: Sequence[str], samples: Sequence[str]) -> list[str]: ...


class OfflineAdapter:
    MODE = "offline"

    def synth_masks(self, lines: Sequence[str], sample_size: int) -> list[Mask]:
        return [Mask.from_dict(m) for m in synth_masks(lines, sample_size, mode=self.MODE)]

    def name_keys(self, tokens: Sequence[str], samples: Sequence[str]) -> list[str]:
        return default_keys(tokens)


class HFDeepSeekAdapter(OfflineAdapter):
    MODE = "hf"


def get_adapter(backend: str) -> LLMAdapter:
    adapters = {"offline": OfflineAdapter, "hf": HFDeepSeekAdapter}
    if backend not in adapters:
        raise ValueError(f"unknown LLM backend {backend!r}; expected one of {sorted(adapters)}")
    return adapters[backend]()
