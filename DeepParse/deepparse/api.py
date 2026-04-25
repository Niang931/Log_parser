"""Lightweight public API mirroring the example usage from the paper.

The two helpers here implement Listing 1 of the paper verbatim::

    patterns = synth_masks(sys_logs, sample_size=50, temperature=0,
                           max_length=512)

    drain = Drain()
    drain.load_masks(patterns)
    parsed = drain.parse_all(sys_logs)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Union

from .drain.drain_engine import DrainEngine
from .masks_types import Mask
from .synth.r1_deepseek_stub import synthesize_offline
from .utils.regex_library import validate_regexes
from .utils.sampling import deterministic_sample

try:  # Optional heavy dependency
    from .synth.hf_deepseek_r1 import synthesize_hf as _synthesize_hf
except Exception:  # pragma: no cover - optional path
    _synthesize_hf = None  # type: ignore[assignment]

MaskLike = Union[Mask, dict]


def _ensure_mask_objects(masks: Iterable[MaskLike]) -> List[Mask]:
    converted: List[Mask] = []
    for mask in masks:
        if isinstance(mask, Mask):
            converted.append(mask)
        else:
            converted.append(Mask.from_dict(mask))
    return converted


def synth_masks(
    logs: Sequence[str],
    sample_size: int = 50,
    *,
    mode: str = "offline",
    temperature: float = 0.0,
    # Paper, Section "Prompt Engineering and Inference": "we use greedy
    # decoding (temperature zero) to minimize output variance".  Greedy
    # decoding == num_beams=1; defaulting to 2 silently turned on beam
    # search and contradicted the paper protocol.
    num_beams: int = 1,
    max_length: int = 512,
    strict: bool = False,
    model_name: str | None = None,
    adapter_path: str | None = None,
) -> List[dict]:
    """Synthesise a regex mask bundle from raw log lines.

    Returns a list of ``{"label", "pattern", "justification"}``
    dictionaries.  In ``mode="offline"`` (the default) the result is
    deterministic and produced entirely on CPU.  ``mode="hf"`` invokes
    the optional Hugging Face pipeline with the requested generation
    controls.
    """
    if not logs:
        raise ValueError("Cannot synthesise masks from an empty log sequence")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    # This function is for performing greedy entropy for sample selection
    sample = deterministic_sample(logs, min(sample_size, len(logs)))
    if mode == "offline":
        masks = synthesize_offline(sample)
    elif mode == "hf":
        if _synthesize_hf is None:  # pragma: no cover - optional dependency
            raise RuntimeError("Hugging Face mode requested but transformers is unavailable")
        masks = _synthesize_hf(
            sample,
            model_name=model_name or "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            adapter_path=adapter_path,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
        )
    else:
        raise ValueError(f"Unsupported synthesis mode: {mode!r}")

    validate_regexes([mask.pattern for mask in masks], strict=strict)
    return [mask.to_dict() for mask in masks]


@dataclass
class Drain:
    """Convenience facade exposing ``load_masks``/``parse_all`` helpers.

    Defaults match the Drain3 baseline cited in the paper (depth 5,
    similarity threshold 0.4).  Override either to tune for unusual
    corpora.
    """

    depth: int = 5
    similarity_threshold: float = 0.4

    def __post_init__(self) -> None:
        self._engine = DrainEngine(
            depth=self.depth,
            similarity_threshold=self.similarity_threshold,
            masks=[],
        )

    def load_masks(self, masks: Iterable[MaskLike]) -> None:
        mask_objs = _ensure_mask_objects(masks)
        self._engine = DrainEngine(
            depth=self.depth,
            similarity_threshold=self.similarity_threshold,
            masks=mask_objs,
        )

    def parse_all(self, logs: Sequence[str]) -> List[str]:
        return self._engine.parse(logs)

    def parse_with_ids(self, logs: Sequence[str]) -> List[tuple[int, str]]:
        return self._engine.parse_with_ids(logs)

    def add_log(self, log: str) -> str:
        return self._engine.add_log(log).template_str()

    @property
    def num_clusters(self) -> int:
        return self._engine.num_clusters
