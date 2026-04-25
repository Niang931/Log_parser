"""High level interface for mask synthesis (offline + Hugging Face)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..dataset_loader import Dataset
from ..logging_utils import get_logger
from ..masks_types import Mask, MaskBundle
from ..utils.regex_library import validate_regexes
from ..utils.sampling import deterministic_sample
from .r1_deepseek_stub import synthesize_offline

LOGGER = get_logger(__name__)


class UnsupportedModeError(ValueError):
    pass


def synthesize_masks(
    dataset: Dataset,
    k: int,
    out_path: Path,
    mode: str = "offline",
    strict: bool = False,
    model_name: str | None = None,
    adapter_path: str | None = None,
) -> MaskBundle:
    LOGGER.info("Synthesising masks for %s with mode=%s", dataset.name, mode)
    sample = deterministic_sample(dataset.logs, k)
    masks: Sequence[Mask]
    if mode == "offline":
        masks = synthesize_offline(sample)
    elif mode == "hf":
        from .hf_deepseek_r1 import synthesize_hf, synthesize_hf_from_checkpoint

        if adapter_path and not model_name:
            # Read the base model name from the saved adapter config so
            # the user doesn't have to repeat it on every invocation.
            masks = synthesize_hf_from_checkpoint(adapter_path, sample)
        else:
            masks = synthesize_hf(
                sample,
                model_name=model_name or "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                adapter_path=adapter_path,
            )
    else:
        raise UnsupportedModeError(mode)

    validate_regexes([mask.pattern for mask in masks], strict=strict)
    bundle = MaskBundle(dataset=dataset.name, masks=list(masks))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle.to_json(), indent=2), encoding="utf-8")
    LOGGER.info("Wrote %d masks to %s", len(masks), out_path)
    return bundle
