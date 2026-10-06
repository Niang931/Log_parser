"""High level interface for mask synthesis (offline, Hugging Face, chat providers)."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from deepparse.dataset_loader import Dataset
from deepparse.logging_utils import get_logger
from deepparse.masks_types import Mask, MaskBundle
from deepparse.synth.r1_deepseek_stub import synthesize_offline
from deepparse.utils.regex_library import validate_regexes
from deepparse.utils.sampling import deterministic_sample

if TYPE_CHECKING:
    from deepparse.synth.chat_provider import LLMSynthConfig

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
    llm_config: LLMSynthConfig | None = None,
) -> MaskBundle:
    LOGGER.info("Synthesising masks for %s with mode=%s", dataset.name, mode)
    sample = deterministic_sample(dataset.logs, k)
    masks: Sequence[Mask]

    if mode == "offline":
        masks = synthesize_offline(sample)

    elif mode == "hf":
        from deepparse.synth.hf_deepseek_r1 import synthesize_hf, synthesize_hf_from_checkpoint

        # Either synth from a fine-tuned model or just the original DeepSeekModel
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
    elif mode == "llm":
        from deepparse.synth.chat_provider import synthesize_llm

        masks = synthesize_llm(sample, llm_config)
    else:
        raise UnsupportedModeError(mode)

    # TODO: the validation is actually getting ran twice, is this really necessary ?
    validate_regexes([mask.pattern for mask in masks], strict=strict)
    bundle = MaskBundle(dataset=dataset.name, masks=list(masks))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle.to_json(), indent=2), encoding="utf-8")
    LOGGER.info("Wrote %d masks to %s", len(masks), out_path)
    return bundle
