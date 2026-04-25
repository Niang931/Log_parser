# mypy: ignore-errors
"""Hugging Face inference backend for the DeepParse mask synthesiser.

Loads a base model (paper default: ``deepseek-ai/DeepSeek-R1-Distill-Llama-8B``)
plus an optional LoRA adapter produced by :mod:`deepparse.training.finetune`,
then prompts it with the exact instruction template used during training
(Listing 2 of the paper).  The model is asked to emit a Python list of
raw regex strings.

Output parsing is defensive: ``ast.literal_eval`` (a safe literal-only
parser, *not* ``eval``) is tried first; on failure we fall back to a
regex sweep that recovers individual ``r"..."`` literals.  Each pattern
is then validated through Python's ``re`` module.  Patterns that fail
validation are dropped and a fallback class is added so that the parser
still has the canonical four core variable categories (paper Section
'Prompt Engineering and Inference': "Masks that fail validation trigger
a fall-back rule that reverts to heuristic patterns for the affected
category, ensuring graceful degradation").

Self-consistency check (paper, same section): the LLM is re-prompted up
to ``self_consistency_attempts`` times if the parsed regex list is empty.
"""
from __future__ import annotations

import ast as _ast  # used only for literal_eval; aliased to make intent clear
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from ..logging_utils import get_logger
from ..masks_types import Mask
from ..utils.regex_library import canonical_masks, validate_regexes
from .prompt_templates import MASK_SYNTH_PROMPT  # noqa: F401  (kept for backwards compat)

LOGGER = get_logger(__name__)

INSTRUCTION = (
    "Generate a Python list of regex patterns that capture the dynamic "
    "(variable) parts in the input log message while preserving the static "
    "structure."
)

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Output:\n"
)

# Recover individual r"..." or r'...' strings from a free-form output.
_FALLBACK_PATTERN_RE = re.compile(r"""r[\"']([^\"']+)[\"']""")


@dataclass
class HFInferenceConfig:
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    adapter_path: str | None = None
    temperature: float = 0.0
    num_beams: int = 1
    max_new_tokens: int = 512
    device: str = "auto"
    self_consistency_attempts: int = 2


def _parse_regex_list(raw: str) -> List[str]:
    """Best-effort safe parser for the model's textual output."""
    snippet = raw.strip()
    if "[" in snippet:
        snippet = snippet[snippet.index("["):]
    if "]" in snippet:
        end = snippet.rfind("]")
        snippet = snippet[: end + 1]
    parsed = None
    try:
        # ast.literal_eval only parses Python literals (lists, strings,
        # numbers, etc.) and cannot execute code, so it is safe to apply
        # to untrusted model output.
        parsed = _ast.literal_eval(snippet)
    except (SyntaxError, ValueError):
        parsed = _FALLBACK_PATTERN_RE.findall(raw)
    if isinstance(parsed, list):
        return [str(p) for p in parsed if isinstance(p, str)]
    return []


def _build_label(pattern: str, index: int) -> str:
    """Heuristically derive a mask label from the regex shape."""
    p = pattern.lower()
    if "0x" in p and "f" in p:
        return "HEX"
    if "uuid" in p or ("8}" in pattern and "12}" in pattern):
        return "UUID"
    if "trace|debug|info" in p:
        return "LOGLEVEL"
    if "ipv4" in p or ("\\d{1,3}\\." in pattern and ".){3}" in pattern):
        return "IPV4"
    if "blk_" in p:
        return "BLK"
    if pattern.startswith(r"\d{4}") or "datetime" in p or "timestamp" in p:
        return "TIMESTAMP"
    if r"-?\d" in pattern or pattern.endswith(r"\d+"):
        return "NUMBER"
    if pattern.startswith("/") or "/[a-z" in p or "(?<!\\s)/" in p:
        return "PATH"
    return f"VAR{index}"


def _to_masks(patterns: Sequence[str]) -> List[Mask]:
    # only add unseen paterns and validate the regex before creating mask instances 
    masks: List[Mask] = []
    seen_pattern: set[str] = set()
    for idx, pattern in enumerate(patterns):
        if not pattern or pattern in seen_pattern:
            continue
        seen_pattern.add(pattern)
        try:
            re.compile(pattern)
        except re.error:
            LOGGER.warning("dropping invalid pattern: %r", pattern)
            continue
        masks.append(Mask(label=_build_label(pattern, idx),
                          pattern=pattern,
                          justification="LLM-synthesised"))
    return masks


def _ensure_core_classes(masks: List[Mask]) -> List[Mask]:
    """Paper safety net: backfill the four core categories on validation failure."""
    present = {m.label for m in masks}
    required = {"TIMESTAMP", "LOGLEVEL", "NUMBER", "IPV4"}
    if required.issubset(present):
        return masks
    by_label = {m.label: m for m in canonical_masks()}
    for label in required - present:
        masks.append(by_label[label])
    return masks


def synthesize_hf(
    logs: Sequence[str],
    *,
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    adapter_path: str | None = None,
    temperature: float = 0.0,
    num_beams: int = 1,
    max_length: int = 512,
    device: str = "auto",
    self_consistency_attempts: int = 2,
) -> List[Mask]:
    """Synthesise a regex mask bundle by prompting an LLM per Listing 2."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face mode requires the optional 'hf' extra. Install with:\n"
            "    pip install -e \".[hf]\""
        ) from exc

    LOGGER.info("loading base model %s%s", model_name,
                f" + adapter {adapter_path}" if adapter_path else "")

    # Loading the tokenizer, apparently the tokenizer can also be retrieved from the LLM model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initializing the model
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    # TODO: not sure about this LORA one
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "Loading a LoRA adapter requires the 'peft' package "
                "(installed as part of the 'hf' or 'train' extra)."
            ) from exc
        model = PeftModel.from_pretrained(model, adapter_path)

    target_device = "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
    model.to(target_device)
    model.eval()

    # Setting all of the configuration
    aggregated: list[str] = []
    seen: set[str] = set()
    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": max_length,
        "num_beams": num_beams,
        "do_sample": do_sample,
        "temperature": max(temperature, 1e-6) if do_sample else 1.0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    for line in logs:
        # fill in the instruction and input first
        base_prompt = PROMPT_TEMPLATE.format(
            instruction=INSTRUCTION, input=line)

        patterns: list[str] = []

        # Default seems to have 2 attempts
        for attempt in range(self_consistency_attempts):
            # Paper, Section "Prompt Engineering and Inference": the
            # self-consistency loop re-prompts with targeted feedback
            # describing the failure mode.  We approximate this here by
            # (1) varying the prompt on retry to nudge the model toward
            # emitting a clean Python list, and (2) bumping temperature
            # slightly off greedy on retry so the second attempt is
            # actually different from the first (a pure retry at
            # temperature 0 produces the same output and is a no-op).
            if attempt == 0:
                prompt = base_prompt
                attempt_kwargs = gen_kwargs

            else:
                # Second attempt seems to set it more strict with lower temperature
                prompt = (
                    base_prompt
                    + "Return ONLY a Python list of raw regex strings, e.g. "
                      '[r"\\d+", r"\\b[A-Z]+\\b"]. No prose, no markdown.\n\n'
                      "### Output:\n"
                )
                attempt_kwargs = dict(gen_kwargs)
                attempt_kwargs["do_sample"] = True
                attempt_kwargs["temperature"] = 0.3

            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(target_device)

            # TODO: add an online module here
            # Generation is performed here tokenized prompt and the kwargs
            with torch.no_grad():
                out_ids = model.generate(**inputs, **attempt_kwargs)

            # Decode the output from the LLM
            decoded = tokenizer.decode(
                out_ids[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            )

            # Validate by running regex evaluation on the result
            patterns = _parse_regex_list(decoded)

            # If successful ten no need to retry
            if patterns:
                break

            LOGGER.debug("self-consistency retry %d for line: %r", attempt + 1, line[:80])

        # NOTE: not sure what the seen here is used for ? the aggregated make sense though
        # Add brand new patterns to seen
        for pattern in patterns:
            if pattern not in seen:
                aggregated.append(pattern)
                seen.add(pattern)

    # return the masks after inference was complete
    masks = _to_masks(aggregated)
    masks = _ensure_core_classes(masks)
    validate_regexes([m.pattern for m in masks])

    LOGGER.info("synthesised %d unique masks", len(masks))
    return masks


def synthesize_hf_from_checkpoint(checkpoint_dir: str | Path,
                                  logs: Sequence[str],
                                  **kwargs) -> List[Mask]:
    """Detect whether ``checkpoint_dir`` is a base model or a LoRA adapter
    and delegate to :func:`synthesize_hf`.
    """
    import json

    # Just a wrapper around synthesize_hf for loading the check point model
    checkpoint_dir = Path(checkpoint_dir)
    cfg_path = checkpoint_dir / "deepparse_finetune_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return synthesize_hf(
            logs,
            model_name=cfg["model_name"],
            adapter_path=str(checkpoint_dir),
            **kwargs,
        )
    return synthesize_hf(logs, model_name=str(checkpoint_dir), **kwargs)
