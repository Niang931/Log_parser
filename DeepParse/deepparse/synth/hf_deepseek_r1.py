# mypy: ignore-errors
"""Hugging Face inference backend for the DeepParse mask synthesiser.

Loads a base model (paper default: ``deepseek-ai/DeepSeek-R1-Distill-Llama-8B``)
plus an optional LoRA adapter produced by :mod:`deepparse.training.finetune`,
then prompts it with the exact instruction template used during training
(listing 2 of the paper).  The model is asked to emit a Python list of
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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from deepparse.logging_utils import get_logger
from deepparse.masks_types import Mask
from deepparse.synth.prompt_templates import (
    MASK_SYNTH_PROMPT,  # noqa: F401  (kept for backwards compat)
)
from deepparse.utils.regex_library import canonical_masks, validate_regexes

LOGGER = get_logger(__name__)

INSTRUCTION = (
    "Generate a Python list of regex patterns that capture the dynamic "
    "(variable) parts in the input log message while preserving the static "
    "structure."
)

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Output:\n"

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


def _parse_regex_list(raw: str) -> list[str]:
    """Best-effort safe parser for the model's textual output."""
    snippet = raw.strip()

    # retrieve only the list, not respond template outside
    if "[" in snippet:
        snippet = snippet[snippet.index("[") :]
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
    """Derive the variable *category* of a regex from its shape.

    The label matters beyond readability: the Drain stage keys on the typed
    placeholder, so two regexes that can match the same slot must carry the
    same label (every pure-digit pattern is ``NUMBER``) or a label whose
    priority in ``masks_application`` settles the overlap the same way on
    every line (``HEXSTR`` beats the generic ``WORDVAL`` on a sha256 whether
    the digest starts with a digit or a letter).  Otherwise the slot renders
    as two different tokens and Drain wildcards the whole ``key=value``.
    """
    p = pattern.lower()
    if pattern.startswith('(?<=[=\\s]")') or '[^"]*' in pattern:
        return "QUOTED"
    if pattern.startswith("(?<==)"):
        return "WORDVAL"
    if "(?::[0-9a-fa-f]{2}){5}" in p:
        return "MAC"
    if "0x" in p and "f" in p:
        return "HEX"
    if "uuid" in p or ("8}" in pattern and "12}" in pattern):
        return "UUID"
    if re.search(r"\[0-9a-fa-f\]\{\d+,?\d*\}", p):
        return "HEXSTR"  # long hex digests without 0x (sha256, EPC tags)
    if "trace|debug|info" in p:
        return "LOGLEVEL"
    if "ipv4" in p or ("\\d{1,3}\\." in pattern and ".){3}" in pattern):
        return "IPV4"
    if "blk_" in p:
        return "BLK"
    if "true|false" in p:
        return "BOOLEAN"
    if r"\d+x\d+" in pattern:
        return "DIMS"
    if r"\d{3}_\d{2,3}" in pattern:
        return "MCCMNC"
    if r"\d+\.\d+\.\d+" in pattern:
        return "SEMVER"
    if pattern.startswith(r"\d{4}") or "datetime" in p or "timestamp" in p:
        return "TIMESTAMP"
    if "(?:[a-za-z]+-)+" in p:
        return "ID"  # dash-joined identifiers: TH-75230, gw-edge-53
    if pattern.startswith("/") or "/[a-z" in p or "(?<!\\s)/" in p or "(?<=[\\s=])/" in p:
        return "PATH"
    if not re.search(r"[a-z]", re.sub(r"\\[dswb]", "", p)):
        return "NUMBER"  # digits-only shapes: \d{6}, -?\d+(?:\.\d+)?, \d{6}\s+\d{6}
    if r"-?\d" in pattern or pattern.endswith(r"\d+"):
        return "NUMBER"
    return f"VAR{index}"


# Plain static words a sane variable regex must never swallow whole.  A pattern
# that fully matches any of them (``[\w.\-:/]+``, ``\b(?:[A-Za-z_][\w.-]*)+``
# and every other spelling of "any token") would turn static template text
# into wildcards and collapse the whole dataset; the exact-string deny-list in
# ``masks_application`` cannot enumerate those spellings, so they are caught
# behaviourally here (paper Section 4.5: masks that fail validation are dropped).
_OVER_BROAD_PROBES = ("hello", "connection", "started", "Thunderbolt", "pressure", "kernel")


def _is_over_broad(compiled: re.Pattern[str]) -> bool:
    return any(compiled.fullmatch(word) for word in _OVER_BROAD_PROBES)


def _to_masks(patterns: Sequence[str]) -> list[Mask]:
    # only add unseen paterns and validate the regex before creating mask instances
    masks: list[Mask] = []
    seen_pattern: set[str] = set()
    forbidden = {r"[^\s]+", r"[\w.\-:/]+", r".*", r".+"}
    for idx, pattern in enumerate(patterns):
        if not pattern or pattern in seen_pattern or pattern in forbidden:
            continue
        seen_pattern.add(pattern)
        try:
            compiled = re.compile(pattern)
        except re.error:
            LOGGER.warning("dropping invalid pattern: %r", pattern)
            continue
        if _is_over_broad(compiled):
            LOGGER.warning("dropping over-broad pattern (matches plain words): %r", pattern)
            continue
        masks.append(
            Mask(label=_build_label(pattern, idx), pattern=pattern, justification="LLM-synthesised")
        )
    return masks


def _ensure_core_classes(masks: list[Mask]) -> list[Mask]:
    """Paper safety net: backfill the four core categories on validation failure."""
    present = {m.label for m in masks}
    # TODO: isnt this shit pre-defined in stub file, why re-declare it here ?
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
    load_in_4bit: bool = False,
    quantization_config=None,
    temperature: float = 0.0,
    num_beams: int = 1,
    max_length: int = 512,
    device: str = "auto",
    self_consistency_attempts: int = 2,
    **kwargs,
) -> list[Mask]:
    """Synthesise a regex mask bundle by prompting an LLM per listing 2."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face mode requires the optional 'hf' extra. Install with:\n"
            '    pip install -e ".[hf]"'
        ) from exc

    LOGGER.info(
        "loading base model %s%s (4bit=%s)", model_name, f" + adapter {adapter_path}" if adapter_path else "", load_in_4bit or (quantization_config is not None)
    )

    # Loading the tokenizer, apparently the tokenizer can also be retrieved from the LLM model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initializing the model
    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    use_4bit = load_in_4bit or (quantization_config is not None)
    if use_4bit:
        if quantization_config is None:
            from transformers import BitsAndBytesConfig
            compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "Loading a LoRA adapter requires the 'peft' package "
                "(installed as part of the 'hf' or 'train' extra)."
            ) from exc
        model = PeftModel.from_pretrained(model, adapter_path)

    if not use_4bit:
        target_device = "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
        model.to(target_device)
    else:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
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
        base_prompt = PROMPT_TEMPLATE.format(instruction=INSTRUCTION, input=line)

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
                    base_prompt + "Return ONLY a Python list of raw regex strings, e.g. "
                    '[r"\\d+", r"\\b[A-Z]+\\b"]. No prose, no markdown.\n\n'
                    "### Output:\n"
                )
                attempt_kwargs = dict(gen_kwargs)
                attempt_kwargs["do_sample"] = True
                attempt_kwargs["temperature"] = 0.3

            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(target_device)

            # Generation is performed here tokenized prompt and the kwargs
            with torch.no_grad():
                out_ids = model.generate(**inputs, **attempt_kwargs)

            # Decode the output from the LLM
            decoded = tokenizer.decode(
                out_ids[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )

            # Validate by running regex evaluation on the result
            patterns = _parse_regex_list(decoded)

            # If successful ten no need to retry
            if patterns:
                break

            LOGGER.debug("self-consistency retry %d for line: %r", attempt + 1, line[:80])

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


def synthesize_hf_from_checkpoint(
    checkpoint_dir: str | Path, logs: Sequence[str], **kwargs
) -> list[Mask]:
    """Detect whether ``checkpoint_dir`` is a base model or a LoRA adapter
    and delegate to :func:`synthesize_hf`.
    """
    import json

    # Just a wrapper around synthesize_hf for loading the check point model
    checkpoint_dir = Path(checkpoint_dir)
    cfg_path = checkpoint_dir / "deepparse_finetune_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if "load_in_4bit" not in kwargs and cfg.get("load_in_4bit"):
            kwargs["load_in_4bit"] = True
        return synthesize_hf(
            logs,
            model_name=cfg["model_name"],
            adapter_path=str(checkpoint_dir),
            **kwargs,
        )
    return synthesize_hf(logs, model_name=str(checkpoint_dir), **kwargs)
