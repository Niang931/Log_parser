"""
DeepParse/deepparse/synth/hf_deepseek_r1.py — LLM Mask Synthesis
=================================================================
Despite the module name (legacy from HuggingFace/DeepSeek R1 prototype),
this module uses whichever LLM is injected via `llm` parameter.

Key design patterns
-------------------
* Self-consistency voting       : N candidate mask sets → majority-vote per regex
* Entropy-greedy sampling       : select the most structurally diverse 3-5 lines to reduce token
* Template-cache RAG grounding  : incorporates the top three similar resolved templates as few-shot examples
* Deduplication + sorting       : most-specific (longest) patterns first
* Self-consistency              : voting across temperature-varied completions
* Structured JSON output        : forcing with robust fence stripping

"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from collections import Counter
from typing import Any

log = logging.getLogger("deepparse.synth")

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Template cache (in-memory RAG store)
# Populated by the pipeline as templates are resolved.
# ---------------------------------------------------------------------------
_TEMPLATE_CACHE: list[str] = []


def register_resolved_template(template: str) -> None:
    """Add a newly resolved template to the RAG cache."""
    if template and template not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE.append(template)
        # Cap cache size
        if len(_TEMPLATE_CACHE) > 500:
            _TEMPLATE_CACHE.pop(0)


def get_similar_templates(sample_lines: list[str], top_k: int = 3) -> list[str]:
    """
    Template-cache RAG (doc section 3.1).
    Find the top_k cached templates most similar to the sample lines
    using token overlap as a fast proxy for semantic similarity.
    """
    if not _TEMPLATE_CACHE:
        return []
    sample_tokens = set()
    for line in sample_lines[:5]:
        sample_tokens.update(re.findall(r'\b\w+\b', line.lower()))

    scored = []
    for tmpl in _TEMPLATE_CACHE:
        tmpl_tokens = set(re.findall(r'\b\w+\b', tmpl.lower()))
        overlap = len(sample_tokens & tmpl_tokens) / max(len(sample_tokens | tmpl_tokens), 1)
        scored.append((overlap, tmpl))

    scored.sort(reverse=True)
    return [t for _, t in scored[:top_k] if _ > 0.05]  # min 5% overlap
# ---------------------------------------------------------------------------
# JSON extraction — robust fence / partial-JSON handling
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_ARRAY_RE      = re.compile(r"\[.*\]", re.DOTALL)


def _extract_json_array(raw: str) -> list[dict]:
    """
    Extract JSON array from LLM response that may contain markdown fences,
    preamble text, or partial JSON. Returns empty list on failure.
    """
    # 1 — Try fenced block first
    m = _JSON_FENCE_RE.search(raw)
    if m:
        raw = m.group(1).strip()

    # 2 — Find outermost [...] array
    m2 = _ARRAY_RE.search(raw)
    if m2:
        raw = m2.group(0)

    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "masks" in result:
            return result["masks"]
    except json.JSONDecodeError as exc:
        log.debug("JSON parse failed: %s — raw=%.120s", exc, raw)

    # 3 — Line-by-line object extraction (last resort)
    objects = []
    for line in raw.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if "regex" in obj and "mask_with" in obj:
                    objects.append(obj)
            except json.JSONDecodeError:
                pass
    return objects


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _is_valid_mask(m: Any) -> bool:
    if not isinstance(m, dict):
        return False
    if "regex" not in m or "mask_with" not in m:
        return False
    try:
        re.compile(m["regex"])
        return True
    except re.error:
        return False


def _specificity_score(mask: dict) -> int:
    """Higher = more specific; used for ordering."""
    return len(mask.get("regex", ""))


# ---------------------------------------------------------------------------
# Self-consistency voting
# ---------------------------------------------------------------------------

def _vote_masks(all_candidates: list[list[dict]], min_votes: int = 1) -> list[dict]:
    """
    Majority-vote across N candidate mask sets.
    A mask survives if its regex appears in ≥ min_votes sets.
    min_votes=1 means union (permissive); higher = intersection (strict).
    """
    regex_counts: Counter = Counter()
    regex_to_mask: dict[str, dict] = {}

    for candidate_set in all_candidates:
        seen_in_this = set()
        for m in candidate_set:
            r = m.get("regex", "")
            if r and r not in seen_in_this:
                regex_counts[r] += 1
                regex_to_mask[r] = m
                seen_in_this.add(r)

    return [
        regex_to_mask[r]
        for r, count in regex_counts.items()
        if count >= min_votes
    ]


# ---------------------------------------------------------------------------
# Primary synthesis function
# ---------------------------------------------------------------------------

def synthesize_online(
    logs: list[str],
    llm: Any,
    max_length: int = 128,
    self_consistency_attempts: int = 3,
    round_hint: str = "",
    min_votes: int = 1,
) -> list[dict]:
    """
    Synthesise regex masks using an LLM with self-consistency voting.

    Parameters
    ----------
    logs                      : sample log lines (up to 30 used)
    llm                       : BaseLLM instance from llm.registry
    max_length                : max chars per line sent to LLM
    self_consistency_attempts : N independent completions to vote over
    round_hint                : optional focus hint for adaptive re-synthesis
    min_votes                 : minimum votes to include a mask in result

    Returns
    -------
    List of validated mask dicts: [{"regex": ..., "mask_with": ...}, ...]
    """
    sample = [l[:max_length] for l in logs[:30]]
    prompt = _build_synthesis_prompt(sample, round_hint=round_hint)

    log.info(
        "Synthesising masks: %d sample lines, %d attempts, provider=%s",
        len(sample), self_consistency_attempts, getattr(llm, "model", "?"),
    )

    candidate_sets: list[list[dict]] = []

    # Vary temperature to increase mask diversity across attempts
    temperatures = [0.0, 0.2, 0.4][:self_consistency_attempts]
    for i, temp in enumerate(temperatures):
        try:
            raw_output = llm.complete_with_retry(
                prompt,
                max_tokens=1500,
                temperature=temp,
            )
            masks = _extract_json_array(raw_output)
            valid = [m for m in masks if _is_valid_mask(m)]
            log.debug(
                "Attempt %d/%d temp=%.1f → %d raw / %d valid masks",
                i+1, self_consistency_attempts, temp, len(masks), len(valid),
            )
            if valid:
                candidate_sets.append(valid)
        except Exception as exc:
            log.warning("Synthesis attempt %d failed: %s", i+1, exc)

    if not candidate_sets:
        log.error("All synthesis attempts failed — returning empty mask list")
        return []

    # Vote and sort by specificity (most specific first)
    voted = _vote_masks(candidate_sets, min_votes=min_votes)
    voted.sort(key=_specificity_score, reverse=True)

    # Deduplicate by regex
    seen: set[str] = set()
    unique = []
    for m in voted:
        if m["regex"] not in seen:
            seen.add(m["regex"])
            unique.append(m)

    log.info(
        "Synthesis complete: %d candidate sets → %d unique masks (voted)",
        len(candidate_sets), len(unique),
    )
    return unique
