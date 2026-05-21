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
# Entropy-greedy sampling
# ---------------------------------------------------------------------------

def _tokenize_line(line: str) -> list[str]:
    """Split a log line into tokens for entropy calculation."""
    return re.findall(r'[A-Za-z0-9_\-\.]+', line)


def entropy_greedy_sample(lines: list[str], k: int = 5) -> list[str]:
    """
    Entropy-greedy sampling (doc section 1.3).

    Greedily selects k lines that maximise token-level entropy across
    the sample set. This ensures the LLM sees the most structurally
    diverse representatives, producing higher-quality templates.

    Doc claim: equal or better template quality at ~90% fewer tokens
    vs sending all lines.

    Args:
        lines: candidate log lines
        k:     number of lines to select (default 5, doc recommends 3-5)
    Returns:
        Up to k most diverse lines
    """
    if len(lines) <= k:
        return list(lines)

    # Count token frequencies across all lines
    global_freq: Counter = Counter()
    tokenised = [_tokenize_line(l) for l in lines]
    for toks in tokenised:
        global_freq.update(set(toks))  # set: count unique per line

    # Entropy of a token = -log2(p) where p = fraction of lines containing it
    n = len(lines)

    def token_entropy(tok: str) -> float:
        p = global_freq[tok] / n
        if p <= 0 or p >= 1:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    # Score each line by sum of entropy of its UNIQUE tokens
    def line_score(toks: list[str], already_seen: set[str]) -> float:
        new_toks = set(toks) - already_seen
        if not new_toks:
            return 0.0
        return sum(token_entropy(t) for t in new_toks)

    selected: list[str] = []
    seen_tokens: set[str] = set()

    for _ in range(k):
        best_idx = -1
        best_score = -1.0
        for i, (line, toks) in enumerate(zip(lines, tokenised)):
            if line in selected:
                continue
            sc = line_score(toks, seen_tokens)
            if sc > best_score:
                best_score = sc
                best_idx = i
        if best_idx == -1:
            break
        selected.append(lines[best_idx])
        seen_tokens.update(set(tokenised[best_idx]))

    return selected


# ---------------------------------------------------------------------------
# Prompt construction with RAG grounding
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """You are a regex-mask engineer specialising in semiconductor equipment logs.
Your task: generate Python regex patterns to mask all variable tokens in silicon-fab log lines.

Rules
-----
1. Return ONLY a JSON array — no preamble, no markdown fences, no commentary.
2. Each element: {"regex": "<valid Python regex>", "mask_with": "<TOKEN_NAME>"}
3. Patterns must be valid Python re module patterns.
4. Order by specificity: most specific (longest, fewest wildcards) FIRST.
5. Do NOT duplicate patterns.
6. Token names must be UPPER_SNAKE_CASE inside angle brackets e.g. <LOT_ID>.
7. Cover: numeric values, IDs, timestamps, IP addresses, hex, file paths,
   equipment constants, recipe IDs, error codes, vendor tokens, YAML keys,
   INI section.key values, XML attribute values, TSV column values."""


def _build_synthesis_prompt(
        sample_lines: list[str],
        rag_templates: list[str],
        round_hint: str = "",
) -> str:
    numbered = "\n".join(f"  {i + 1:3d}. {l}" for i, l in enumerate(sample_lines))
    hint_block = f"\nFocus: {round_hint}\n" if round_hint else ""

    rag_block = ""
    if rag_templates:
        rag_block = (
                "\nSimilar resolved templates (few-shot grounding — "
                "use consistent variable names):\n" +
                "\n".join(f"  • {t}" for t in rag_templates) + "\n"
        )

    return (
        f"{_SYSTEM_INSTRUCTION}\n"
        f"{rag_block}"
        f"{hint_block}\n"
        f"Log lines to analyse:\n{numbered}\n\n"
        "Return JSON array now:"
    )

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
