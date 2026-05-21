"""
pipeline/masks.py — Mask loading, validation and adaptive synthesis
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from pipeline.config import ADAPTIVE_MAX_RETRIES, ADAPTIVE_PARSE_THRESHOLD
from pipeline.telemetry import TELEMETRY, Telemetry

log = logging.getLogger("deepparse.masks")


# ---------------------------------------------------------------------------
# Load / save / validate
# ---------------------------------------------------------------------------

def load_masks_from_cache(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_masks(masks: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [m.to_dict() if hasattr(m, "to_dict") else m for m in masks]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_universal_masks(masks_path: Path | None = None) -> list[dict]:
    """Load masks_fab_universal.json — auto-detected if path not given."""
    import __main__
    candidates = [
        masks_path,
        Path(__file__).parent.parent / "masks_fab_universal.json",
        Path(__file__).parent.parent / "masks" / "masks_fab_universal.json",
        Path(getattr(__main__, "__file__", ".")) .parent / "masks_fab_universal.json",
    ]
    for p in candidates:
        if p and Path(p).exists():
            raw   = json.loads(Path(p).read_text(encoding="utf-8"))
            valid = [m for m in raw if "regex" in m and "mask_with" in m]
            log.info("Loaded %d universal masks from '%s'", len(valid), p)
            return valid
    log.warning("masks_fab_universal.json not found")
    return []


def validate_mask_regex(masks: list[dict]) -> list[dict]:
    good = []
    for m in masks:
        try:
            re.compile(m["regex"])
            good.append(m)
        except re.error as exc:
            log.warning("Dropped invalid mask '%s': %s", m.get("regex", "?"), exc)
    return good


def merge_masks(a: list[dict], b: list[dict]) -> list[dict]:
    seen: set[str] = set()
    merged = []
    for m in (*a, *b):
        key = m.get("regex", "")
        if key and key not in seen:
            seen.add(key)
            merged.append(m)
    return merged


# ---------------------------------------------------------------------------
# Parse-rate probing helpers
# ---------------------------------------------------------------------------

def probe_parse_rate(logs: list[str], masks: list[dict]) -> float:
    if not logs:
        return 0.0
    patterns = [re.compile(m["regex"]) for m in masks
                if _safe_compile(m["regex"])]
    return sum(
        1 for line in logs if any(p.search(line) for p in patterns)
    ) / len(logs)


def get_unparsed_lines(logs: list[str], masks: list[dict]) -> list[str]:
    patterns = [re.compile(m["regex"]) for m in masks
                if _safe_compile(m["regex"])]
    return [l for l in logs
            if not any(p.search(l) for p in patterns)][:30]


def _safe_compile(pattern: str) -> re.Pattern | None:
    try:
        return re.compile(pattern)
    except re.error:
        return None


# ---------------------------------------------------------------------------
# LLM synthesis call
# ---------------------------------------------------------------------------

def call_llm_for_masks(
    logs: list[str],
    llm_provider: str,
    max_length: int,
    cache_path: Path,
    telemetry: Telemetry,
    round_hint: str = "",
) -> list[dict]:
    from llm.registry import init_llm
    from DeepParse.deepparse.synth.hf_deepseek_r1 import synthesize_online

    t0 = time.monotonic()
    try:
        llm       = init_llm(provider=llm_provider)
        raw_masks = synthesize_online(
            logs=logs, llm=llm, max_length=max_length,
            self_consistency_attempts=3, round_hint=round_hint,
        )
        latency = time.monotonic() - t0
        telemetry.record(
            "llm_call",
            provider=llm_provider,
            latency_s=round(latency, 3),
            tokens_out=len(raw_masks) * 10,
            cost_usd=getattr(llm, "_total_cost_usd", 0.0),
        )
        result = [m.to_dict() if hasattr(m, "to_dict") else m
                  for m in raw_masks]
        return validate_mask_regex(result)
    except Exception as exc:
        log.error("LLM synthesis failed: %s", exc)
        telemetry.record("llm_error", error=str(exc))
        return []


# ---------------------------------------------------------------------------
# Adaptive synthesis with feedback loop
# ---------------------------------------------------------------------------

def synthesize_masks_adaptive(
    logs: list[str],
    llm_provider: str,
    max_length: int,
    mask_cache_path: Path,
    use_cache: bool,
    static_masks_path: Path | None,
    telemetry: Telemetry,
    max_rounds: int  = ADAPTIVE_MAX_RETRIES,
    threshold: float = ADAPTIVE_PARSE_THRESHOLD,
) -> list[dict]:
    """
    Three-stage mask pipeline:
      1. Load curated static masks (masks_fab_universal.json)
      2. Load or synthesise LLM masks
      3. Adaptive feedback: re-synthesise if parse-rate < threshold
    """
    # Stage 1 — static masks
    static_masks = load_universal_masks(static_masks_path)

    # Stage 2 — LLM masks (from cache or fresh synthesis)
    llm_masks: list[dict] = []
    if use_cache:
        cached = load_masks_from_cache(mask_cache_path)
        if cached is not None:
            log.info("Mask cache hit (%d masks)", len(cached))
            llm_masks = cached

    if not llm_masks:
        log.info("Synthesising masks via LLM (provider=%s)…", llm_provider)
        llm_masks = call_llm_for_masks(
            logs, llm_provider, max_length, mask_cache_path, telemetry
        )

    combined = merge_masks(static_masks, llm_masks)

    # Stage 3 — adaptive feedback loop
    for round_n in range(1, max_rounds + 1):
        rate = probe_parse_rate(logs[:50], combined)
        log.info("Round %d parse-rate probe: %.1f%%", round_n, rate * 100)
        TELEMETRY.record("parse_rate_probe", round=round_n, rate=round(rate, 4))

        if rate >= threshold:
            log.info("Parse-rate %.1f%% ≥ threshold — accepted", rate * 100)
            break

        log.warning("%.1f%% < %.0f%% — re-synthesising (round %d/%d)",
                    rate * 100, threshold * 100, round_n, max_rounds)
        unparsed = get_unparsed_lines(logs[:50], combined)
        hint     = (f"Round {round_n}: {len(unparsed)} unparsed lines. "
                    "Cover all variable token families.")
        new_masks = call_llm_for_masks(
            unparsed, llm_provider, max_length,
            mask_cache_path.with_suffix(f".round{round_n}.json"),
            telemetry=telemetry, round_hint=hint,
        )
        combined = merge_masks(combined, new_masks)

    save_masks(combined, mask_cache_path)
    return combined