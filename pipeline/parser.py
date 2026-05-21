"""
pipeline/parser.py — Drain wrapping, variable extraction, parse_to_records
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from DeepParse.deepparse import Drain
from pipeline.telemetry import TELEMETRY

log = logging.getLogger("deepparse.parser")

_MASK_TOKEN_RE = re.compile(r"<[A-Z_*][A-Z0-9_*]*>")


# ---------------------------------------------------------------------------
# Parsed flag
# ---------------------------------------------------------------------------

def is_parsed(raw: str, template: str) -> int:
    """
    1 if template contains at least one mask token AND differs from raw.
    Handles our pre-masking architecture where variables become named tokens
    like <MCH_ID> rather than generic Drain <*>.
    """
    return 1 if (_MASK_TOKEN_RE.search(template) and template != raw) else 0


# ---------------------------------------------------------------------------
# Variable extraction
# ---------------------------------------------------------------------------

def extract_variables(template: str, raw: str) -> dict[str, str]:
    """
    Reverse mask substitution to recover variable values.
    Returns {TOKEN_NAME: raw_value} e.g. {"MCH_ID": "MCH0001"}.
    """
    tokens = _MASK_TOKEN_RE.findall(template)
    if not tokens:
        return {}

    parts   = _MASK_TOKEN_RE.split(template)
    pattern = ""
    for i, part in enumerate(parts):
        pattern += re.escape(part)
        if i < len(tokens):
            pattern += "(.+?)"
    pattern += "$"

    try:
        m = re.match(pattern, raw)
        if m:
            return {tok.strip("<>"): val
                    for tok, val in zip(tokens, m.groups())}
    except re.error:
        pass
    return {f"var_{i}": tok for i, tok in enumerate(tokens)}


# ---------------------------------------------------------------------------
# Teach-back (Loop 2 — doc section 1.1)
# ---------------------------------------------------------------------------

def teach_back_to_drain(
    drain: Drain,
    records: list[dict],
    masks: list[dict],       # kept for API compat; unused internally
) -> None:
    """
    Loop 2 — LLM-to-Drain teach-back.
    Force-parse every resolved template through Drain so it lands in the
    prefix-tree cache and hits the fast-path on the next run.
    """
    seen: set[str] = set()
    taught = 0
    for r in records:
        if r["parsed"] != 1:
            continue
        tmpl = r["template"]
        if tmpl in seen:
            continue
        seen.add(tmpl)
        _ = drain.parse(tmpl)
        taught += 1
    if taught:
        log.info("Loop 2 teach-back: %d templates taught to Drain", taught)
        TELEMETRY.record("teach_back", templates_taught=taught)


# ---------------------------------------------------------------------------
# Batch parse
# ---------------------------------------------------------------------------

def parse_to_records(
    logs: list[str],
    drain: Drain,
    run_id: str,
) -> list[dict]:
    """Parse all lines, extract named variables, return record dicts."""
    templates = drain.parse_all(logs)
    now       = datetime.now(timezone.utc).isoformat()
    records   = []
    for line, tmpl in zip(logs, templates):
        parsed = is_parsed(line, tmpl)
        vars_  = extract_variables(tmpl, line) if parsed else {}
        records.append({
            "run_id":        run_id,
            "timestamp_utc": now,
            "raw":           line,
            "template":      tmpl,
            "variables":     json.dumps(vars_),
            "parsed":        parsed,
            "var_count":     len(vars_),
        })
    return records


# ---------------------------------------------------------------------------
# Schema validation guardrail
# ---------------------------------------------------------------------------

def validate_records(
    records: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Validate every record against OUTPUT_SCHEMA.
    Returns (valid, invalid). Invalid records go to quarantine, not dropped.
    """
    from pipeline.config import OUTPUT_SCHEMA
    try:
        import jsonschema
        validate = jsonschema.validate
    except ImportError:
        log.warning("jsonschema not installed — skipping validation")
        return records, []

    valid, invalid = [], []
    for r in records:
        probe = dict(r)
        try:
            probe["variables"] = (json.loads(r["variables"])
                                  if isinstance(r["variables"], str)
                                  else r["variables"])
            validate(instance=probe, schema=OUTPUT_SCHEMA)
            valid.append(r)
        except Exception as exc:
            log.debug("Validation failed: %s | %.60s", exc, r.get("raw", ""))
            invalid.append({**r, "_validation_error": str(exc)})

    if invalid:
        log.warning("Schema validation: %d/%d failed", len(invalid), len(records))
        TELEMETRY.record("validation", total=len(records),
                         valid=len(valid), invalid=len(invalid))
    return valid, invalid


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def health_check(records: list[dict], run_meta: dict) -> None:
    assert records,                           "FAIL: zero output records"
    assert run_meta["parsed_count"] >= 0,     "FAIL: negative parsed_count"
    assert 0.0 <= run_meta["parse_rate"] <= 1.0, "FAIL: parse_rate out of range"
    templates = {r["template"] for r in records}
    assert templates, "FAIL: no templates extracted"
    log.info(
        "Health OK — %d records, %d templates, parse_rate=%.1f%%",
        len(records), len(templates), run_meta["parse_rate"] * 100,
    )
    TELEMETRY.record("health_check", records=len(records),
                     templates=len(templates), parse_rate=run_meta["parse_rate"])