"""
pipeline/config.py — Constants, thresholds and output schema
"""
from __future__ import annotations
import textwrap

# ---------------------------------------------------------------------------
# Adaptive parsing thresholds
# ---------------------------------------------------------------------------
ADAPTIVE_PARSE_THRESHOLD = 0.70   # re-synthesise masks below this parse-rate
ADAPTIVE_MAX_RETRIES     = 3      # max feedback rounds before graceful fallback

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
INJECTION_MAX_LEN = 512
INJECTION_BLOCK_PATTERNS = [
    r"ignore (?:all |previous |above )?instructions",
    r"you are now",
    r"DAN mode",
    r"system prompt",
    r"<\|.*?\|>",
    r"\\n---\\n",
]

# ---------------------------------------------------------------------------
# JSON-Schema for every output record
# ---------------------------------------------------------------------------
OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema",
    "type": "object",
    "required": ["run_id", "timestamp_utc", "raw", "template",
                 "variables", "parsed", "var_count"],
    "properties": {
        "run_id":        {"type": "string"},
        "timestamp_utc": {"type": "string"},
        "raw":           {"type": "string"},
        "template":      {"type": "string"},
        "variables":     {"type": ["object", "string"]},
        "parsed":        {"type": "integer", "enum": [0, 1]},
        "var_count":     {"type": "integer", "minimum": 0},
    },
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# LLM grounding context (RAG-lite domain schema)
# ---------------------------------------------------------------------------
FAB_DOMAIN_CONTEXT = textwrap.dedent("""
    You are a silicon-fab equipment-data parsing specialist.
    The logs originate from these process types:
      Lithography (ArF 193nm), Dry Etch (CCP/ICP), LPCVD, Thermal Oxidation,
      Ion Implantation, PVD Sputtering, CMP, Wet Clean, Optical Metrology, RTP Anneal.

    Common variable token families (generate regex masks for these):
      - Entity IDs : CJOB_*, PRJOB_*, EQP_*, RCP_*, LOT_*, WFR_*, MOD_*, SENSOR_*,
                     SLOT_*, NET_*, VER_*, NAME_*, *_EC_*, CREATOR_*, VENDOR_*
      - Numeric    : floats, scientific notation (1.5e-7), large integers (SVID refs)
      - Timestamps : ISO-8601 (2026-02-18T08:00:00Z), date-only, syslog sub-second
      - Syslog     : Machine:MCH*, ER-*, DW-*, RH-*, KU-*, IVR position tokens
      - XML nodes  : <SetPoint><*></SetPoint>, <Value><*></Value>
      - Units      : quantities combined with nm/°C/sccm/mTorr/keV/W/rpm

    Output masks ordered by specificity (most specific first).
    Each mask MUST be a valid Python regex string.
""").strip()

# ---------------------------------------------------------------------------
# Per-source Drain similarity thresholds (doc section 1.4)
# ---------------------------------------------------------------------------
SOURCE_SIM_THRESHOLDS: dict[str, float] = {
    ".yaml": 0.25, ".yml": 0.25,
    ".json": 0.50, ".jsonl": 0.50,
    ".xml":  0.35,
    ".csv":  0.30, ".tsv": 0.30,
    ".ini":  0.20, ".cfg": 0.20,
    ".log":  0.40, ".txt": 0.40,
}