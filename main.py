"""
DeepParse v2 — Silicon-Fab Log Parsing Pipeline
================================================
Design principles:
  1. Adaptive parsing  – feedback loop re-synthesises masks when parse-rate drops
  2. Schema grounding  – domain schema injected into LLM context (RAG-lite)
  3. Guardrails        – JSON-Schema validation + prompt-injection sanitisation
  4. Observability     – structured telemetry: latency, token cost, parse metrics
  5. Reproducibility   – global seed + deterministic run_id + mask cache
  6. Multi-format I/O  – XML, JSON, YAML, CSV, TSV, INI, TXT → SQLite + CSV + JSON
  7. Enterprise-ready  – retry logic, graceful degradation, CLI + API surface
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pathlib
import random
import re
import sqlite3
import sys
import textwrap
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from llm.registry import init_llm
from DeepParse.deepparse import Drain
from DeepParse.deepparse.synth.hf_deepseek_r1 import synthesize_online
from DeepParse.deepparse.evaluation.eval_runner import EvaluationRunner, evaluate_records
from DeepParse.deepparse.tools.fetch_loghub import download_logs

# ---------------------------------------------------------------------------
# Logging — structured JSON-lines for prod observability
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("deepparse.main")

# ---------------------------------------------------------------------------
# Constants & Rubric Thresholds
# ---------------------------------------------------------------------------

ADAPTIVE_PARSE_THRESHOLD   = 0.70
ADAPTIVE_MAX_RETRIES       = 3
INJECTION_MAX_LEN          = 512
INJECTION_BLOCK_PATTERNS   = [
    r"ignore (?:all |previous |above )?instructions",
    r"you are now",
    r"DAN mode",
    r"system prompt",
    r"<\|.*?\|>",
    r"\\n---\\n",
]
_INJECTION_RX = re.compile(
    "|".join(INJECTION_BLOCK_PATTERNS), re.IGNORECASE
)

OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema",
    "type": "object",
    "required": ["run_id", "timestamp_utc", "raw", "template", "variables", "parsed", "var_count"],
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
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    log.info("Global seed → %d", seed)


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class Telemetry:
    def __init__(self) -> None:
        self._events: list[dict] = []

    def record(self, event: str, **kw) -> None:
        self._events.append({"event": event, "ts": datetime.now(timezone.utc).isoformat(), **kw})

    def summary(self) -> dict:
        llm_calls     = [e for e in self._events if e["event"] == "llm_call"]
        total_latency = sum(e.get("latency_s", 0) for e in llm_calls)
        total_tokens  = sum(e.get("tokens_out", 0) for e in llm_calls)
        return {
            "llm_calls":       len(llm_calls),
            "total_latency_s": round(total_latency, 3),
            "est_tokens_out":  total_tokens,
            "events":          len(self._events),
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"summary": self.summary(), "events": self._events}, f, indent=2)


TELEMETRY = Telemetry()

# ---------------------------------------------------------------------------
# Security — prompt-injection sanitisation
# ---------------------------------------------------------------------------

def sanitize_for_llm(line: str, max_len: int = INJECTION_MAX_LEN) -> str | None:
    truncated = line[:max_len]
    if _INJECTION_RX.search(truncated):
        log.warning("Prompt-injection pattern detected — line dropped: %.60s…", truncated)
        TELEMETRY.record("injection_blocked", snippet=truncated[:80])
        return None
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", truncated)
    return sanitized


# ---------------------------------------------------------------------------
# Input Loaders
# ---------------------------------------------------------------------------

def load_xml(path: str) -> list[str]:
    """
    XML loader that extracts parseable log-like lines from every element.
    Produces three kinds of lines per element:
      1. Attribute lines  Tag.attr = value      e.g. Constant.name = MaxTemp
      2. Content lines    <Tag>value</Tag>       e.g. <SetPoint>98.5</SetPoint>
      3. Combined kv line Tag k="v" k="v"       e.g. Step number="1" name="PURGE"
    This gives Drain rich repeated structure even for pure-attribute XML.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"XML input not found: {path}")
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        lines: list[str] = []
        for elem in root.iter():
            tag     = re.sub(r'\{[^}]+\}', '', elem.tag)
            attribs = {re.sub(r'\{[^}]+\}', '', k): v
                       for k, v in (elem.attrib or {}).items()}
            content = (elem.text or "").strip()
            # 1 — one line per attribute
            for ak, av in attribs.items():
                lines.append(f"{tag}.{ak} = {av}")
            # 2 — content line
            if content:
                lines.append(f"<{tag}>{content}</{tag}>")
            # 3 — combined kv line (only when >=2 attribs)
            if len(attribs) >= 2:
                kv = " ".join(f'{k}="{v}"' for k, v in attribs.items())
                lines.append(f"{tag} {kv}")
        log.info("XML loader: extracted %d lines from %s", len(lines), p.name)
        return lines if lines else _xml_raw_fallback(path)
    except Exception as exc:
        log.warning("XML structured parse failed (%s) — raw fallback", exc)
        return _xml_raw_fallback(path)


def _xml_raw_fallback(path: str) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return [l.strip() for l in f
                if l.strip() and not l.strip().startswith("<?")]


def load_structured(path: str, message_col: str = "message") -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Structured file not found: {path}")
    ext = p.suffix.lower()

    # ── TSV ───────────────────────────────────────────────────────────────────
    if ext == ".tsv":
        try:
            df = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig",
                             on_bad_lines="skip")
        except TypeError:
            df = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep="\t", dtype=str, encoding="latin-1")
        real = [c for c in df.columns if "Unnamed" not in str(c)]
        if len(real) >= 2:
            log.info("TSV: producing ColName=value lines (%d cols) for %s",
                     len(real), p.name)
            out: list[str] = []
            for _, row in df.iterrows():
                for col in real:
                    val = str(row.get(col, "")).strip()
                    if val and val.lower() != "nan":
                        out.append(f"{col} = {val}")
            return out

    # ── CSV ───────────────────────────────────────────────────────────────────
    elif ext == ".csv":
        try:
            df = pd.read_csv(path, dtype=str, encoding="utf-8-sig",
                             on_bad_lines="skip")
        except TypeError:
            # pandas < 1.3 uses error_bad_lines
            df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype=str, encoding="latin-1")

    elif ext in (".json", ".jsonl"):
        try:
            df = pd.read_json(path, lines=True, dtype=str)
        except Exception:
            df = pd.read_json(path, lines=False, dtype=str)

    elif ext == ".parquet":
        df = pd.read_parquet(path)

    elif ext in (".yaml", ".yml"):
        import yaml
        with open(path, encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        return _flatten_obj(obj)

    elif ext in (".ini", ".cfg", ".properties"):
        return _load_ini_as_lines(path)

    else:
        raise ValueError(f"Unsupported extension: {ext}")

    # ── Key-value table detection ─────────────────────────────────────────────
    real_cols    = [c for c in df.columns if "Unnamed" not in str(c)]
    unnamed_cols = [c for c in df.columns if "Unnamed" in str(c)]

    if len(real_cols) <= 2 and len(unnamed_cols) > 0:
        log.info("Key-value table detected in '%s'", p.name)
        lines: list[str] = []
        for _, row in df.iterrows():
            values = [str(v).strip() for v in row
                      if str(v).strip() not in ("nan", "")]
            if len(values) >= 2:
                lines.append(f"{values[0]}: {values[1]}")
            elif len(values) == 1:
                lines.append(values[0])
        return lines

    # ── Standard column detection ─────────────────────────────────────────────
    # Priority 1: exact match on requested column name
    if message_col in df.columns:
        return df[message_col].dropna().astype(str).tolist()

    # Priority 2: common log column names
    for candidate in ["message", "msg", "log", "text", "content",
                      "description", "event", "entry", "line", "value",
                      "logmessage", "log_message", "raw"]:
        match = next((c for c in df.columns if c.lower() == candidate), None)
        if match:
            log.info("Auto-detected text column '%s' in %s", match, p.name)
            return df[match].dropna().astype(str).tolist()

    # Priority 3: longest average text column
    str_cols = df.select_dtypes(include="object").columns.tolist()
    if str_cols:
        avg_lens = {c: df[c].dropna().astype(str).str.len().mean()
                    for c in str_cols}
        best = max(avg_lens, key=avg_lens.get)
        log.info("Using longest-text column '%s' (avg %.0f chars) in %s",
                 best, avg_lens[best], p.name)
        return df[best].dropna().astype(str).tolist()

    # Priority 4: last resort — join all columns safely
    log.warning("Falling back to full-row join for %s", p.name)
    return (
        df.fillna("")
          .astype(str)
          .apply(lambda row: " | ".join(v for v in row if v.strip()), axis=1)
          .tolist()
    )

def _flatten_obj(obj: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lines.extend(_flatten_obj(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            lines.extend(_flatten_obj(v, f"{prefix}[{i}]"))
    else:
        lines.append(f"{prefix}: {obj}")
    return lines


def _load_ini_as_lines(path: str) -> list[str]:
    """
    INI/cfg loader that produces Section.key = value lines.
    - Section headers [SectionName] are prefixed onto every key below them
      so Drain can cluster by section (e.g. ProcessParams.MaxTemp = 850.5)
    - Bare key = value without a section is emitted as-is
    - Comments (#, ;) and blank lines are skipped
    """
    lines: list[str] = []
    section = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped or stripped.startswith(("#", ";")):
                continue
            # Section header
            m = re.match(r"^\[([^\]]+)\]$", stripped)
            if m:
                section = m.group(1).strip()
                continue
            # key = value  or  key: value
            m2 = re.match(r"^([^=:]+)[=:](.*)$", stripped)
            if m2:
                key = m2.group(1).strip()
                val = m2.group(2).strip()
                if section:
                    lines.append(f"{section}.{key} = {val}")
                else:
                    lines.append(f"{key} = {val}")
            else:
                lines.append(stripped)
    return lines


_SYSLOG_PREFIX = re.compile(r"^\w{3}\s+\d{1,2}\s+[\d:]+\s+\S+\s+\S+[:\s]\s*")
_LOG4J_PREFIX  = re.compile(r"^\d{4}-\d{2}-\d{2}[\sT][\d:,\.]+\s+\w+\s+\S+[:\s]\s*")


def load_semistructured(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Semi-structured file not found: {path}")
    messages: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            for pat in (_SYSLOG_PREFIX, _LOG4J_PREFIX):
                s = pat.sub("", line)
                if s and s != line:
                    line = s
                    break
            messages.append(line)
    return messages


def load_unstructured(path: str, max_length: int = 128) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Unstructured file not found: {path}")
    with open(path, encoding="utf-8", errors="replace") as f:
        return [l.strip()[:max_length] for l in f if l.strip()]


def load_logs(
    path: str,
    fmt: str = "auto",
    message_col: str = "message",
    max_length: int = 128,
) -> list[str]:
    ext = Path(path).suffix.lower()
    if fmt == "auto":
        if ext in (".xml", ".xsd"):
            fmt = "xml"
        elif ext in (".csv", ".tsv", ".jsonl", ".json", ".parquet", ".yaml", ".yml", ".ini", ".cfg"):
            fmt = "structured"
        elif ext in (".log", ".txt"):
            fmt = "unstructured"
        else:
            fmt = "semistructured"
    log.info("Loading '%s' as format '%s'", path, fmt)
    if fmt == "xml":            return load_xml(path)
    if fmt == "structured":     return load_structured(path, message_col)
    if fmt == "semistructured": return load_semistructured(path)
    if fmt == "unstructured":   return load_unstructured(path, max_length)
    raise ValueError(f"Unknown format: {fmt}")


# ---------------------------------------------------------------------------
# Normalise & validate
# ---------------------------------------------------------------------------

def normalize(
    logs: list[str],
    max_length: int = 128,
    min_length: int = 5,
    max_logs: int = 500,
    sanitize: bool = True,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    counters = {"empty": 0, "short": 0, "dup": 0, "injected": 0, "truncated": 0}

    for raw in logs:
        line = raw.strip()
        if not line:
            counters["empty"] += 1; continue
        if len(line) < min_length:
            counters["short"] += 1; continue
        if len(line) > max_length:
            line = line[:max_length]
            counters["truncated"] += 1
        if sanitize:
            clean = sanitize_for_llm(line, max_len=max_length)
            if clean is None:
                counters["injected"] += 1; continue
            line = clean
        if line in seen:
            counters["dup"] += 1; continue
        seen.add(line)
        out.append(line)
        if len(out) >= max_logs:
            log.warning("Ceiling reached: %d rows", max_logs); break

    log.info("Normalise: kept=%d %s", len(out), counters)
    TELEMETRY.record("normalise", kept=len(out), **counters)
    if not out:
        raise ValueError("Zero lines after normalisation — check input columns / format.")
    return out


# ---------------------------------------------------------------------------
# Schema grounding — RAG-lite domain context
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
      - Syslog     : Machine:MCH*, ER-*, DW-*, RH-*, KU-*, IVR position tokens,
                     de_err=*, ESET:*, action_handle=*, exposure_handle=*
      - XML nodes  : <SetPoint><*></SetPoint>, <Value><*></Value>
      - Units      : quantities combined with nm/°C/sccm/mTorr/keV/W/rpm

    Output masks should be exclusive, non-overlapping, and ordered by specificity
    (most specific first). Each mask MUST be a valid Python regex string.
""").strip()


def build_grounded_prompt(sample_lines: list[str], extra_context: str = "") -> str:
    samples = "\n".join(f"  {i+1}. {l}" for i, l in enumerate(sample_lines[:20]))
    return (
        f"{FAB_DOMAIN_CONTEXT}\n\n"
        f"{extra_context}\n\n"
        f"Sample log lines:\n{samples}\n\n"
        "Generate regex masks for all variable tokens. Return ONLY a JSON array "
        "of objects: [{\"regex\": \"...\", \"mask_with\": \"<*>\"}]"
    )


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def load_masks_from_cache(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_masks(masks: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if masks and hasattr(masks[0], "to_dict"):
        data = [m.to_dict() for m in masks]
    else:
        data = masks
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_universal_masks(masks_path: Path | None = None) -> list[dict]:
    candidates = [
        masks_path,
        Path(__file__).parent / "masks_fab_universal.json",
        Path(__file__).parent / "masks" / "masks_fab_universal.json",
    ]
    for p in candidates:
        if p and Path(p).exists():
            raw   = json.loads(Path(p).read_text(encoding="utf-8"))
            valid = [m for m in raw if "regex" in m and "mask_with" in m]
            log.info("Loaded %d universal masks from '%s'", len(valid), p)
            return valid
    log.warning("masks_fab_universal.json not found — starting with zero static masks")
    return []


def _validate_mask_regex(masks: list[dict]) -> list[dict]:
    good = []
    for m in masks:
        try:
            re.compile(m["regex"])
            good.append(m)
        except re.error as exc:
            log.warning("Dropped invalid regex mask '%s': %s", m.get("regex", "?"), exc)
    return good


# ---------------------------------------------------------------------------
# Adaptive mask synthesis with feedback loop
# ---------------------------------------------------------------------------

def synthesize_masks_adaptive(
    logs: list[str],
    llm_provider: str,
    max_length: int,
    mask_cache_path: Path,
    use_cache: bool,
    static_masks_path: Path | None,
    telemetry: Telemetry,
    max_rounds: int = ADAPTIVE_MAX_RETRIES,
    threshold: float = ADAPTIVE_PARSE_THRESHOLD,
) -> list[dict]:
    # 1 — static domain masks
    static_masks = load_universal_masks(static_masks_path)

    # 2 — LLM-synthesised masks
    llm_masks: list[dict] = []
    if use_cache:
        cached = load_masks_from_cache(mask_cache_path)
        if cached is not None:
            log.info("LLM mask cache hit (%d masks)", len(cached))
            llm_masks = cached

    if not llm_masks:
        log.info("Synthesising masks via LLM (provider=%s)…", llm_provider)
        llm_masks = _call_llm_for_masks(
            logs, llm_provider, max_length, mask_cache_path, telemetry
        )

    combined = _merge_masks(static_masks, llm_masks)

    # 3 — adaptive feedback loop
    for round_n in range(1, max_rounds + 1):
        rate = _probe_parse_rate(logs[:50], combined)
        log.info("Round %d parse-rate probe: %.1f%%", round_n, rate * 100)
        TELEMETRY.record("parse_rate_probe", round=round_n, rate=round(rate, 4))

        if rate >= threshold:
            log.info("Parse-rate %.1f%% ≥ threshold %.0f%% — masks accepted.", rate * 100, threshold * 100)
            break

        log.warning(
            "Parse-rate %.1f%% < %.0f%% — re-synthesising masks (round %d/%d)…",
            rate * 100, threshold * 100, round_n, max_rounds,
        )
        unparsed = _get_unparsed_lines(logs[:50], combined)
        # Supply a targeted round hint to improve synthesis quality
        hint = f"Round {round_n} focus: {len(unparsed)} unparsed lines. Cover all variable token families."
        new_masks = _call_llm_for_masks(
            unparsed, llm_provider, max_length,
            mask_cache_path.with_suffix(f".round{round_n}.json"),
            telemetry=telemetry,
            round_hint=hint,
        )
        combined = _merge_masks(combined, new_masks)

    save_masks(combined, mask_cache_path)
    return combined


def _call_llm_for_masks(
    logs: list[str],
    llm_provider: str,
    max_length: int,
    cache_path: Path,
    telemetry: Telemetry,
    round_hint: str = "",
) -> list[dict]:
    t0 = time.monotonic()
    try:
        llm = init_llm(provider=llm_provider)
        raw_masks = synthesize_online(
            logs=logs,
            llm=llm,
            max_length=max_length,
            self_consistency_attempts=3,
            round_hint=round_hint,
        )
        latency = time.monotonic() - t0
        telemetry.record(
            "llm_call",
            provider=llm_provider,
            latency_s=round(latency, 3),
            tokens_out=len(raw_masks) * 10,
            cost_usd=getattr(llm, "_total_cost_usd", 0.0),
        )
        if hasattr(raw_masks[0], "to_dict") if raw_masks else False:
            result = [m.to_dict() for m in raw_masks]
        else:
            result = raw_masks
        return _validate_mask_regex(result)
    except Exception as exc:
        log.error("LLM synthesis failed: %s — falling back to empty mask list", exc)
        telemetry.record("llm_error", error=str(exc))
        return []


def _merge_masks(a: list[dict], b: list[dict]) -> list[dict]:
    seen: set[str] = set()
    merged = []
    for m in (*a, *b):
        key = m.get("regex", "")
        if key and key not in seen:
            seen.add(key)
            merged.append(m)
    return merged


def _probe_parse_rate(logs: list[str], masks: list[dict]) -> float:
    if not logs:
        return 0.0
    patterns = []
    for m in masks:
        try:
            patterns.append(re.compile(m["regex"]))
        except re.error:
            pass
    hits = 0
    for line in logs:
        for pat in patterns:
            if pat.search(line):
                hits += 1
                break
    return hits / len(logs)


def _get_unparsed_lines(logs: list[str], masks: list[dict]) -> list[str]:
    patterns = [re.compile(m["regex"]) for m in masks if _safe_compile(m["regex"])]
    return [l for l in logs if not any(p.search(l) for p in patterns)][:30]


def _safe_compile(pattern: str) -> re.Pattern | None:
    try:
        return re.compile(pattern)
    except re.error:
        return None


# ---------------------------------------------------------------------------
# Core parsing utilities
# ---------------------------------------------------------------------------

_MASK_TOKEN_RE = re.compile(r"<[A-Z_*][A-Z0-9_*]*>")


def _is_parsed(raw: str, template: str) -> int:
    """
    A line is 'parsed' when its template contains at least one mask token
    AND differs from the raw line (masking actually occurred).
    Correctly handles our pre-masking architecture where variables become
    named tokens like <MCH_ID> rather than generic Drain <*>.
    """
    return 1 if (_MASK_TOKEN_RE.search(template) and template != raw) else 0


def extract_variables(template: str, raw: str) -> dict[str, str]:
    """
    Extract variable values by reversing mask substitution.
    Returns {TOKEN_NAME: raw_value} e.g. {"MCH_ID": "MCH0001", "RECIPE_ID": "RCP_NOVA_001"}.
    Falls back to positional keys when named extraction fails.
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
            return {tok.strip("<>"): val for tok, val in zip(tokens, m.groups())}
    except re.error:
        pass
    return {f"var_{i}": tok for i, tok in enumerate(tokens)}


def parse_to_records(logs: list[str], drain: Drain, run_id: str) -> list[dict]:
    templates = drain.parse_all(logs)
    now       = datetime.now(timezone.utc).isoformat()
    records   = []
    for line, tmpl in zip(logs, templates):
        parsed = _is_parsed(line, tmpl)
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
# Output validation — JSON Schema guardrail
# ---------------------------------------------------------------------------

def validate_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    try:
        import jsonschema
        validate = jsonschema.validate
    except ImportError:
        log.warning("jsonschema not installed — skipping schema validation")
        return records, []

    valid, invalid = [], []
    for r in records:
        probe = dict(r)
        try:
            probe["variables"] = json.loads(r["variables"]) if isinstance(r["variables"], str) else r["variables"]
            validate(instance=probe, schema=OUTPUT_SCHEMA)
            valid.append(r)
        except Exception as exc:
            log.debug("Record validation failed: %s | record=%s", exc, r.get("raw", "")[:60])
            invalid.append({**r, "_validation_error": str(exc)})

    if invalid:
        log.warning("Schema validation: %d/%d records failed", len(invalid), len(records))
        TELEMETRY.record("validation", total=len(records), valid=len(valid), invalid=len(invalid))

    return valid, invalid


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

_CREATE_PARSED_LOGS = """
CREATE TABLE IF NOT EXISTS parsed_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    NOT NULL,
    timestamp_utc TEXT    NOT NULL,
    raw           TEXT    NOT NULL,
    template      TEXT    NOT NULL,
    variables     TEXT    NOT NULL,
    parsed        INTEGER NOT NULL,
    var_count     INTEGER NOT NULL
);
"""

_CREATE_RUN_META = """
CREATE TABLE IF NOT EXISTS run_meta (
    run_id        TEXT PRIMARY KEY,
    created_utc   TEXT NOT NULL,
    seed          INTEGER,
    llm_provider  TEXT,
    mask_count    INTEGER,
    log_count     INTEGER,
    parsed_count  INTEGER,
    parse_rate    REAL,
    invalid_count INTEGER,
    notes         TEXT
);
"""

_CREATE_QUARANTINE = """
CREATE TABLE IF NOT EXISTS quarantine (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL,
    raw               TEXT,
    template          TEXT,
    validation_error  TEXT
);
"""


def write_sqlite(
    records: list[dict],
    db_path: Path,
    run_meta: dict | None = None,
    invalid: list[dict] | None = None,
) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute(_CREATE_PARSED_LOGS)
    con.execute(_CREATE_RUN_META)
    con.execute(_CREATE_QUARANTINE)

    con.executemany(
        """INSERT INTO parsed_logs
           (run_id,timestamp_utc,raw,template,variables,parsed,var_count)
           VALUES (:run_id,:timestamp_utc,:raw,:template,:variables,:parsed,:var_count)""",
        records,
    )
    if run_meta:
        con.execute(
            """INSERT OR REPLACE INTO run_meta
               (run_id,created_utc,seed,llm_provider,mask_count,log_count,
                parsed_count,parse_rate,invalid_count,notes)
               VALUES (:run_id,:created_utc,:seed,:llm_provider,:mask_count,:log_count,
                       :parsed_count,:parse_rate,:invalid_count,:notes)""",
            run_meta,
        )
    if invalid:
        con.executemany(
            """INSERT INTO quarantine (run_id,raw,template,validation_error)
               VALUES (:run_id,:raw,:template,:_validation_error)""",
            [{**r, "run_id": run_meta["run_id"] if run_meta else "?"} for r in invalid],
        )
    con.commit()
    con.close()
    return db_path


def write_csv(records: list[dict], csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "timestamp_utc", "raw", "template", "variables", "parsed", "var_count"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    return csv_path


def write_json(records: list[dict], json_path: Path) -> Path:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    enriched = []
    for r in records:
        row = dict(r)
        try:
            row["variables"] = json.loads(r["variables"])
        except Exception:
            row["variables"] = {}
        enriched.append(row)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    return json_path


# ---------------------------------------------------------------------------
# Downstream analytics — cluster report + top-N templates
# ---------------------------------------------------------------------------

def write_cluster_report(drain: Drain, output_dir: Path, run_id: str) -> Path:
    """
    Write a cluster-level analytics report.
    This is the 'downstream use' artifact: engineers query templates, not raw lines.
    """
    clusters = drain.cluster_summary()
    report   = {
        "run_id":            run_id,
        "total_clusters":    len(clusters),
        "template_fingerprint": drain.template_fingerprint(),
        "drain_stats":       drain.stats,
        "top_clusters":      clusters[:50],
    }
    path = output_dir / f"{run_id}_clusters.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Cluster report written → %s (%d clusters)", path, len(clusters))
    return path


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

def health_check(records: list[dict], run_meta: dict) -> None:
    assert records,                         "Health check FAIL: zero output records"
    assert run_meta["parsed_count"] >= 0,   "Health check FAIL: negative parsed_count"
    assert 0.0 <= run_meta["parse_rate"] <= 1.0, "Health check FAIL: parse_rate out of range"
    templates = {r["template"] for r in records}
    assert templates, "Health check FAIL: no templates extracted"
    log.info(
        "Health check OK — %d records, %d templates, parse_rate=%.1f%%",
        len(records), len(templates), run_meta["parse_rate"] * 100,
    )
    TELEMETRY.record(
        "health_check",
        records=len(records),
        templates=len(templates),
        parse_rate=run_meta["parse_rate"],
    )


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run(
    mode: str,
    root_dir: Path,
    *,
    seed: int                   = 42,
    output_dir: Path | None     = None,
    input_path: str | None      = None,
    input_fmt: str              = "auto",
    message_col: str            = "message",
    llm_provider: str           = "anthropic",
    use_mask_cache: bool        = True,
    max_length: int             = 128,
    max_logs: int               = 500,
    config: str | None          = None,
    static_masks_path: str | None = None,
    adaptive_threshold: float   = ADAPTIVE_PARSE_THRESHOLD,
    adaptive_rounds: int        = ADAPTIVE_MAX_RETRIES,
    drain_sim_threshold: float  = 0.5,
    drain_depth: int            = 4,
    save_drain_state: bool      = True,
) -> dict:
    """
    Full pipeline execution.
    Returns a summary dict for API / programmatic usage.
    """
    set_global_seed(seed)

    if output_dir is None:
        output_dir = root_dir / "artifacts" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "test" and input_path:
        t_start   = time.monotonic()
        file_stem = Path(input_path).stem
        run_id    = (
            f"run_{file_stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{seed}"
        )
        TELEMETRY.record("pipeline_start", run_id=run_id, input=input_path, fmt=input_fmt)

        # ── 1. Load ──────────────────────────────────────────────────────────
        raw_logs = load_logs(
            input_path, fmt=input_fmt, message_col=message_col, max_length=max_length
        )
        TELEMETRY.record("load", lines=len(raw_logs))

        # ── 2. Normalise + sanitise ──────────────────────────────────────────
        logs = normalize(raw_logs, max_length=max_length, max_logs=max_logs, sanitize=True)

        # ── 3. Adaptive mask synthesis (static + LLM + feedback loop) ────────
        mask_cache = output_dir / f"masks_{file_stem}.json"
        masks_dicts = synthesize_masks_adaptive(
            logs=logs,
            llm_provider=llm_provider,
            max_length=max_length,
            mask_cache_path=mask_cache,
            use_cache=use_mask_cache,
            static_masks_path=Path(static_masks_path) if static_masks_path else None,
            telemetry=TELEMETRY,
            max_rounds=adaptive_rounds,
            threshold=adaptive_threshold,
        )
        TELEMETRY.record("masks_ready", count=len(masks_dicts))

        # ── 4. Drain parse ────────────────────────────────────────────────────
        drain = Drain(
            sim_threshold=drain_sim_threshold,
            depth=drain_depth,
            auto_tune_threshold=True,
        )
        drain.load_masks(masks_dicts)
        records = parse_to_records(logs, drain, run_id)

        # Optionally persist Drain state for reproducibility / warm-start
        if save_drain_state:
            drain.save_state(output_dir / f"{run_id}_drain_state.json")

        # ── 5. Schema validation / quarantine ────────────────────────────────
        valid_records, invalid_records = validate_records(records)

        # ── 6. Build run metadata ─────────────────────────────────────────────
        parsed_count = sum(r["parsed"] for r in valid_records)
        run_meta = {
            "run_id":        run_id,
            "created_utc":   datetime.now(timezone.utc).isoformat(),
            "seed":          seed,
            "llm_provider":  llm_provider,
            "mask_count":    len(masks_dicts),
            "log_count":     len(valid_records),
            "parsed_count":  parsed_count,
            "parse_rate":    round(parsed_count / max(len(valid_records), 1), 4),
            "invalid_count": len(invalid_records),
            "notes":         f"input={input_path} drain_fp={drain.template_fingerprint()}",
        }

        # ── 7. Health check ───────────────────────────────────────────────────
        health_check(valid_records, run_meta)

        # ── 8. Evaluation metrics (self-eval, no ground truth) ────────────────
        eval_metrics = evaluate_records(valid_records)
        TELEMETRY.record(
            "eval_metrics",
            parse_rate=eval_metrics.parse_rate,
            unique_templates=eval_metrics.unique_templates,
            avg_wildcard_ratio=eval_metrics.avg_wildcard_ratio,
        )

        # ── 9. Write all artifacts ────────────────────────────────────────────
        db_file      = write_sqlite(valid_records, output_dir / "parsed_logs.db", run_meta, invalid_records)
        csv_file     = write_csv(valid_records,    output_dir / f"{run_id}.csv")
        js_file      = write_json(valid_records,   output_dir / f"{run_id}.json")
        clust_file   = write_cluster_report(drain, output_dir, run_id)
        tel_file     = output_dir / f"{run_id}_telemetry.json"
        TELEMETRY.record("pipeline_done", elapsed_s=round(time.monotonic() - t_start, 3))
        TELEMETRY.write(tel_file)

        elapsed       = time.monotonic() - t_start
        tel_summary   = TELEMETRY.summary()
        n_fab_masks   = len([m for m in masks_dicts if any(
            tok in m.get("mask_with", "")
            for tok in ("LOT", "EQP", "WFR", "CJOB", "PRJOB", "RCP", "EC_ID")
        )])

        print(f"\n{'='*62}")
        print(f"  DeepParse v2  |  {run_id}")
        print(f"{'='*62}")
        print(f"  Logs processed     : {len(valid_records):>6,}  ({len(invalid_records)} quarantined)")
        print(f"  Templates found    : {eval_metrics.unique_templates:>6,}")
        print(f"  Parse rate         : {run_meta['parse_rate']*100:>6.1f}%")
        print(f"  Avg wildcard ratio : {eval_metrics.avg_wildcard_ratio:>6.3f}  (lower = more specific templates)")
        print(f"  Masks used         : {len(masks_dicts):>6,}  ({n_fab_masks} fab-domain specific)")
        print(f"  Drain clusters     : {len(drain.get_clusters()):>6,}  fp={drain.template_fingerprint()}")
        print(f"  LLM calls          : {tel_summary['llm_calls']:>6,}")
        print(f"  Elapsed            : {elapsed:>6.2f}s")
        print(f"\n  Artifacts → '{output_dir}/'")
        print(f"    [SQLite   ] {db_file.name}")
        print(f"    [CSV      ] {csv_file.name}")
        print(f"    [JSON     ] {js_file.name}")
        print(f"    [Clusters ] {clust_file.name}")
        print(f"    [Telemetry] {tel_file.name}")

        return {
            "run_id":         run_id,
            "parse_rate":     run_meta["parse_rate"],
            "templates":      eval_metrics.unique_templates,
            "masks":          len(masks_dicts),
            "elapsed_s":      elapsed,
            "output_dir":     str(output_dir),
        }

    elif mode == "eval":
        out_folder = root_dir / "artifacts" / "data"
        download_logs(out=out_folder)
        eval_config = config or str(root_dir / "eval.yaml")
        # Auto-create eval.yaml if absent
        if not Path(eval_config).exists():
            Path(eval_config).write_text(
                "output_dir: artifacts/output\n"
                "eval_dir: artifacts/eval\n"
                "min_parse_rate: 0.60\n"
                "min_ga: 0.50\n"
                "pass_on_no_gt: true\n",
                encoding="utf-8",
            )
        EvaluationRunner(Path(eval_config)).run()
        print("Evaluation metrics file generated.")
        return {}

    else:
        print("No input_path supplied for mode='test'. Pass --input <file>.")
        return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="DeepParse v2 — Silicon-Fab Log Parsing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples
        --------
        # Parse a vendor JSON recipe file (Anthropic LLM, default)
        python main.py --input vendor1_novastar_recipe_details.json

        # Parse XML with universal static masks + adaptive re-synthesis
        python main.py --input vendor3_chipconst_recipe_details.xml \\
                       --static-masks masks_fab_universal.json \\
                       --adaptive-threshold 0.80 --adaptive-rounds 3

        # Parse syslog, no cache, Groq provider
        python main.py --input 01012025.txt --fmt semistructured \\
                       --llm-provider groq --no-mask-cache

        # Parse with tighter Drain clustering
        python main.py --input fab_logs.log --drain-sim 0.6 --drain-depth 5

        # Evaluation mode (auto-downloads LogHub benchmark data)
        python main.py --mode eval --config eval.yaml

        # Offline / CI mode (MockLLM, universal masks only)
        python main.py --input fab_logs.log --llm-provider mock \\
                       --static-masks masks_fab_universal.json
        """),
    )
    p.add_argument("--mode",               choices=["test", "eval"], default="test")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--input",              default=None)
    p.add_argument("--fmt",                choices=["auto","xml","structured","semistructured","unstructured"], default="auto")
    p.add_argument("--message-col",        default="message")
    p.add_argument("--llm-provider",       default="anthropic",
                   help="LLM backend: anthropic (default), openai, groq, ollama, mock")
    p.add_argument("--no-mask-cache",      action="store_true")
    p.add_argument("--max-length",         type=int, default=128)
    p.add_argument("--max-logs",           type=int, default=500)
    p.add_argument("--output-dir",         default=None)
    p.add_argument("--config",             default=None)
    p.add_argument("--static-masks",       default=None,
                   help="Path to masks_fab_universal.json (auto-detected if omitted)")
    p.add_argument("--adaptive-threshold", type=float, default=ADAPTIVE_PARSE_THRESHOLD)
    p.add_argument("--adaptive-rounds",    type=int,   default=ADAPTIVE_MAX_RETRIES)
    p.add_argument("--drain-sim",          type=float, default=0.5,
                   help="Drain similarity threshold (0–1, default 0.5)")
    p.add_argument("--drain-depth",        type=int,   default=4,
                   help="Drain prefix-tree depth (default 4)")
    p.add_argument("--no-drain-state",     action="store_true",
                   help="Skip saving Drain state (faster, non-reproducible)")

    args = p.parse_args(argv)
    run(
        mode=args.mode,
        root_dir=Path(__file__).parent,
        seed=args.seed,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        input_path=args.input,
        input_fmt=args.fmt,
        message_col=args.message_col,
        llm_provider=args.llm_provider,
        use_mask_cache=not args.no_mask_cache,
        max_length=args.max_length,
        max_logs=args.max_logs,
        config=args.config,
        static_masks_path=args.static_masks,
        adaptive_threshold=args.adaptive_threshold,
        adaptive_rounds=args.adaptive_rounds,
        drain_sim_threshold=args.drain_sim,
        drain_depth=args.drain_depth,
        save_drain_state=not args.no_drain_state,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
