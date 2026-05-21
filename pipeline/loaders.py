"""
pipeline/loaders.py — Multi-format log file loaders + normaliser
Supports: XML, CSV, TSV, JSON, YAML, INI, syslog, plain text
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.security import sanitize_for_llm
from pipeline.telemetry import TELEMETRY

log = logging.getLogger("deepparse.loaders")

_SYSLOG_PREFIX = re.compile(r"^\w{3}\s+\d{1,2}\s+[\d:]+\s+\S+\s+\S+[:\s]\s*")
_LOG4J_PREFIX  = re.compile(r"^\d{4}-\d{2}-\d{2}[\sT][\d:,\.]+\s+\w+\s+\S+[:\s]\s*")


# ---------------------------------------------------------------------------
# XML
# ---------------------------------------------------------------------------

def load_xml(path: str) -> list[str]:
    """
    Extract three line types per element:
      1. Tag.attr = value     (one per attribute)
      2. <Tag>content</Tag>   (text content)
      3. Tag k="v" k="v"      (combined when >=2 attributes)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"XML not found: {path}")
    try:
        root  = ET.parse(path).getroot()
        lines: list[str] = []
        for elem in root.iter():
            tag     = re.sub(r'\{[^}]+\}', '', elem.tag)
            attribs = {re.sub(r'\{[^}]+\}', '', k): v
                       for k, v in (elem.attrib or {}).items()}
            content = (elem.text or "").strip()
            for ak, av in attribs.items():
                lines.append(f"{tag}.{ak} = {av}")
            if content:
                lines.append(f"<{tag}>{content}</{tag}>")
            if len(attribs) >= 2:
                kv = " ".join(f'{k}="{v}"' for k, v in attribs.items())
                lines.append(f"{tag} {kv}")
        log.info("XML: extracted %d lines from %s", len(lines), p.name)
        return lines if lines else _xml_raw_fallback(path)
    except Exception as exc:
        log.warning("XML parse failed (%s) — raw fallback", exc)
        return _xml_raw_fallback(path)


def _xml_raw_fallback(path: str) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return [l.strip() for l in f
                if l.strip() and not l.strip().startswith("<?")]


# ---------------------------------------------------------------------------
# CSV / TSV / JSON / YAML / INI
# ---------------------------------------------------------------------------

def load_structured(path: str, message_col: str = "message") -> list[str]:
    p   = Path(path)
    ext = p.suffix.lower()

    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # ── TSV: early return as ColName = value lines ────────────────────────
    if ext == ".tsv":
        try:
            df = pd.read_csv(path, sep="\t", dtype=str,
                             encoding="utf-8-sig", on_bad_lines="skip")
        except TypeError:
            df = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep="\t", dtype=str, encoding="latin-1")
        real = [c for c in df.columns if "Unnamed" not in str(c)]
        if len(real) >= 2:
            log.info("TSV: ColName=value lines (%d cols) for %s", len(real), p.name)
            out: list[str] = []
            for _, row in df.iterrows():
                for col in real:
                    val = str(row.get(col, "")).strip()
                    if val and val.lower() != "nan":
                        out.append(f"{col} = {val}")
            return out

    # ── CSV ───────────────────────────────────────────────────────────────
    elif ext == ".csv":
        try:
            df = pd.read_csv(path, dtype=str, encoding="utf-8-sig",
                             on_bad_lines="skip")
        except TypeError:
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

    # ── Key-value table detection (2 real cols + Unnamed cols) ────────────
    real_cols    = [c for c in df.columns if "Unnamed" not in str(c)]
    unnamed_cols = [c for c in df.columns if "Unnamed" in str(c)]
    if len(real_cols) <= 2 and len(unnamed_cols) > 0:
        log.info("Key-value table in '%s'", p.name)
        lines: list[str] = []
        for _, row in df.iterrows():
            vals = [str(v).strip() for v in row
                    if str(v).strip() not in ("nan", "")]
            if len(vals) >= 2:
                lines.append(f"{vals[0]}: {vals[1]}")
            elif len(vals) == 1:
                lines.append(vals[0])
        return lines

    # ── Standard column detection ─────────────────────────────────────────
    if message_col in df.columns:
        return df[message_col].dropna().astype(str).tolist()

    for candidate in ["message", "msg", "log", "text", "content",
                      "description", "event", "entry", "line", "value",
                      "logmessage", "log_message", "raw"]:
        match = next((c for c in df.columns if c.lower() == candidate), None)
        if match:
            log.info("Auto-detected column '%s' in %s", match, p.name)
            return df[match].dropna().astype(str).tolist()

    str_cols = df.select_dtypes(include="object").columns.tolist()
    if str_cols:
        avg_lens = {c: df[c].dropna().astype(str).str.len().mean()
                    for c in str_cols}
        best = max(avg_lens, key=avg_lens.get)
        log.info("Using longest-text column '%s' in %s", best, p.name)
        return df[best].dropna().astype(str).tolist()

    log.warning("Full-row join fallback for %s", p.name)
    return (
        df.fillna("").astype(str)
          .apply(lambda row: " | ".join(v for v in row if v.strip()), axis=1)
          .tolist()
    )


def _flatten_obj(obj: Any, prefix: str = "") -> list[str]:
    """
    Flatten YAML/JSON to key: value lines.
    Array indices normalised to [*] so Drain clusters
    Steps[0].name and Steps[1].name into one template.
    """
    lines: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lines.extend(_flatten_obj(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for v in obj:
            lines.extend(_flatten_obj(v, f"{prefix}[*]"))
    else:
        val = str(obj).strip()
        if val and val.lower() not in ("none", "null", "nan"):
            lines.append(f"{prefix}: {val}")
    return lines


def _load_ini_as_lines(path: str) -> list[str]:
    """Produce Section.key = value lines from INI/cfg files."""
    lines: list[str] = []
    section = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped or stripped.startswith(("#", ";")):
                continue
            m = re.match(r"^\[([^\]]+)\]$", stripped)
            if m:
                section = m.group(1).strip()
                continue
            m2 = re.match(r"^([^=:]+)[=:](.*)$", stripped)
            if m2:
                key = m2.group(1).strip()
                val = m2.group(2).strip()
                lines.append(f"{section}.{key} = {val}" if section
                              else f"{key} = {val}")
            else:
                lines.append(stripped)
    return lines


# ---------------------------------------------------------------------------
# Syslog / unstructured
# ---------------------------------------------------------------------------

def load_semistructured(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
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
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, encoding="utf-8", errors="replace") as f:
        return [l.strip()[:max_length] for l in f if l.strip()]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

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
        elif ext in (".csv", ".tsv", ".jsonl", ".json", ".parquet",
                     ".yaml", ".yml", ".ini", ".cfg"):
            fmt = "structured"
        elif ext in (".log", ".txt"):
            fmt = "unstructured"
        else:
            fmt = "semistructured"
    log.info("Loading '%s' as '%s'", path, fmt)
    if fmt == "xml":            return load_xml(path)
    if fmt == "structured":     return load_structured(path, message_col)
    if fmt == "semistructured": return load_semistructured(path)
    if fmt == "unstructured":   return load_unstructured(path, max_length)
    raise ValueError(f"Unknown format: {fmt}")


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------

def normalize(
    logs: list[str],
    max_length: int = 128,
    min_length: int = 5,
    max_logs: int   = 500,
    sanitize: bool  = True,
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
            line = line[:max_length]; counters["truncated"] += 1
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
            log.warning("Max logs ceiling: %d", max_logs); break

    log.info("Normalise: kept=%d %s", len(out), counters)
    TELEMETRY.record("normalise", kept=len(out), **counters)
    if not out:
        raise ValueError("Zero lines after normalisation — check file format.")
    return out