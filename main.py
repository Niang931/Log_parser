"""
main.py — DeepParse log parsing pipeline
=========================================
Output format: SQLite (primary, machine-readable + queryable)
               CSV     (human-readable, spreadsheet-friendly)
               JSON    (API/downstream integration)

All three are written on every run to artifacts/output/.
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
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from llm.registry import init_llm
from DeepParse.deepparse import Drain
from DeepParse.deepparse.synth.hf_deepseek_r1 import synthesize_online
from DeepParse.deepparse.evaluation.eval_runner import EvaluationRunner
from DeepParse.deepparse.tools.fetch_loghub import download_logs

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("deepparse.main")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Pin all RNG sources so runs are reproducible."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # noqa: F401
        np.random.seed(seed)
    except ImportError:
        pass
    log.info("Global seed set to %d", seed)


# ---------------------------------------------------------------------------
# Input loaders — structured / semi-structured / unstructured
# ---------------------------------------------------------------------------

def load_structured(path: str, message_col: str = "message") -> list[str]:
    """
    Load log messages from a structured file (CSV, JSON-lines, Parquet).

    Parameters
    ----------
    path        : file path
    message_col : column name that contains the raw log text
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Structured input not found: {path}")

    ext = p.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")          # handles BOM
    elif ext in (".jsonl", ".json"):
        df = pd.read_json(path, lines=True)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported structured format: {ext}")

    if message_col not in df.columns:
        available = list(df.columns)
        raise KeyError(
            f"Column '{message_col}' not found. Available columns: {available}"
        )

    return df[message_col].dropna().astype(str).tolist()


# Common syslog prefix: "Jan  5 12:00:00 hostname daemon[123]:"
_SYSLOG_PREFIX = re.compile(
    r"^\w{3}\s+\d{1,2}\s+[\d:]+\s+\S+\s+\S+[:\s]\s*"
)
# Common log4j / HDFS prefix: "2024-01-05 12:00:00,123 INFO ClassName:"
_LOG4J_PREFIX = re.compile(
    r"^\d{4}-\d{2}-\d{2}[\sT][\d:,\.]+\s+\w+\s+\S+[:\s]\s*"
)


def load_semistructured(path: str) -> list[str]:
    """
    Load semi-structured logs (syslog, HDFS, log4j).
    Strips timestamp / severity / hostname prefixes so Drain sees clean messages.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Semi-structured input not found: {path}")

    messages: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Try to strip known prefixes; fall back to raw line
            for pattern in (_SYSLOG_PREFIX, _LOG4J_PREFIX):
                stripped = pattern.sub("", line)
                if stripped and stripped != line:
                    line = stripped
                    break
            messages.append(line)
    return messages


def load_unstructured(path: str, max_length: int = 128) -> list[str]:
    """
    Load plain text log files (.log, .txt).
    Each non-empty line becomes one log entry.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Unstructured input not found: {path}")

    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [l.strip() for l in f if l.strip()]

    # Truncate long lines at read time (before dedup) so identical truncated
    # lines collapse correctly.
    return [l[:max_length] for l in lines]


def load_logs(
    path: str,
    fmt: str = "auto",
    message_col: str = "message",
    max_length: int = 128,
) -> list[str]:
    """
    Unified entry-point. fmt ∈ {"auto", "structured", "semistructured", "unstructured"}.
    "auto" infers from file extension.
    """
    ext = Path(path).suffix.lower()
    if fmt == "auto":
        if ext in (".csv", ".jsonl", ".json", ".parquet"):
            fmt = "structured"
        elif ext in (".log", ".txt"):
            fmt = "unstructured"
        else:
            fmt = "semistructured"

    if fmt == "structured":
        return load_structured(path, message_col)
    if fmt == "semistructured":
        return load_semistructured(path)
    if fmt == "unstructured":
        return load_unstructured(path, max_length)
    raise ValueError(f"Unknown fmt: {fmt}")


# ---------------------------------------------------------------------------
# Normalize + validate
# ---------------------------------------------------------------------------

def normalize(
    logs: list[str],
    max_length: int = 128,
    min_length: int = 5,
    max_logs: int = 500,
) -> list[str]:
    """
    Clean and deduplicate log lines before they reach the LLM.

    Raises
    ------
    ValueError  if the list is empty after filtering (pipeline would silently
                produce zero templates, which is almost never intentional).
    """
    out: list[str] = []
    seen: set[str] = set()
    skipped_empty = 0
    skipped_short = 0
    skipped_dup = 0

    for raw in logs:
        line = raw.strip()

        if not line:
            skipped_empty += 1
            continue
        if len(line) < min_length:
            skipped_short += 1
            continue
        if len(line) > max_length:
            line = line[:max_length]
        if line in seen:
            skipped_dup += 1
            continue

        seen.add(line)
        out.append(line)

        if len(out) >= max_logs:
            log.warning(
                "Reached max_logs=%d; %d lines after this point were ignored.",
                max_logs,
                len(logs) - len(out) - skipped_empty - skipped_short - skipped_dup,
            )
            break

    log.info(
        "normalize: kept=%d  skipped(empty=%d short=%d dup=%d)",
        len(out),
        skipped_empty,
        skipped_short,
        skipped_dup,
    )

    if not out:
        raise ValueError(
            "normalize() produced zero log lines. "
            "Check your input file and column name."
        )

    return out


# ---------------------------------------------------------------------------
# Mask cache  (skip the LLM on replay runs)
# ---------------------------------------------------------------------------

def save_masks(masks: list[Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([m.to_dict() for m in masks], f, indent=2)
    log.info("Masks saved → %s", path)


def load_masks_from_cache(path: str | Path) -> list[dict] | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    log.info("Masks loaded from cache ← %s  (%d masks)", path, len(data))
    return data


# ---------------------------------------------------------------------------
# Template → regex
# ---------------------------------------------------------------------------

def template_to_regex(template: str) -> str:
    """Convert a Drain template (uses <*> wildcards) to a Python regex."""
    escaped = re.escape(template)
    regex = escaped.replace(r"<\*>", r"(.*?)")
    return f"^{regex}$"


def extract_variables(template: str, line: str) -> dict[str, str] | None:
    """
    Return a dict of captured wildcard values, or None if the template
    does not match the line (indicates a Drain clustering mismatch).
    Uses fullmatch so partial prefix matches don't silently pass.
    """
    try:
        regex = template_to_regex(template)
        match = re.fullmatch(regex, line)
        if match is None:
            return None
        return {f"var_{i}": v for i, v in enumerate(match.groups())}
    except re.error as exc:
        log.warning("Regex error for template %r: %s", template, exc)
        return None


# ---------------------------------------------------------------------------
# Parse → structured records
# ---------------------------------------------------------------------------

def parse_to_records(
    logs: list[str],
    drain: Drain,
    run_id: str,
) -> list[dict]:
    """
    Run Drain.parse_all and build one dict per log line.

    Schema
    ------
    run_id        TEXT   — unique identifier for this pipeline run
    timestamp_utc TEXT   — ISO-8601 wall-clock time of this parse (UTC)
    raw           TEXT   — original log line
    template      TEXT   — Drain template (e.g. "PacketResponder <*> terminating")
    variables     TEXT   — JSON-encoded dict of wildcard captures
    parsed        INT    — 1 if regex matched, 0 if not
    var_count     INT    — number of wildcards captured
    """
    templates = drain.parse_all(logs)
    now = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []

    for line, template in zip(logs, templates):
        variables = extract_variables(template, line)
        records.append(
            {
                "run_id":        run_id,
                "timestamp_utc": now,
                "raw":           line,
                "template":      template,
                "variables":     json.dumps(variables) if variables else "{}",
                "parsed":        1 if variables is not None else 0,
                "var_count":     len(variables) if variables else 0,
            }
        )

    parsed_count = sum(r["parsed"] for r in records)
    log.info(
        "parse_to_records: %d lines → %d parsed (%.1f%%)",
        len(records),
        parsed_count,
        100 * parsed_count / max(len(records), 1),
    )
    return records


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── SQLite ─────────────────────────────────────────────────────────────────

_CREATE_PARSED_LOGS = textwrap.dedent("""
    CREATE TABLE IF NOT EXISTS parsed_logs (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id        TEXT    NOT NULL,
        timestamp_utc TEXT    NOT NULL,
        raw           TEXT    NOT NULL,
        template      TEXT    NOT NULL,
        variables     TEXT    NOT NULL DEFAULT '{}',
        parsed        INTEGER NOT NULL DEFAULT 0,
        var_count     INTEGER NOT NULL DEFAULT 0
    );
""")

_CREATE_RUN_META = textwrap.dedent("""
    CREATE TABLE IF NOT EXISTS run_meta (
        run_id        TEXT PRIMARY KEY,
        created_utc   TEXT NOT NULL,
        seed          INTEGER,
        llm_provider  TEXT,
        mask_count    INTEGER,
        log_count     INTEGER,
        parsed_count  INTEGER,
        notes         TEXT
    );
""")

_CREATE_TEMPLATES = textwrap.dedent("""
    CREATE TABLE IF NOT EXISTS templates (
        run_id       TEXT NOT NULL,
        template     TEXT NOT NULL,
        occurrences  INTEGER NOT NULL DEFAULT 1,
        PRIMARY KEY (run_id, template)
    );
""")


def write_sqlite(
    records: list[dict],
    db_path: str | Path,
    run_meta: dict | None = None,
) -> Path:
    """
    Upsert records into SQLite.

    Three tables:
      parsed_logs  — one row per log line (the main fact table)
      templates    — aggregated template frequency per run
      run_meta     — run-level metadata (seed, LLM provider, counts, …)
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")   # safe for concurrent readers
    con.execute("PRAGMA foreign_keys=ON;")

    con.execute(_CREATE_PARSED_LOGS)
    con.execute(_CREATE_RUN_META)
    con.execute(_CREATE_TEMPLATES)
    con.commit()

    # Insert parsed_logs rows
    con.executemany(
        """
        INSERT INTO parsed_logs
            (run_id, timestamp_utc, raw, template, variables, parsed, var_count)
        VALUES
            (:run_id, :timestamp_utc, :raw, :template, :variables, :parsed, :var_count)
        """,
        records,
    )

    # Aggregate template counts
    from collections import Counter
    counts = Counter(r["template"] for r in records)
    run_id = records[0]["run_id"] if records else "unknown"
    con.executemany(
        """
        INSERT INTO templates (run_id, template, occurrences)
        VALUES (?, ?, ?)
        ON CONFLICT(run_id, template) DO UPDATE SET
            occurrences = occurrences + excluded.occurrences
        """,
        [(run_id, tmpl, cnt) for tmpl, cnt in counts.items()],
    )

    # Write run-level metadata
    if run_meta:
        con.execute(
            """
            INSERT OR REPLACE INTO run_meta
                (run_id, created_utc, seed, llm_provider,
                 mask_count, log_count, parsed_count, notes)
            VALUES
                (:run_id, :created_utc, :seed, :llm_provider,
                 :mask_count, :log_count, :parsed_count, :notes)
            """,
            run_meta,
        )

    con.commit()
    con.close()
    log.info("SQLite written → %s  (%d rows)", db_path, len(records))
    return db_path


# ── CSV ────────────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "run_id", "timestamp_utc", "raw", "template",
    "variables", "parsed", "var_count",
]


def write_csv(records: list[dict], csv_path: str | Path) -> Path:
    """
    Write records to CSV.  Human-readable in any spreadsheet tool;
    'variables' column is JSON-encoded so it round-trips cleanly.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    log.info("CSV written → %s  (%d rows)", csv_path, len(records))
    return csv_path


# ── JSON ───────────────────────────────────────────────────────────────────

def write_json(records: list[dict], json_path: str | Path) -> Path:
    """
    Write records to a JSON array. Each 'variables' value is re-parsed
    from its string form so downstream consumers get a proper nested object.
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    enriched = []
    for r in records:
        row = dict(r)
        try:
            row["variables"] = json.loads(r["variables"])
        except (json.JSONDecodeError, TypeError):
            row["variables"] = {}
        enriched.append(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    log.info("JSON written → %s  (%d records)", json_path, len(enriched))
    return json_path


# ---------------------------------------------------------------------------
# Core pipeline steps
# ---------------------------------------------------------------------------

def synthesize_masks(
    logs: list[str],
    llm_provider: str,
    max_length: int,
    self_consistency_attempts: int,
    mask_cache_path: Path,
    use_cache: bool = True,
) -> tuple[list[Any], list[dict]]:
    """
    Either load masks from cache or call the LLM.
    Returns (raw mask objects, list-of-dicts for Drain.load_masks).
    """
    if use_cache:
        cached = load_masks_from_cache(mask_cache_path)
        if cached is not None:
            return [], cached   # raw objects not needed when replaying

    llm = init_llm(provider=llm_provider)
    masks = synthesize_online(
        logs=logs,
        llm=llm,
        max_length=max_length,
        self_consistency_attempts=self_consistency_attempts,
    )

    if not masks:
        log.warning("LLM returned zero masks — Drain will run without masking.")

    save_masks(masks, mask_cache_path)
    return masks, [m.to_dict() for m in masks]


# ---------------------------------------------------------------------------
# Top-level test() and eval()
# ---------------------------------------------------------------------------

def test(
    seed: int = 42,
    llm_provider: str = "groq",
    use_mask_cache: bool = True,
    output_dir: Path | None = None,
) -> None:
    """
    Smoke-test the full pipeline on a small hard-coded sample and write all
    three output formats to output_dir.
    """
    set_global_seed(seed)

    max_length = 128
    run_id = f"test_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{seed}"

    raw_logs = [
        "PacketResponder 1 for block blk_38865049064139660 terminating",
        "PacketResponder 0 for block blk_-6952295868487656571 terminating",
        "BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010",
        "workerEnv.init() ok /etc/httpd/conf/workers2.properties",
        "mod_jk child workerEnv in error state 6",
    ]

    logs = normalize(raw_logs, max_length=max_length)

    if output_dir is None:
        output_dir = Path(__file__).parent / "artifacts" / "output"
    _ensure_dir(output_dir)
    mask_cache = output_dir / "masks.json"

    masks_raw, masks_dicts = synthesize_masks(
        logs=logs,
        llm_provider=llm_provider,
        max_length=max_length,
        self_consistency_attempts=1,
        mask_cache_path=mask_cache,
        use_cache=use_mask_cache,
    )

    drain = Drain()
    drain.load_masks(masks_dicts)

    records = parse_to_records(logs, drain, run_id)

    # Print a quick preview to stdout
    print(f"\n{'─'*60}")
    print(f"run_id: {run_id}")
    print(f"{'─'*60}")
    for r in records:
        print(f"  raw      : {r['raw']}")
        print(f"  template : {r['template']}")
        print(f"  variables: {r['variables']}")
        print(f"  parsed   : {'✓' if r['parsed'] else '✗'}")
        print()

    # Write all three formats
    run_meta = {
        "run_id":        run_id,
        "created_utc":   datetime.now(timezone.utc).isoformat(),
        "seed":          seed,
        "llm_provider":  llm_provider,
        "mask_count":    len(masks_dicts),
        "log_count":     len(records),
        "parsed_count":  sum(r["parsed"] for r in records),
        "notes":         "test mode — hard-coded sample logs",
    }

    db_path  = write_sqlite(records, output_dir / "parsed_logs.db", run_meta)
    csv_path = write_csv(records,    output_dir / f"{run_id}.csv")
    js_path  = write_json(records,   output_dir / f"{run_id}.json")

    print(f"Outputs written:")
    print(f"  SQLite → {db_path}")
    print(f"  CSV    → {csv_path}")
    print(f"  JSON   → {js_path}")


def eval(
    config: str,
    seed: int | None = None,
    output_dir: Path | None = None,
) -> None:
    runner = EvaluationRunner(Path(config))
    if seed is not None:
        runner.seed = seed
    runner.run()
    print(f"Wrote metrics CSV to {runner.config.output_csv}")


# ---------------------------------------------------------------------------
# run() — unified entrypoint (also callable programmatically)
# ---------------------------------------------------------------------------

def run(
    mode: str,
    root_dir: Path,
    *,
    # shared
    seed: int = 42,
    output_dir: Path | None = None,
    # test mode
    input_path: str | None = None,
    input_fmt: str = "auto",
    message_col: str = "message",
    llm_provider: str = "groq",
    use_mask_cache: bool = True,
    max_length: int = 128,
    # eval mode
    config: str | None = None,
) -> None:
    """
    Unified entrypoint — called by CLI and importable programmatically.

    Parameters
    ----------
    mode          : "test" | "eval"
    root_dir      : project root (used to locate default paths)
    seed          : global RNG seed for reproducibility
    output_dir    : where to write SQLite / CSV / JSON (default: artifacts/output)
    input_path    : path to a log file (structured / semi / unstructured)
    input_fmt     : "auto" | "structured" | "semistructured" | "unstructured"
    message_col   : column name when input_fmt="structured"
    llm_provider  : passed to init_llm()
    use_mask_cache: if True, skip LLM call when masks.json already exists
    max_length    : truncate log lines at this many chars
    config        : path to eval.yaml (eval mode only)
    """
    set_global_seed(seed)

    if output_dir is None:
        output_dir = root_dir / "artifacts" / "output"
    _ensure_dir(output_dir)

    # ── TEST mode ──────────────────────────────────────────────────────────
    if mode == "test":
        if input_path:
            # Ingest a real file instead of the built-in sample
            raw_logs = load_logs(
                input_path,
                fmt=input_fmt,
                message_col=message_col,
                max_length=max_length,
            )
            logs = normalize(raw_logs, max_length=max_length)
            run_id = (
                f"test_{Path(input_path).stem}_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{seed}"
            )
            mask_cache = output_dir / f"masks_{Path(input_path).stem}.json"

            masks_raw, masks_dicts = synthesize_masks(
                logs=logs,
                llm_provider=llm_provider,
                max_length=max_length,
                self_consistency_attempts=1,
                mask_cache_path=mask_cache,
                use_cache=use_mask_cache,
            )

            drain = Drain()
            drain.load_masks(masks_dicts)

            records = parse_to_records(logs, drain, run_id)
            run_meta = {
                "run_id":        run_id,
                "created_utc":   datetime.now(timezone.utc).isoformat(),
                "seed":          seed,
                "llm_provider":  llm_provider,
                "mask_count":    len(masks_dicts),
                "log_count":     len(records),
                "parsed_count":  sum(r["parsed"] for r in records),
                "notes":         f"input: {input_path}",
            }

            db_path  = write_sqlite(records, output_dir / "parsed_logs.db", run_meta)
            csv_path = write_csv(records,    output_dir / f"{run_id}.csv")
            js_path  = write_json(records,   output_dir / f"{run_id}.json")

            print(f"SQLite → {db_path}")
            print(f"CSV    → {csv_path}")
            print(f"JSON   → {js_path}")
        else:
            # Fall back to the built-in smoke-test sample
            test(
                seed=seed,
                llm_provider=llm_provider,
                use_mask_cache=use_mask_cache,
                output_dir=output_dir,
            )
        return

    # ── EVAL mode ──────────────────────────────────────────────────────────
    if mode == "eval":
        out_folder = root_dir / "artifacts" / "data"
        download_logs(out=out_folder)
        eval_config = config or str(root_dir / "eval.yaml")
        eval(eval_config, seed=seed, output_dir=output_dir)
        return

    raise ValueError(f"Unknown mode: {mode!r}. Choose 'test' or 'eval'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DeepParse — log parsing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["test", "eval"],
        default="test",
        help="Pipeline mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global RNG seed (reproducibility)",
    )
    parser.add_argument(
        "--input",
        default=None,
        metavar="PATH",
        help="Path to a log file (CSV / JSONL / Parquet / syslog / .log). "
             "Omit to run the built-in smoke-test sample.",
    )
    parser.add_argument(
        "--fmt",
        choices=["auto", "structured", "semistructured", "unstructured"],
        default="auto",
        help="Input format (auto infers from file extension)",
    )
    parser.add_argument(
        "--message-col",
        default="message",
        help="Column name for log text when --fmt=structured",
    )
    parser.add_argument(
        "--llm-provider",
        default="groq",
        help="LLM provider passed to init_llm()",
    )
    parser.add_argument(
        "--no-mask-cache",
        action="store_true",
        help="Force LLM re-synthesis even if masks.json already exists",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Truncate log lines to this many characters",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Where to write SQLite / CSV / JSON outputs",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to eval.yaml (eval mode only)",
    )

    args = parser.parse_args(argv)

    root_dir   = Path(__file__).parent
    output_dir = Path(args.output_dir) if args.output_dir else None

    run(
        mode=args.mode,
        root_dir=root_dir,
        seed=args.seed,
        output_dir=output_dir,
        input_path=args.input,
        input_fmt=args.fmt,
        message_col=args.message_col,
        llm_provider=args.llm_provider,
        use_mask_cache=not args.no_mask_cache,
        max_length=args.max_length,
        config=args.config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
