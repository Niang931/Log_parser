"""
pipeline/writers.py — Output writers: SQLite, CSV, JSON, cluster report
"""
from __future__ import annotations

import csv
import json
import logging
import sqlite3
from pathlib import Path

from DeepParse.deepparse import Drain

log = logging.getLogger("deepparse.writers")

# ---------------------------------------------------------------------------
# SQLite DDL
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
);"""

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
);"""

_CREATE_QUARANTINE = """
CREATE TABLE IF NOT EXISTS quarantine (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL,
    raw              TEXT,
    template         TEXT,
    validation_error TEXT
);"""


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

def write_sqlite(
    records: list[dict],
    db_path: Path,
    run_meta: dict | None   = None,
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
               VALUES (:run_id,:created_utc,:seed,:llm_provider,:mask_count,
                       :log_count,:parsed_count,:parse_rate,:invalid_count,:notes)""",
            run_meta,
        )
    if invalid:
        con.executemany(
            """INSERT INTO quarantine (run_id,raw,template,validation_error)
               VALUES (:run_id,:raw,:template,:_validation_error)""",
            [{**r, "run_id": run_meta["run_id"] if run_meta else "?"}
             for r in invalid],
        )
    con.commit()
    con.close()
    return db_path


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csv(records: list[dict], csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "timestamp_utc", "raw", "template",
              "variables", "parsed", "var_count"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    return csv_path


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

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
# Cluster analytics report
# ---------------------------------------------------------------------------

def write_cluster_report(drain: Drain, output_dir: Path, run_id: str) -> Path:
    """Downstream analytics artifact — engineers query templates, not raw lines."""
    clusters = drain.cluster_summary()
    report   = {
        "run_id":               run_id,
        "total_clusters":       len(clusters),
        "template_fingerprint": drain.template_fingerprint(),
        "drain_stats":          drain.stats,
        "top_clusters":         clusters[:50],
    }
    path = output_dir / f"{run_id}_clusters.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Cluster report → %s (%d clusters)", path, len(clusters))
    return path


# ---------------------------------------------------------------------------
# Optional PostgreSQL writer
# ---------------------------------------------------------------------------

def write_postgres(
    records: list[dict],
    pg_url: str,
    run_meta: dict | None      = None,
    invalid: list[dict] | None = None,
) -> None:
    """Write records to PostgreSQL (requires POSTGRES_URL env var)."""
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        raise ImportError("pip install psycopg2-binary")

    con = psycopg2.connect(pg_url)
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS parsed_logs (
            id SERIAL PRIMARY KEY,
            run_id TEXT NOT NULL,
            timestamp_utc TIMESTAMPTZ NOT NULL,
            raw TEXT NOT NULL,
            template TEXT NOT NULL,
            variables JSONB NOT NULL DEFAULT '{}',
            parsed SMALLINT NOT NULL,
            var_count INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_pl_run ON parsed_logs(run_id);
        CREATE INDEX IF NOT EXISTS idx_pl_tmpl ON parsed_logs(template);
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS run_meta (
            run_id TEXT PRIMARY KEY,
            created_utc TIMESTAMPTZ,
            seed INTEGER, llm_provider TEXT,
            mask_count INTEGER, log_count INTEGER,
            parsed_count INTEGER, parse_rate REAL,
            invalid_count INTEGER, notes TEXT
        );
    """)

    psycopg2.extras.execute_batch(
        cur,
        """INSERT INTO parsed_logs
           (run_id,timestamp_utc,raw,template,variables,parsed,var_count)
           VALUES (%(run_id)s,%(timestamp_utc)s,%(raw)s,%(template)s,
                   %(variables)s::jsonb,%(parsed)s,%(var_count)s)""",
        records, page_size=500,
    )

    if run_meta:
        cur.execute(
            """INSERT INTO run_meta
               (run_id,created_utc,seed,llm_provider,mask_count,log_count,
                parsed_count,parse_rate,invalid_count,notes)
               VALUES (%(run_id)s,%(created_utc)s,%(seed)s,%(llm_provider)s,
                       %(mask_count)s,%(log_count)s,%(parsed_count)s,
                       %(parse_rate)s,%(invalid_count)s,%(notes)s)
               ON CONFLICT (run_id) DO UPDATE SET
                   parse_rate=EXCLUDED.parse_rate,
                   parsed_count=EXCLUDED.parsed_count""",
            run_meta,
        )

    con.commit(); cur.close(); con.close()
    log.info("PostgreSQL: %d records written", len(records))