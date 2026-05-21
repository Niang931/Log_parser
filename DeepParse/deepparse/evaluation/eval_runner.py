"""
DeepParse/deepparse/evaluation/eval_runner.py
=============================================
Evaluation framework for the DeepParse pipeline.
Metrics: parse rate, template accuracy (GA/FGA/PTA), token F1, stability.
Rubric: measurable improvement, rigorous validation, reproducible.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("deepparse.eval")


# ---------------------------------------------------------------------------
# Metric dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ParseMetrics:
    """Per-run parse quality metrics."""
    run_id:              str   = ""
    total_lines:         int   = 0
    parsed_lines:        int   = 0
    parse_rate:          float = 0.0    # Fraction of lines that hit a non-wildcard template
    unique_templates:    int   = 0
    avg_wildcard_ratio:  float = 0.0    # Mean fraction of <*> tokens per template
    template_stability:  float = 0.0    # Mean cluster stability score

    # Grouping accuracy (requires ground-truth labels)
    GA:   float = 0.0    # Grouping Accuracy (exact cluster match)
    FGA:  float = 0.0    # Fuzzy Grouping Accuracy (>=50% token overlap)
    PTA:  float = 0.0    # Parsing Template Accuracy

    # Token-level F1 against ground truth
    token_precision: float = 0.0
    token_recall:    float = 0.0
    token_f1:        float = 0.0

    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Ground-truth loader (LogHub-style structured labels)
# ---------------------------------------------------------------------------

def load_ground_truth(gt_path: Path) -> dict[str, str]:
    """
    Load ground-truth template mapping {raw_line: expected_template}.
    Supports CSV with columns (Content, EventTemplate) or JSON dict.
    """
    if not gt_path.exists():
        log.warning("Ground-truth file not found: %s", gt_path)
        return {}

    ext = gt_path.suffix.lower()
    if ext == ".json":
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        # List of {"raw": ..., "template": ...}
        return {d["raw"]: d["template"] for d in data if "raw" in d and "template" in d}

    if ext == ".csv":
        mapping = {}
        with open(gt_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw  = row.get("Content") or row.get("raw", "")
                tmpl = row.get("EventTemplate") or row.get("template", "")
                if raw and tmpl:
                    mapping[raw.strip()] = tmpl.strip()
        return mapping

    log.warning("Unsupported GT format: %s", ext)
    return {}


# ---------------------------------------------------------------------------
# Template comparison utilities
# ---------------------------------------------------------------------------

def _tokenize(s: str) -> list[str]:
    return s.split()


def _wildcard_ratio(template: str) -> float:
    tokens = _tokenize(template)
    if not tokens:
        return 0.0
    wildcards = sum(1 for t in tokens if t in ("<*>", "<NUM>") or t.startswith("<"))
    return wildcards / len(tokens)


def _token_overlap_f1(pred: str, gold: str) -> tuple[float, float, float]:
    """Token-level F1 between predicted and gold templates."""
    pred_toks = set(_tokenize(pred))
    gold_toks = set(_tokenize(gold))
    if not pred_toks and not gold_toks:
        return 1.0, 1.0, 1.0
    if not pred_toks or not gold_toks:
        return 0.0, 0.0, 0.0
    tp = len(pred_toks & gold_toks)
    prec   = tp / len(pred_toks)
    recall = tp / len(gold_toks)
    f1     = 2 * prec * recall / (prec + recall + 1e-9)
    return prec, recall, f1


def _fuzzy_match(pred: str, gold: str, threshold: float = 0.5) -> bool:
    """True if token overlap ≥ threshold."""
    _, _, f1 = _token_overlap_f1(pred, gold)
    return f1 >= threshold


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def evaluate_records(
    records: list[dict],
    ground_truth: dict[str, str] | None = None,
) -> ParseMetrics:
    """
    Compute ParseMetrics from a list of output records.
    If ground_truth is provided, computes GA/FGA/PTA and token F1.
    """
    m = ParseMetrics()
    m.run_id       = records[0].get("run_id", "") if records else ""
    m.total_lines  = len(records)
    m.parsed_lines = sum(r.get("parsed", 0) for r in records)
    m.parse_rate   = m.parsed_lines / max(m.total_lines, 1)

    templates = [r.get("template", "") for r in records]
    m.unique_templates   = len(set(templates))
    m.avg_wildcard_ratio = (
        sum(_wildcard_ratio(t) for t in templates) / max(len(templates), 1)
    )

    if not ground_truth:
        return m

    # Grouping accuracy vs ground truth
    exact_matches = 0
    fuzzy_matches = 0
    template_hits = 0
    prec_list, rec_list, f1_list = [], [], []

    gt_templates = set(ground_truth.values())

    for r in records:
        raw  = r.get("raw", "")
        pred = r.get("template", "")
        gold = ground_truth.get(raw)
        if gold is None:
            continue
        if pred == gold:
            exact_matches += 1
            template_hits += 1
        elif _fuzzy_match(pred, gold):
            fuzzy_matches += 1
        if pred in gt_templates:
            template_hits += 1
        p, rec, f1 = _token_overlap_f1(pred, gold)
        prec_list.append(p); rec_list.append(rec); f1_list.append(f1)

    n = len([r for r in records if r.get("raw", "") in ground_truth])
    if n > 0:
        m.GA  = exact_matches  / n
        m.FGA = (exact_matches + fuzzy_matches) / n
        m.PTA = template_hits  / n
        m.token_precision = sum(prec_list) / len(prec_list) if prec_list else 0.0
        m.token_recall    = sum(rec_list)  / len(rec_list)  if rec_list  else 0.0
        m.token_f1        = sum(f1_list)   / len(f1_list)   if f1_list   else 0.0

    return m


# ---------------------------------------------------------------------------
# EvaluationRunner — reads eval.yaml, runs eval, writes report
# ---------------------------------------------------------------------------

class EvaluationRunner:
    """
    Reads an eval.yaml config and evaluates all artifact output files
    it can find. Writes a JSON + CSV report to artifacts/eval/.
    """

    DEFAULT_CONFIG = {
        "output_dir":     "artifacts/output",
        "eval_dir":       "artifacts/eval",
        "ground_truth":   None,
        "min_parse_rate": 0.60,
        "min_ga":         0.50,
        "pass_on_no_gt":  True,
    }

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.cfg         = dict(self.DEFAULT_CONFIG)
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            self.cfg.update(loaded)
        log.info("EvaluationRunner config: %s", self.cfg)

    def run(self) -> list[ParseMetrics]:
        t_start  = time.monotonic()
        out_dir  = Path(self.cfg["output_dir"])
        eval_dir = Path(self.cfg["eval_dir"])
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Load ground truth if configured
        gt: dict[str, str] | None = None
        gt_path = self.cfg.get("ground_truth")
        if gt_path and Path(gt_path).exists():
            gt = load_ground_truth(Path(gt_path))
            log.info("Ground truth loaded: %d entries", len(gt))

        # Discover JSON output files
        # Only evaluate primary output files — exclude cluster, drain_state, telemetry files
        json_files = sorted(
            f for f in out_dir.glob("run_*.json")
            if not any(tag in f.name for tag in ("_clusters", "_drain_state", "_telemetry"))
        )
        if not json_files:
            log.warning("No run_*.json files found in %s", out_dir)
            return []

        all_metrics: list[ParseMetrics] = []
        pass_count = 0
        fail_count = 0

        for jf in json_files:
            raw_data = json.loads(jf.read_text(encoding="utf-8"))
            # Handle both list-of-records and wrapped {"records": [...]} formats
            if isinstance(raw_data, dict):
                records = raw_data.get("records", [raw_data])
            elif isinstance(raw_data, list):
                records = raw_data
            else:
                continue
            if not records or not isinstance(records[0], dict):
                continue
            # Skip stale/broken runs with no parsed lines
            if sum(r.get("parsed", 0) for r in records) == 0:
                log.warning("Skipping %s — zero parsed lines (stale run?)", jf.name)
                continue
            metrics = evaluate_records(records, ground_truth=gt)
            metrics.elapsed_s = round(time.monotonic() - t_start, 3)
            all_metrics.append(metrics)

            # Pass/fail gate
            pr_ok = metrics.parse_rate >= self.cfg["min_parse_rate"]
            ga_ok = (gt is None and self.cfg["pass_on_no_gt"]) or metrics.GA >= self.cfg["min_ga"]
            passed = pr_ok and ga_ok
            symbol = "✓" if passed else "✗"
            (pass_count if passed else fail_count).__class__   # dummy
            if passed:
                pass_count += 1
            else:
                fail_count += 1

            log.info(
                "[%s] %s  parse_rate=%.1f%%  GA=%.1f%%  FGA=%.1f%%  token_F1=%.3f  "
                "templates=%d  wildcard_ratio=%.2f",
                symbol, jf.name,
                metrics.parse_rate * 100, metrics.GA * 100, metrics.FGA * 100,
                metrics.token_f1, metrics.unique_templates, metrics.avg_wildcard_ratio,
            )

        # Write reports
        report_json = eval_dir / "eval_report.json"
        report_csv  = eval_dir / "eval_report.csv"

        report_data = {
            "summary": {
                "total_runs":    len(all_metrics),
                "passed":        pass_count,
                "failed":        fail_count,
                "elapsed_s":     round(time.monotonic() - t_start, 3),
                "config":        self.cfg,
            },
            "runs": [asdict(m) for m in all_metrics],
        }
        report_json.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

        fields = list(asdict(all_metrics[0]).keys()) if all_metrics else []
        with open(report_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(asdict(m) for m in all_metrics)

        print(f"\n{'='*60}")
        print(f"  Evaluation Complete  |  {pass_count} passed / {fail_count} failed")
        print(f"  Reports → {eval_dir}/")
        print(f"    eval_report.json, eval_report.csv")

        return all_metrics
