"""Parsing accuracy on Loghub-2k through the calibration path.

Each system is calibrated exactly as production does it, with ground truth
attached, so the DeepParse metrics (GA/PA) come out of ``calibrate`` itself.
``baseline`` is the unmodified DeepParse configuration (same masks without
the IoT ``MEASURE`` mask, Drain depth 5 / similarity 0.4), i.e. what
DeepParse's own ``EvaluationRunner`` reports for the offline stub.
``sample_size=50`` matches DeepParse's default ``k``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from deepparse.tools.fetch_loghub import DEFAULT_SYSTEMS

from ..calibrate import GroundTruth, calibrate
from ..enrich.llm import LLMAdapter
from .datasets import fetch_loghub2k


def run_parsing_eval(adapter: LLMAdapter, systems=DEFAULT_SYSTEMS, tune: bool = True,
                     out_dir: Path | None = None, log=print) -> list[dict]:
    root = fetch_loghub2k(systems)
    rows = []
    for system in systems:
        path = root / system
        if not (path / "templates.json").exists():
            log(f"[parsing] {system}: missing, skipped")
            continue
        lines, truth = GroundTruth.from_deepparse_dir(path)
        result = calibrate(lines, adapter, sample_size=50, ground_truth=truth, tune=tune)
        r = result.report
        rows.append({
            "system": system, "GA": r.ga, "PA": r.pa,
            "baseline_GA": r.baseline_ga, "baseline_PA": r.baseline_pa,
            "depth": r.depth, "similarity_threshold": r.similarity_threshold,
            "templates": r.templates, "gold_templates": len(set(truth.cluster_ids)),
            "holdout_coverage": r.holdout_coverage,
        })
        log(f"[parsing] {system}: GA={r.ga} PA={r.pa} (DeepParse baseline GA={r.baseline_ga} "
            f"PA={r.baseline_pa}) depth={r.depth} sim={r.similarity_threshold} "
            f"coverage={r.holdout_coverage}")
    if rows:
        avg = {k: round(sum(row[k] for row in rows) / len(rows), 4)
               for k in ("GA", "PA", "baseline_GA", "baseline_PA", "holdout_coverage")}
        rows.append({"system": "MacroAvg", **avg})
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "parsing.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        (out_dir / "parsing.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows
