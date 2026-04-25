"""Evaluation runner for DeepParse.

For each dataset the runner:

1. Synthesises (or loads) a regex mask bundle from a small sample.
2. Parses every line through the deterministic Drain engine.
3. Compares the produced templates and cluster ids against a ground
   truth.  When the dataset directory contains ``templates.json``
   (mapping each line index to its canonical template id and template
   string) it is used as the gold reference; otherwise the runner falls
   back to a *canonical* pipeline that applies the built-in
   :data:`REGEX_CLASSES` to derive ground-truth templates.

The ``MacroAvg`` row reports unweighted means across datasets, matching
the reporting convention used in the paper.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..dataset_loader import Dataset, load_dataset
from ..drain.drain_engine import DrainEngine
from ..io_paths import build_paths
from ..logging_utils import get_logger
from ..masks_types import Mask
from ..metrics import grouping_accuracy, parsing_accuracy
from ..seeds import resolve_seed, set_global_seed
from ..synth import synthesize_masks
from ..utils.regex_library import canonical_masks, validate_regexes
from ..utils.yaml_loader import load_yaml

LOGGER = get_logger(__name__)


def _load_masks(path: Path) -> List[Mask]:
    data = json.loads(path.read_text(encoding="utf-8"))
    masks = [Mask.from_dict(entry) for entry in data]
    validate_regexes([mask.pattern for mask in masks])
    return masks


def _ground_truth_from_canonical(dataset: Dataset) -> Tuple[List[str], List[int]]:
    """Derive ground-truth templates by running the canonical mask
    bundle through Drain.  This is the reference behaviour when no
    annotated ground truth file is available.
    """
    engine = DrainEngine(masks=canonical_masks())
    pairs = engine.parse_with_ids(dataset.logs)
    cluster_ids = [pair[0] for pair in pairs]
    templates = [pair[1] for pair in pairs]
    return templates, cluster_ids


def _ground_truth_from_file(dataset: Dataset) -> Tuple[List[str], List[int]] | None:
    gt_path = dataset.path / "templates.json"
    if not gt_path.exists():
        return None
    payload = json.loads(gt_path.read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload, dict) else payload
    if entries is None or len(entries) != len(dataset.logs):
        LOGGER.warning(
            "Ground truth length mismatch for %s (entries=%s logs=%d); falling back to canonical",
            dataset.name,
            "missing" if entries is None else len(entries),
            len(dataset.logs),
        )
        return None
    templates = [entry["template"] for entry in entries]
    cluster_ids = [int(entry["cluster_id"]) for entry in entries]
    return templates, cluster_ids


@dataclass
class EvaluationConfig:
    base_config: Path
    datasets: Sequence[str]
    output_csv: Path
    timing_csv: Path

    @classmethod
    def from_file(cls, path: Path) -> "EvaluationConfig":
        config_data = load_yaml(path)
        base_path = Path(config_data["base_config"])
        datasets = config_data.get("datasets", [])
        output_csv = Path(config_data["output_csv"])
        timing_csv = Path(config_data.get("timing_csv", ""))
        return cls(
            base_config=base_path,
            datasets=datasets,
            output_csv=output_csv,
            timing_csv=timing_csv,
        )


class EvaluationRunner:
    def __init__(self, config_path: Path):
        self.config = EvaluationConfig.from_file(config_path)
        base_data = load_yaml(self.config.base_config)
        self.seed = resolve_seed(base_data.get("seed"))
        self.paths = build_paths(
            dataset_dir=base_data["dataset_dir"],
            mask_dir=base_data["mask_dir"],
            output_dir=base_data["output_dir"],
            log_dir=base_data["log_dir"],
        )
        self.mode = base_data.get("mode", "offline")
        self.k = int(base_data.get("k", 50))
        self.strict = bool(base_data.get("strict", False))

    def _ensure_masks(self, dataset: Dataset) -> Path:
        mask_path = self.paths.mask_dir / f"{dataset.name}.json"
        if not mask_path.exists():
            LOGGER.info("Masks missing for %s; synthesising", dataset.name)
            synthesize_masks(dataset, self.k, mask_path, mode=self.mode, strict=self.strict)
        return mask_path

    def evaluate_dataset(self, dataset_name: str) -> Dict[str, float | str]:
        dataset = load_dataset(dataset_name, self.paths)
        mask_path = self._ensure_masks(dataset)
        masks = _load_masks(mask_path)
        engine = DrainEngine(masks=masks)
        predicted_pairs = engine.parse_with_ids(dataset.logs)
        predicted_ids = [pair[0] for pair in predicted_pairs]
        predicted_templates = [pair[1] for pair in predicted_pairs]

        ground = _ground_truth_from_file(dataset) or _ground_truth_from_canonical(dataset)
        gt_templates, gt_ids = ground

        ga = grouping_accuracy(gt_ids, predicted_ids)
        pa = parsing_accuracy(gt_templates, predicted_templates)
        LOGGER.info("Dataset %s: GA=%.3f PA=%.3f", dataset_name, ga, pa)
        return {
            "dataset": dataset_name,
            "method": "DeepParse",
            "GA": round(ga, 4),
            "PA": round(pa, 4),
        }

    def run(self) -> List[Dict[str, float | str]]:
        set_global_seed(self.seed)
        rows = [self.evaluate_dataset(name) for name in self.config.datasets]
        if rows:
            ga_avg = sum(float(row["GA"]) for row in rows) / len(rows)
            pa_avg = sum(float(row["PA"]) for row in rows) / len(rows)
            rows.append(
                {
                    "dataset": "MacroAvg",
                    "method": "DeepParse",
                    "GA": round(ga_avg, 4),
                    "PA": round(pa_avg, 4),
                }
            )
        self.config.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["dataset", "method", "GA", "PA"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        LOGGER.info("Wrote metrics CSV to %s", self.config.output_csv)
        return rows
