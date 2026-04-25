"""Timing benchmark harness."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

from ..dataset_loader import load_dataset
from ..drain.drain_engine import DrainEngine
from ..io_paths import PathConfig
from ..logging_utils import get_logger
from ..masks_types import Mask
from ..utils.regex_library import validate_regexes

LOGGER = get_logger(__name__)


def _load_masks(path: Path) -> list[Mask]:
    data = json.loads(path.read_text(encoding="utf-8"))
    masks = [Mask(**entry) for entry in data]
    validate_regexes([mask.pattern for mask in masks])
    return masks


@dataclass
class TimingResult:
    dataset: str
    seconds: float


def run_timing_benchmark(dataset_name: str, paths: PathConfig, mask_path: Path, n: int, output_csv: Path) -> TimingResult:
    dataset = load_dataset(dataset_name, paths)
    masks = _load_masks(mask_path)
    engine = DrainEngine(masks=masks)
    sample_logs = dataset.logs[:n]
    start = time.perf_counter()
    engine.parse(sample_logs)
    elapsed = time.perf_counter() - start
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", "seconds", "n_logs"])
        writer.writeheader()
        writer.writerow({"dataset": dataset_name, "seconds": elapsed, "n_logs": len(sample_logs)})
    LOGGER.info("Timing benchmark for %s: %.4fs", dataset_name, elapsed)
    return TimingResult(dataset=dataset_name, seconds=elapsed)
