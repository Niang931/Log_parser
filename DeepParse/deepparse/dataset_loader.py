"""Dataset loading utilities for corrected LogHub-style corpora.

A *dataset directory* is expected to contain at minimum:

* ``raw.log`` — one log line per row.
* ``manifest.json`` (optional) — metadata: name, line count, sha256.
* ``templates.json`` (optional) — annotated ground truth as a list of
  objects ``{"cluster_id": <int>, "template": <str>}`` aligned to
  ``raw.log``.

The bundled :func:`load_dataset` helper auto-creates a tiny synthetic
``DemoTiny`` dataset with full ground truth so that the demo pipeline
runs entirely offline and produces non-trivial GA/PA scores.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from .io_paths import PathConfig
from .logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class Dataset:
    name: str
    path: Path
    logs: Sequence[str]

    @property
    def checksum(self) -> str:
        data = "\n".join(self.logs).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


EXPECTED_FILES = ["raw.log"]


# A small synthetic corpus that exercises every variable class supported
# by the canonical regex bundle.  The ground truth is generated from the
# same canonical pipeline so that a faithful DeepParse implementation
# achieves GA = PA = 1.0 on this dataset.
_DEMO_LOGS: List[str] = [
    "2024-01-01 00:00:00 INFO Worker-1 Completed task 42 in 0.5s",
    "2024-01-01 00:00:01 INFO Worker-2 Completed task 43 in 0.7s",
    "2024-01-01 00:00:02 INFO Worker-3 Completed task 44 in 0.9s",
    "2024-01-01 00:00:03 WARN Worker-1 Retrying task 45 attempt 1",
    "2024-01-01 00:00:04 WARN Worker-2 Retrying task 46 attempt 2",
    "2024-01-01 00:00:05 ERROR Worker-3 Failed task 47 with code 500",
    "2024-01-01 00:00:06 ERROR Worker-1 Failed task 48 with code 503",
    "2024-01-01 00:00:07 INFO Connection from 192.168.1.10 established",
    "2024-01-01 00:00:08 INFO Connection from 192.168.1.11 established",
    "2024-01-01 00:00:09 INFO Connection from 10.0.0.5 established",
    "2024-01-01 00:00:10 DEBUG Memory pointer 0xDEADBEEF freed",
    "2024-01-01 00:00:11 DEBUG Memory pointer 0xCAFEBABE freed",
    "2024-01-01 00:00:12 INFO File /var/log/app.log rotated successfully",
    "2024-01-01 00:00:13 INFO File /var/log/system.log rotated successfully",
]


def _create_demo_dataset(path: Path) -> None:
    if (path / "raw.log").exists() and (path / "templates.json").exists():
        return
    LOGGER.info("Creating bundled demo dataset at %s", path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "raw.log").write_text("\n".join(_DEMO_LOGS) + "\n", encoding="utf-8")

    # Build ground truth using the canonical mask bundle so that any
    # faithful DeepParse implementation matches it exactly.
    from .drain.drain_engine import DrainEngine  # local import to avoid cycle
    from .utils.regex_library import canonical_masks

    engine = DrainEngine(masks=canonical_masks())
    pairs = engine.parse_with_ids(_DEMO_LOGS)
    entries = [
        {"cluster_id": cid, "template": template}
        for cid, template in pairs
    ]
    (path / "templates.json").write_text(
        json.dumps({"entries": entries}, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "name": "DemoTiny",
        "logs": len(_DEMO_LOGS),
        "checksum": hashlib.sha256("\n".join(_DEMO_LOGS).encode("utf-8")).hexdigest(),
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_dataset(name: str, paths: PathConfig, create_demo: bool = True) -> Dataset:
    dataset_root = paths.dataset_dir / name
    if create_demo and name == "DemoTiny":
        _create_demo_dataset(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset {name} missing at {dataset_root}")

    for expected in EXPECTED_FILES:
        file_path = dataset_root / expected
        if not file_path.exists():
            raise FileNotFoundError(f"Expected file {expected} in {dataset_root}")

    logs = [
        line.rstrip("\n")
        for line in (dataset_root / "raw.log").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset = Dataset(name=name, path=dataset_root, logs=logs)
    LOGGER.info("Loaded dataset %s with %d logs", name, len(logs))
    return dataset


def load_many(names: Iterable[str], paths: PathConfig) -> List[Dataset]:
    return [load_dataset(name, paths) for name in names]
