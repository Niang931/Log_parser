"""Centralised path helpers for datasets and outputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    dataset_dir: Path
    mask_dir: Path
    output_dir: Path
    log_dir: Path

    def ensure(self) -> None:
        for path in (self.dataset_dir, self.mask_dir, self.output_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)


def build_paths(dataset_dir: str, mask_dir: str, output_dir: str, log_dir: str) -> PathConfig:
    cfg = PathConfig(
        dataset_dir=Path(dataset_dir),
        mask_dir=Path(mask_dir),
        output_dir=Path(output_dir),
        log_dir=Path(log_dir),
    )
    cfg.ensure()
    return cfg
