"""Benchmark datasets: Loghub-2k (parsing) and HDFS / BGL (anomaly detection).

Loghub-2k goes through DeepParse's fetcher (``deepparse.tools.fetch_loghub``),
so parsing numbers are computed on exactly the data the DeepParse
evaluation uses.  HDFS and BGL come from the Loghub Zenodo record, the
same archives LogBERT's ``scripts/download_*.sh`` fetch.  They are sampled
down to CPU-sized subsets with a fixed seed.
"""
from __future__ import annotations

import csv
import random
import re
import shutil
import tarfile
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from deepparse.tools.fetch_loghub import DEFAULT_SYSTEMS, fetch_systems

DATA_DIR = Path("artifacts/datasets")
ZENODO = "https://zenodo.org/record/3227177/files/{name}?download=1"
ARCHIVES = {"HDFS": "HDFS_1.tar.gz", "BGL": "BGL.tar.gz"}

_HDFS_LINE = re.compile(r"^(\d{6}) (\d{6}) (\d+) (\w+) ([^:]+): (.*)$")
_BLOCK = re.compile(r"blk_-?\d+")


def fetch_loghub2k(systems=DEFAULT_SYSTEMS, root: Path = DATA_DIR) -> Path:
    out = root / "loghub2k"
    fetch_systems(systems, out)
    return out


def fetch_archive(name: str, root: Path = DATA_DIR) -> Path:
    """Download and unpack the full HDFS or BGL archive once."""
    target = root / name
    marker = target / ".extracted"
    if marker.exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    archive = target / ARCHIVES[name]
    if not archive.exists():
        print(f"[datasets] downloading {ARCHIVES[name]} ...")
        request = urllib.request.Request(ZENODO.format(name=ARCHIVES[name]),
                                         headers={"User-Agent": "logpipe-eval/1.0"})
        with urllib.request.urlopen(request, timeout=120) as resp, archive.open("wb") as fh:
            shutil.copyfileobj(resp, fh, length=1 << 20)
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    marker.touch()
    return target


def _find(root: Path, filename: str) -> Path:
    found = next(root.rglob(filename), None)
    if found is None:
        raise FileNotFoundError(f"{filename} not found under {root}")
    return found


@dataclass
class Sessions:
    """Labelled sequences of raw log *content*, one per session or window."""

    name: str
    lines: dict[str, list[str]]
    labels: dict[str, int]

    @property
    def normal(self) -> list[str]:
        return [k for k, v in self.labels.items() if v == 0 and k in self.lines]

    @property
    def abnormal(self) -> list[str]:
        return [k for k, v in self.labels.items() if v == 1 and k in self.lines]


def load_hdfs(n_normal: int = 20_000, n_abnormal: int = 2_000, seed: int = 42,
              root: Path = DATA_DIR) -> Sessions:
    """Sample block sessions from HDFS_v1, keeping the log content per block (as LogBERT)."""
    base = fetch_archive("HDFS", root)
    labels_all: dict[str, int] = {}
    with _find(base, "anomaly_label.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels_all[row["BlockId"]] = int(row["Label"] == "Anomaly")
    rng = random.Random(seed)
    normal = sorted(b for b, y in labels_all.items() if y == 0)
    abnormal = sorted(b for b, y in labels_all.items() if y == 1)
    chosen = set(rng.sample(normal, min(n_normal, len(normal))))
    chosen |= set(rng.sample(abnormal, min(n_abnormal, len(abnormal))))
    lines: dict[str, list[str]] = {}
    with _find(base, "HDFS.log").open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            match = _HDFS_LINE.match(raw.rstrip("\n"))
            if not match:
                continue
            content = match.group(6)
            for block in set(_BLOCK.findall(content)):
                if block in chosen:
                    lines.setdefault(block, []).append(content)
    return Sessions("HDFS", lines, {b: labels_all[b] for b in chosen})


def _bgl_records(path: Path, max_lines: int) -> Iterator[tuple[int, int, str]]:
    """(unix_ts, label, content) for the first ``max_lines`` lines of BGL.log."""
    with path.open(encoding="utf-8", errors="replace") as fh:
        for i, raw in enumerate(fh):
            if i >= max_lines:
                break
            parts = raw.rstrip("\n").split(maxsplit=9)
            if len(parts) < 10 or not parts[1].isdigit():
                continue
            yield int(parts[1]), int(parts[0] != "-"), parts[9]


def load_bgl(max_lines: int = 1_000_000, window_s: int = 300, step_s: int = 60,
             root: Path = DATA_DIR) -> Sessions:
    """Sliding time windows over BGL (LogBERT's 5-minute window, 1-minute step)."""
    base = fetch_archive("BGL", root)
    records = sorted(_bgl_records(_find(base, "BGL.log"), max_lines), key=lambda r: r[0])
    lines: dict[str, list[str]] = {}
    labels: dict[str, int] = {}
    if not records:
        return Sessions("BGL", lines, labels)
    lo = 0
    start = records[0][0]
    end_ts = records[-1][0]
    while start <= end_ts:
        while lo < len(records) and records[lo][0] < start:
            lo += 1
        hi = lo
        while hi < len(records) and records[hi][0] < start + window_s:
            hi += 1
        if hi > lo:
            key = f"w{start}"
            lines[key] = [r[2] for r in records[lo:hi]]
            labels[key] = int(any(r[1] for r in records[lo:hi]))
        start += step_s
    return Sessions("BGL", lines, labels)
