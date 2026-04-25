"""Download the corrected LogHub-2k benchmark from logpai/loghub-2.0.

For every system the script writes::

    <out>/<NAME>/raw.log              # one log line per row
    <out>/<NAME>/templates.json       # ground truth: cluster_id + template per line
    <out>/<NAME>/manifest.json        # name, line count, sha256

The same layout is consumed by :mod:`deepparse.dataset_loader` and the
evaluation runner — no further conversion is needed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.error
import urllib.request
from io import StringIO
from pathlib import Path
from typing import Iterable

LOGHUB_BASE = (
    "https://raw.githubusercontent.com/logpai/loghub-2.0/main/2k_dataset/{system}/"
)

# 16 systems shipped under loghub-2.0/2k_dataset.  These are the same set
# the DeepParse paper evaluates on (Section 'Datasets', LogHub-2k).
DEFAULT_SYSTEMS = [
    "Apache",
    "BGL",
    "HDFS",
    "HPC",
    "Hadoop",
    "HealthApp",
    "Linux",
    "Mac",
    "OpenSSH",
    "OpenStack",
    "Proxifier",
    "Spark",
    "Thunderbird",
    "Zookeeper",
]


# This just download the file from the url
def _fetch(url: str, timeout: float = 30.0) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": "deepparse-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


# Read the entire csv file into logs, entries lists (entries has cluster_id and template)
def _convert(csv_text: str) -> tuple[list[str], list[dict]]:
    """Convert a ``*_structured_corrected.csv`` payload to (logs, entries)."""
    reader = csv.DictReader(StringIO(csv_text))
    logs: list[str] = []
    entries: list[dict] = []
    template_to_id: dict[str, int] = {}
    for row in reader:
        content = row.get("Content", "").rstrip("\n")
        template = row.get("EventTemplate", "").rstrip("\n")
        if not content:
            continue
        cluster_id = template_to_id.setdefault(template, len(template_to_id))
        logs.append(content)
        entries.append({"cluster_id": cluster_id, "template": template})
    return logs, entries


# Just writing what we got from _convert into files and also a manifest file for metadata
def _write_dataset(out_dir: Path, name: str, logs: list[str], entries: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw.log").write_text("\n".join(logs) + "\n", encoding="utf-8")
    (out_dir / "templates.json").write_text(
        json.dumps({"entries": entries}, indent=2), encoding="utf-8"
    )
    manifest = {
        "name": name,
        "logs": len(logs),
        "templates": len({e["cluster_id"] for e in entries}),
        "checksum": hashlib.sha256("\n".join(logs).encode("utf-8")).hexdigest(),
        "source": "logpai/loghub-2.0 (corrected)",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# Write the files only if they do not exists, if not then ignore them
def fetch_systems(systems: Iterable[str], out_dir: Path, force: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for system in systems:
        target = out_dir / system
        if (target / "raw.log").exists() and (target / "templates.json").exists() and not force:
            print(f"[fetch_loghub] {
                  system}: already present, skipping (use --force to refetch)")
            continue
        url = LOGHUB_BASE.format(system=system) + \
            f"{system}_2k.log_structured_corrected.csv"
        print(f"[fetch_loghub] {system}: downloading...")
        try:
            csv_text = _fetch(url)
        except urllib.error.URLError as exc:  # network failure
            print(f"[fetch_loghub] {system}: FAILED ({exc})", file=sys.stderr)
            continue
        logs, entries = _convert(csv_text)
        if not logs:
            print(f"[fetch_loghub] {
                  system}: empty content; skipping", file=sys.stderr)
            continue
        _write_dataset(target, system, logs, entries)
        templates = len({e["cluster_id"] for e in entries})
        print(f"[fetch_loghub] {system}: wrote {
              len(logs)} logs / {templates} templates")


def download_logs(
    systems: list[str] | None = DEFAULT_SYSTEMS,
    out: Path | str = "artifacts/data",
    force: bool = False,
) -> None:
    """
    Core logic that can be imported and reused.
    """
    if systems is None:
        systems = DEFAULT_SYSTEMS

    out_path = Path(out)
    fetch_systems(systems, out_path, force=force)


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/data",
                        type=Path, help="Output directory")
    parser.add_argument(
        "--systems",
        nargs="*",
        default=DEFAULT_SYSTEMS,
        help="Subset of systems to fetch (default: all 16)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if files already exist"
    )

    args = parser.parse_args(argv)
    download_logs(systems=args.systems, out=args.out, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
