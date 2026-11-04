"""Sample training lines per template from the full Loghub-2.0 datasets.

Loghub-2.0 (Zenodo record 8275861, CC-BY-4.0) labels every line of the 14
systems, with far more templates than the 2k benchmark. This streams each
``<System>.zip`` (the CSV is read in place, never extracted), keeps up to
``--per-template`` lines per gold template by reservoir sampling, and drops
every line that appears in the Loghub-2k evaluation set so training never
sees an eval line. Output uses the 2k layout (``raw.log`` + ``templates.json``)
so :mod:`deepparse.tools.build_training_set` reads it unchanged.

    python -m deepparse.tools.sample_loghub_full --zips artifacts/datasets/loghub2_full/zips \\
        --out artifacts/datasets/loghub2_full/sampled --exclude-2k artifacts/datasets/loghub2k
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import zipfile
from pathlib import Path

from deepparse.tools.fetch_loghub import DEFAULT_SYSTEMS, _write_dataset


def sample_system(
    zip_path: Path, per_template: int, exclude: set[str], seed: int
) -> tuple[list[str], list[dict]]:
    """Reservoir-sample up to ``per_template`` lines per template from one Loghub-2.0 zip."""
    rng = random.Random(seed)
    reservoirs: dict[str, list[str]] = {}
    seen: dict[str, int] = {}
    with zipfile.ZipFile(zip_path) as archive:
        member = next(
            n for n in archive.namelist() if n.endswith("_full.log_structured.csv")
        )
        with archive.open(member) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"))
            for row in reader:
                content = (row.get("Content") or "").rstrip("\n")
                template = (row.get("EventTemplate") or "").rstrip("\n")
                if not content or not template or content in exclude:
                    continue
                n = seen[template] = seen.get(template, 0) + 1
                bucket = reservoirs.setdefault(template, [])
                if len(bucket) < per_template:
                    bucket.append(content)
                elif (j := rng.randrange(n)) < per_template:
                    bucket[j] = content
    logs, entries = [], []
    for cluster_id, (template, bucket) in enumerate(sorted(reservoirs.items())):
        for line in dict.fromkeys(bucket):  # identical lines teach nothing twice
            logs.append(line)
            entries.append({"cluster_id": cluster_id, "template": template})
    return logs, entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zips", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-2k", type=Path, default=None,
                        help="Loghub-2k dir; its lines are never sampled")
    parser.add_argument("--systems", nargs="*", default=DEFAULT_SYSTEMS)
    parser.add_argument("--per-template", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)
    csv.field_size_limit(sys.maxsize)

    for system in args.systems:
        zip_path = args.zips / f"{system}.zip"
        if not zip_path.exists():
            print(f"[sample_loghub_full] {system}: {zip_path} missing, skipped", file=sys.stderr)
            continue
        exclude: set[str] = set()
        if args.exclude_2k and (raw := args.exclude_2k / system / "raw.log").exists():
            exclude = set(raw.read_text(encoding="utf-8").splitlines())
        logs, entries = sample_system(zip_path, args.per_template, exclude, args.seed)
        _write_dataset(args.out / system, system, logs, entries)
        templates = len({e["cluster_id"] for e in entries})
        print(f"[sample_loghub_full] {system}: {len(logs)} lines / {templates} templates")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
