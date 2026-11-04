"""Deliberate retrain of the mask model: build the training set, then score adapters.

Nothing here runs on its own schedule; a person runs each step.

1. ``data``: build ``train.jsonl`` from Loghub-2k (50 entropy-greedy lines per
   system), the synthetic IoT fleet (N lines per template) and, if the zips are
   present, Loghub-2.0 samples. Every ``--holdout`` system is left out of all
   sources so it can stand in for a never-seen log source.
2. Fine-tune on a GPU (Colab: ``notebooks/colab_finetune_eval.ipynb``) with
   ``python -m deepparse.training.finetune --train <out>/train.jsonl``, then
   write one mask bundle per system with ``scripts/synth_all_with_adapter.sh``.
3. ``eval``: score mask bundles on CPU (GA/PA per system, plain Drain depth 5 /
   similarity 0.4) next to the offline masks, with held-out systems reported
   separately.

    python -m deepparse.tools.retrain data --holdout OpenSSH \\
        --loghub2-zips artifacts/datasets/loghub2_full/zips
    python -m deepparse.tools.retrain eval --holdout OpenSSH \\
        --masks v2=artifacts/retrain/masks_v2 --masks v1=artifacts/retrain/masks_v1
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from deepparse.api import synth_masks
from deepparse.drain.drain_engine import DrainEngine
from deepparse.masks_types import Mask
from deepparse.metrics.grouping_accuracy import grouping_accuracy
from deepparse.metrics.parsing_accuracy import parsing_accuracy
from deepparse.tools.build_training_set import build_examples, write_jsonl
from deepparse.tools.fetch_loghub import DEFAULT_SYSTEMS, fetch_systems

IOT = "IoT_fleet"


def build_data(
    out: Path, holdout: list[str], data_2k: Path, iot_dir: Path, loghub2_zips: Path | None,
    loghub2_per_template: int, iot_per_template: int, entropy_k: int,
) -> dict:
    """Write ``out/train.jsonl`` and ``out/manifest.json``; return the manifest."""
    from deepparse.tools.sample_loghub_full import main as sample_full
    from deepparse.tools.synth_iot_fleet import generate_iot_fleet, write_dataset

    fetch_systems(DEFAULT_SYSTEMS, data_2k)
    if not (iot_dir / "raw.log").exists():
        write_dataset(iot_dir, *generate_iot_fleet())
    train = [s for s in DEFAULT_SYSTEMS if s not in holdout]

    counts: dict[str, int] = {}
    examples = []
    for system in train:
        ex = build_examples(data_2k / system, entropy_k=entropy_k)
        counts[f"loghub2k/{system}"] = len(ex)
        examples += ex
    if IOT not in holdout:
        ex = build_examples(iot_dir, per_template=iot_per_template)
        counts[IOT] = len(ex)
        examples += ex
    if loghub2_zips is not None:
        sampled = out / "loghub2_sampled"
        sample_full([
            "--zips", str(loghub2_zips), "--out", str(sampled), "--exclude-2k", str(data_2k),
            "--per-template", str(loghub2_per_template), "--systems", *train,
        ])
        for system in train:
            if (sampled / system / "raw.log").exists():
                ex = build_examples(sampled / system)
                counts[f"loghub2.0/{system}"] = len(ex)
                examples += ex

    n = write_jsonl(examples, out / "train.jsonl")
    manifest = {"holdout": holdout, "examples": n, "per_source": counts,
                "entropy_k": entropy_k, "iot_per_template": iot_per_template,
                "loghub2_per_template": loghub2_per_template if loghub2_zips else None}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _load(path: Path) -> tuple[list[str], list[str], list[int]]:
    lines = [line for line in (path / "raw.log").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    entries = json.loads((path / "templates.json").read_text(encoding="utf-8"))["entries"]
    return lines, [e["template"] for e in entries], [e["cluster_id"] for e in entries]


def score(masks: list[Mask], lines: list[str], templates: list[str], ids: list[int]) -> tuple:
    pairs = DrainEngine(masks=masks).parse_with_ids(lines)
    return (round(grouping_accuracy(ids, [c for c, _ in pairs]), 4),
            round(parsing_accuracy(templates, [t for _, t in pairs]), 4))


def evaluate(mask_sets: dict[str, Path], data_2k: Path, iot_dir: Path) -> dict:
    """GA/PA per system for the offline masks and every ``name -> masks dir``."""
    systems = {s: data_2k / s for s in DEFAULT_SYSTEMS} | {IOT: iot_dir}
    results: dict[str, dict[str, tuple]] = {}
    for system, path in systems.items():
        if not (path / "raw.log").exists():
            continue
        lines, templates, ids = _load(path)
        row = {"offline": score([Mask.from_dict(m) for m in synth_masks(lines, 50)],
                                lines, templates, ids)}
        for name, mask_dir in mask_sets.items():
            bundle = mask_dir / f"{system}.json"
            if bundle.exists():
                masks = [Mask(**m) for m in json.loads(bundle.read_text(encoding="utf-8"))]
                row[name] = score(masks, lines, templates, ids)
        results[system] = row
    return results


def _report(results: dict, holdout: list[str]) -> str:
    names = list(dict.fromkeys(n for row in results.values() for n in row))
    head = "| system | " + " | ".join(f"{n} GA | {n} PA" for n in names) + " |"
    out = [head, "|" + "---|" * (1 + 2 * len(names))]

    def row(label: str, cells: list) -> str:
        return f"| {label} | " + " | ".join(cells) + " |"

    for system, scores in results.items():
        cells = [f"{scores[n][i]:.3f}" if n in scores else "" for n in names for i in (0, 1)]
        out.append(row(system + (" (held out)" if system in holdout else ""), cells))
    for label, keep in (("mean, trained systems", lambda s: s not in holdout),
                        ("mean, held out", lambda s: s in holdout)):
        cells = []
        for n in names:
            vals = [r[n] for s, r in results.items() if keep(s) and n in r]
            cells += [f"{statistics.mean(v[i] for v in vals):.3f}" if vals else "" for i in (0, 1)]
        out.append(row(f"**{label}**", cells))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in (data := sub.add_parser("data")), (ev := sub.add_parser("eval")):
        cmd.add_argument("--holdout", nargs="*", default=["OpenSSH"],
                         help="systems kept out of training and reported separately")
        cmd.add_argument("--data-2k", type=Path, default=Path("artifacts/data"))
        cmd.add_argument("--iot-dir", type=Path, default=Path("artifacts/data") / IOT)
    data.add_argument("--out", type=Path, default=Path("artifacts/retrain"))
    data.add_argument("--loghub2-zips", type=Path, default=None)
    data.add_argument("--loghub2-per-template", type=int, default=2)
    data.add_argument("--iot-per-template", type=int, default=4)
    data.add_argument("--entropy-k", type=int, default=50)
    ev.add_argument("--masks", action="append", default=[], metavar="NAME=DIR",
                    help="mask bundles <DIR>/<system>.json to score (repeatable)")
    ev.add_argument("--out", type=Path, default=None, help="write results JSON + table here")
    args = parser.parse_args(argv)

    if args.cmd == "data":
        manifest = build_data(args.out, args.holdout, args.data_2k, args.iot_dir,
                              args.loghub2_zips, args.loghub2_per_template,
                              args.iot_per_template, args.entropy_k)
        print(json.dumps(manifest, indent=2))
        return 0

    mask_sets = {}
    for item in args.masks:
        name, sep, path = item.partition("=")
        if not sep:
            parser.error(f"--masks expects NAME=DIR, got {item!r}")
        mask_sets[name] = Path(path)
    results = evaluate(mask_sets, args.data_2k, args.iot_dir)
    table = _report(results, args.holdout)
    print(table)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "retrain_eval.json").write_text(json.dumps(results, indent=2), "utf-8")
        (args.out / "retrain_eval.md").write_text(table + "\n", "utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
