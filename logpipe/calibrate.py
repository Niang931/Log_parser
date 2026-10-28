"""Offline calibration: bootstrap registry v1 from historical logs, and evaluate it.

Masks are synthesised from an entropy-diverse sample, then a *learning*
Drain clusters the whole history.  Each converged cluster becomes a
registry template with key names from the LLM adapter.

Every calibration also measures itself:

* **Holdout coverage** (no labels needed): mine on the first 80% of lines,
  freeze, and match the last 20%.  This is the share of fresh traffic the
  hot path would resolve immediately instead of leaving PENDING.
* **GA / PA** (when ground truth is given): DeepParse's grouping and
  parsing accuracy (:mod:`deepparse.metrics`) of the mined clusters,
  together with the plain DeepParse configuration (no IoT ``MEASURE``
  mask, default Drain parameters, pure-digit bucket keys) as a baseline.
* **Tuning** (``tune=True`` with ground truth): grid search over Drain
  depth and similarity threshold, keeping the best GA (then PA, then the
  fewest templates).
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from deepparse.drain.drain_engine import DrainEngine
from deepparse.masks_types import Mask
from deepparse.metrics import grouping_accuracy, parsing_accuracy

from .enrich.llm import LLMAdapter
from .enrich.validate import clean_keys, safe_masks
from .parsing.engine import make_engine
from .parsing.frozen_parser import FrozenParser, Status
from .parsing.kv import slot_labels
from .registry import NewTemplate, RegistrySnapshot, SchemaRegistry

# Most damn common mask, added here as a safeguard to never miss
# e.g. name=Kato, age=18, etc
MEASURE_MASK = Mask(
    label="MEASURE",
    pattern=r"(?<=[=:])[-+]?\d[^\s,;]*",
    justification="numeric key=value reading, optionally with a unit suffix",
)

DEFAULT_GRID = tuple((d, s) for d in (3, 4, 5) for s in (0.3, 0.4, 0.5, 0.6, 0.7))


def with_predefined_masks(masks: Sequence[Mask]) -> list[Mask]:
    """Insert MEASURE_MASK after timestamp masks so it doesn't get overriden."""
    timestamps = [m for m in masks if "TIME" in m.label.upper()]
    rest = [m for m in masks if "TIME" not in m.label.upper()]
    return [*timestamps, MEASURE_MASK, *rest]


@dataclass(frozen=True)
class GroundTruth:
    """Per-line gold templates and cluster ids, aligned with the calibration lines."""

    templates: tuple[str, ...]
    cluster_ids: tuple[int, ...]

    @classmethod
    def from_deepparse_dir(cls, path: Path) -> tuple[list[str], GroundTruth]:
        """Read a DeepParse dataset directory (``raw.log`` + ``templates.json``)."""
        lines = (path / "raw.log").read_text(encoding="utf-8").splitlines()
        payload = json.loads((path / "templates.json").read_text(encoding="utf-8"))
        entries = payload.get("entries") if isinstance(payload, dict) else payload
        if len(entries) != len(lines):
            raise ValueError(f"{path}: {len(entries)} labels for {len(lines)} lines")
        return lines, cls(
            tuple(e["template"] for e in entries), tuple(int(e["cluster_id"]) for e in entries)
        )


@dataclass
class CalibrationReport:
    lines: int
    templates: int
    masks: list[str]
    depth: int
    similarity_threshold: float
    holdout_lines: int
    holdout_coverage: float
    holdout_extraction_rate: float
    ga: float | None = None
    pa: float | None = None
    baseline_ga: float | None = None
    baseline_pa: float | None = None
    tuning: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class CalibrationResult:
    snapshot: RegistrySnapshot
    report: CalibrationReport


def _mine(lines: Sequence[str], masks: Sequence[Mask], depth: int, sim: float) -> DrainEngine:
    """Utility function to make an engine and add all the logs in one line."""
    engine = make_engine(masks, depth=depth, similarity_threshold=sim)
    for line in lines:
        engine.add_log(line)
    return engine


def _accuracy(
    lines: Sequence[str], masks: Sequence[Mask], depth: int, sim: float, truth: GroundTruth,
    *, deepparse_default: bool = False,
) -> tuple[float, float, int]:
    engine = (
        DrainEngine(depth=depth, similarity_threshold=sim, masks=list(masks))
        if deepparse_default
        else make_engine(masks, depth=depth, similarity_threshold=sim)
    )
    pairs = engine.parse_with_ids(lines)
    ga = grouping_accuracy(list(truth.cluster_ids), [cid for cid, _ in pairs])
    pa = parsing_accuracy(list(truth.templates), [tmpl for _, tmpl in pairs])
    return ga, pa, engine.num_clusters


def _snapshot(
    engine: DrainEngine, lines: Sequence[str], masks: Sequence[Mask], adapter: LLMAdapter | None
) -> RegistrySnapshot:

    # given a sequence of logs, add them to a dict with cluster id as key, max (5) per cluster
    samples: dict[int, list[str]] = {}
    for line in lines:
        cluster = engine.match(line)
        if cluster is not None:
            bucket = samples.setdefault(cluster.cluster_id, [])
            if len(bucket) < 5:
                bucket.append(line)

    # same pipeline as in worker
    new_templates = []
    for cluster in engine.clusters:
        tokens = tuple(cluster.template)
        proposed = adapter.name_keys(tokens, samples.get(cluster.cluster_id, [])) if adapter else None
        new_templates.append(NewTemplate(tokens=tokens, keys=tuple(clean_keys(proposed, tokens))))

    snapshot, _ids = SchemaRegistry().publish(
        new_templates, masks=masks, key_vocab=[k for t in new_templates for k in t.keys]
    )
    return snapshot


def _holdout(
    lines: Sequence[str], masks: Sequence[Mask], depth: int, sim: float, fraction: float
) -> tuple[int, float, float]:
    """(lines held out, share matched by the frozen tree, share of matches with a full payload)."""

    # train / test dataset seperation
    cut = int(len(lines) * (1 - fraction))
    train, test = lines[:cut], lines[cut:]
    if not train or not test:
        return 0, 0.0, 0.0

    snapshot = _snapshot(_mine(train, masks, depth, sim), train, masks, None)
    parser = FrozenParser(snapshot, depth=depth)
    final = [r for r in map(parser.parse, test) if r.status == Status.FINAL]
    # The frozen parser returns an empty payload when the extractor fails.
    extracted = sum(
        1 for r in final
        if r.payload or not slot_labels(snapshot.templates[r.template_id].tokens)
    )
    return len(test), len(final) / len(test), (extracted / len(final) if final else 0.0)


def calibrate(
    lines: Sequence[str],
    adapter: LLMAdapter,
    *,
    sample_size: int = 200,
    depth: int = 5,
    similarity_threshold: float = 0.4,
    ground_truth: GroundTruth | None = None,
    tune: bool = False,
    grid: Sequence[tuple[int, float]] = DEFAULT_GRID,
    holdout_fraction: float = 0.2,
) -> CalibrationResult:
    # Validation
    if ground_truth is None:
        lines = [line for line in lines if line.strip()]
    elif len(ground_truth.cluster_ids) != len(lines):
        raise ValueError("ground truth must be aligned with the calibration lines")

    if not lines:
        raise ValueError("calibration needs at least one non-empty log line")

    base_masks = safe_masks(adapter.synth_masks(lines, sample_size))
    masks = with_predefined_masks(base_masks)

    # this tune the model parameters to output best benchmark results
    tuning: list[dict] = []
    if tune and ground_truth is not None:
        for d, s in grid:
            ga, pa, n = _accuracy(lines, masks, d, s, ground_truth)
            tuning.append({"depth": d, "similarity_threshold": s, "GA": ga, "PA": pa, "templates": n})
        best = max(tuning, key=lambda r: (r["GA"], r["PA"], -r["templates"]))
        depth, similarity_threshold = best["depth"], best["similarity_threshold"]

    # perform the evaluation based on pipeline
    engine = _mine(lines, masks, depth, similarity_threshold)
    snapshot = _snapshot(engine, lines, masks, adapter)
    holdout_lines, coverage, extraction = _holdout(
        lines, masks, depth, similarity_threshold, holdout_fraction
    )

    report = CalibrationReport(
        lines=len(lines),
        templates=len(snapshot.templates),
        masks=[m.label for m in masks],
        depth=depth,
        similarity_threshold=similarity_threshold,
        holdout_lines=holdout_lines,
        holdout_coverage=round(coverage, 4),
        holdout_extraction_rate=round(extraction, 4),
        tuning=tuning,
    )

    # if ground truth was provided, then can further eval
    if ground_truth is not None:
        ga, pa, _ = _accuracy(lines, masks, depth, similarity_threshold, ground_truth)
        base_ga, base_pa, _ = _accuracy(
            lines, base_masks, 5, 0.4, ground_truth, deepparse_default=True
        )
        report.ga, report.pa = round(ga, 4), round(pa, 4)
        report.baseline_ga, report.baseline_pa = round(base_ga, 4), round(base_pa, 4)

    return CalibrationResult(snapshot, report)
