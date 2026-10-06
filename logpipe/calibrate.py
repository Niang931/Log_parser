"""Offline calibration: bootstrap registry v1 from historical logs.

Masks are synthesised from an entropy-diverse sample, then a *learning*
Drain clusters the whole history.  Each converged cluster becomes a
registry template with key names from the LLM adapter.
"""
from __future__ import annotations

from collections.abc import Sequence

from deepparse.drain.drain_engine import DrainEngine
from deepparse.masks_types import Mask

from .enrich.llm import LLMAdapter
from .registry import NewTemplate, RegistrySnapshot, SchemaRegistry

# Most damn common mask, added here as a safeguard to never miss
# e.g. name=Kato, age=18, etc
MEASURE_MASK = Mask(
    label="MEASURE",
    pattern=r"(?<=[=:])[-+]?\d[^\s,;]*",
    justification="numeric key=value reading, optionally with a unit suffix",
)


def with_predefined_masks(masks: Sequence[Mask]) -> list[Mask]:
    """Insert MEASURE_MASK after timestamp masks so it doesn't get overriden."""
    timestamps = [m for m in masks if "TIME" in m.label.upper()]
    rest = [m for m in masks if "TIME" not in m.label.upper()]
    return [*timestamps, MEASURE_MASK, *rest]


def calibrate(
    lines: Sequence[str],
    adapter: LLMAdapter,
    *,
    sample_size: int = 200,
    depth: int = 5,
    similarity_threshold: float = 0.4,
) -> RegistrySnapshot:
    lines = [line for line in lines if line.strip()]
    if not lines:
        raise ValueError("calibration needs at least one non-empty log line")

    masks = with_predefined_masks(adapter.synth_masks(lines, sample_size))
    engine = DrainEngine(depth=depth, similarity_threshold=similarity_threshold, masks=masks)
    samples: dict[int, list[str]] = {}

    for line in lines:
        cluster = engine.add_log(line)
        bucket = samples.setdefault(cluster.cluster_id, [])
        if len(bucket) < 5:
            bucket.append(line)

    new_templates = []
    for cluster in engine.clusters():
        tokens = tuple(cluster.template)
        keys = tuple(adapter.name_keys(tokens, samples[cluster.cluster_id]))
        new_templates.append(NewTemplate(tokens=tokens, keys=keys))

    snapshot, _ids = SchemaRegistry().publish(new_templates, masks=masks)
    return snapshot
