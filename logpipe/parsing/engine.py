"""The one place logpipe builds Drain engines, so calibration, the frozen hot-path
tree and the enrichment worker's shadow tree always bucket lines identically."""
from __future__ import annotations

from collections.abc import Sequence

from deepparse.drain.drain_engine import DrainEngine
from deepparse.masks_types import Mask


def make_engine(
    masks: Sequence[Mask], *, depth: int = 5, similarity_threshold: float = 0.4
) -> DrainEngine:
    return DrainEngine(depth=depth, similarity_threshold=similarity_threshold, masks=list(masks))
