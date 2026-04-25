"""Utilities for deterministic seeding across libraries."""
from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Optional

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - executed when NumPy missing
    np = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeedState:
    """Container storing the active random seed."""

    seed: int
    deterministic: bool


def set_global_seed(seed: int, deterministic: bool = True) -> SeedState:
    """Seed Python, NumPy, and PyTorch if available.

    Args:
        seed: Seed value to apply across libraries.
        deterministic: Whether to enable deterministic CUDA kernels.

    Returns:
        SeedState describing the applied seed.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:  # pragma: no branch - optional
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(deterministic)
        if torch.cuda.is_available():  # pragma: no branch - depends on hardware
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    state = SeedState(seed=seed, deterministic=deterministic)
    LOGGER.debug("Seeds initialised: %s", state)
    return state


def resolve_seed(seed: Optional[int]) -> int:
    """Return a consistent seed value if none provided."""

    return 1337 if seed is None else seed
