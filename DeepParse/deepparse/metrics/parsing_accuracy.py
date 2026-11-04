"""Parsing accuracy metric."""
from __future__ import annotations

from collections.abc import Sequence

from deepparse.metrics.template_rules import normalise_template


def parsing_accuracy(
    true_templates: Sequence[str], predicted_templates: Sequence[str], *, normalise: bool = True
) -> float:
    """Share of lines whose template matches exactly; ``normalise`` applies the LogHub
    template rules (:mod:`deepparse.metrics.template_rules`) to both sides first."""
    if len(true_templates) != len(predicted_templates):
        raise ValueError("Mismatched lengths for PA computation")
    if not true_templates:
        return 0.0
    if normalise:
        cache: dict[str, str] = {}

        def norm(t: str) -> str:
            if t not in cache:
                cache[t] = normalise_template(t)
            return cache[t]
    else:
        def norm(t: str) -> str:
            return t
    correct = sum(
        1 for true, pred in zip(true_templates, predicted_templates) if norm(true) == norm(pred)
    )
    return correct / len(true_templates)
