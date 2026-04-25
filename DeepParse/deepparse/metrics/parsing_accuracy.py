"""Parsing accuracy metric."""
from __future__ import annotations

from typing import Sequence


def parsing_accuracy(true_templates: Sequence[str], predicted_templates: Sequence[str]) -> float:
    if len(true_templates) != len(predicted_templates):
        raise ValueError("Mismatched lengths for PA computation")
    if not true_templates:
        return 0.0
    correct = sum(1 for true, pred in zip(true_templates, predicted_templates) if true == pred)
    return correct / len(true_templates)
