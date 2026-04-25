"""Dataclasses for regex mask specifications."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Mask:
    label: str
    pattern: str
    justification: str

    def to_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "pattern": self.pattern,
            "justification": self.justification,
        }

    @staticmethod
    def from_dict(payload: dict[str, str]) -> "Mask":
        """Construct a :class:`Mask` from a mapping.

        The helper is used by the lightweight public API so that callers can
        provide plain dictionaries (for example when loading JSON produced by
        ``synth_masks``) without being aware of the dataclass structure.
        """

        return Mask(
            label=payload["label"],
            pattern=payload["pattern"],
            justification=payload.get("justification", ""),
        )


@dataclass
class MaskBundle:
    dataset: str
    masks: List[Mask]

    def to_json(self) -> List[dict[str, str]]:
        return [mask.to_dict() for mask in self.masks]
