"""Template-id vocabulary for LogBERT.

The registry owns template ids. This maps them to dense embedding indices
behind LogBERT's special tokens.  Template ids not seen in training
(new templates, or ones learned after the model was trained) map to
``[UNK]``.  ``[UNK]`` is a real token that appears during training, so it
is not ``[MASK]``, which the model is trained to *predict*.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..registry import UniqueTemplateID

# TODO: maybe just put this into an enum which hold 3 separate values
PAD, UNK, EOS, SOS, MASK = 0, 1, 2, 3, 4
SPECIALS = ("[PAD]", "[UNK]", "[EOS]", "[SOS]", "[MASK]")
FIRST_KEY = len(SPECIALS)


@dataclass(frozen=True)
class TemplateVocab:
    template_ids: tuple[int, ...]
    _index: dict[int, int] = field(init=False, repr=False, compare=False)

    @classmethod
    def build(cls, sequences: Iterable[Sequence[int]], min_freq: int = 1) -> TemplateVocab:
        # Reserved registry ids (unset, UNK) are never real log keys.
        counts = Counter(
            tid for seq in sequences for tid in seq if tid >= UniqueTemplateID.FIRST_TEMPLATE_ID
        )
        # Frequency order like the original WordVocab, ties broken by id for determinism.
        kept = (tid for tid, c in counts.items() if c >= min_freq)
        return cls(tuple(sorted(kept, key=lambda t: (-counts[t], t))))

    def __post_init__(self) -> None:
        index = {tid: i + FIRST_KEY for i, tid in enumerate(self.template_ids)}
        object.__setattr__(self, "_index", index)

    def __len__(self) -> int:
        return FIRST_KEY + len(self.template_ids)

    def __contains__(self, template_id: int) -> bool:
        return template_id in self._index

    def index(self, template_id: int) -> int:
        return self._index.get(template_id, UNK)

    def encode(self, sequence: Sequence[int]) -> list[int]:
        return [self._index.get(tid, UNK) for tid in sequence]

    def to_json(self) -> list[int]:
        return list(self.template_ids)

    @classmethod
    def from_json(cls, payload: Sequence[int]) -> TemplateVocab:
        return cls(tuple(int(t) for t in payload))
