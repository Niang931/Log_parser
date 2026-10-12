"""Masking and batching of encoded log-key sequences.

Adapted from ``bert_pytorch/dataset/log_dataset.py`` in
https://github.com/HelenGuohx/logbert (MIT, Copyright (c) 2023 Helennnnn).

Changes from the original:
* ``[UNK]`` positions are never masked.  They carry no ground-truth key, so
  they cannot be scored.  The original masks them and labels them
  ``unk_index``, which rewards predicting "unknown".
* Training replaces a small fraction (``unk_ratio``) of unmasked inputs with
  ``[UNK]``, so the model sees unknown templates in context, as it will once
  the registry outgrows the vocabulary.
* For the 10% "random token" replacement, only real log keys are drawn,
  never special tokens.
* Prediction-time masking is deterministic: a per-sequence RNG is seeded
  from the sequence content.  The original used the global RNG, so two
  runs over the same window could disagree.
* The time-interval channel is dropped (``is_time=False`` in every config).
"""
from __future__ import annotations

import random
import zlib
from collections.abc import Sequence

import torch

from ..vocab import FIRST_KEY, MASK, PAD, SOS, UNK


def mask_sequence(
    seq: Sequence[int],
    vocab_size: int,
    mask_ratio: float,
    rng: random.Random,
    *,
    predict: bool,
    unk_ratio: float = 0.0,
    ensure_one: bool = False,
) -> tuple[list[int], list[int]]:
    """Return ``(inputs, labels)`` with ``[SOS]`` prepended. Label 0 means "not scored".

    ``ensure_one`` masks one known position when chance masked none, so
    short windows still get scored.
    """
    inputs, labels = [SOS], [PAD]
    for token in seq:

        # If UNK then we dont even consider masking the token
        if token == UNK:
            inputs.append(UNK)
            labels.append(PAD)
            continue

        prob = rng.random()
        if prob < mask_ratio:
            labels.append(token)
            if predict:
                inputs.append(MASK)
                continue
            prob /= mask_ratio

            # TODO: make these consts instead
            # Deliberately not replacing with MASK sometime so
            # inference mismatch impact is less noticable (80/10/10)
            if prob < 0.8:
                inputs.append(MASK)
            elif prob < 0.9 and vocab_size > FIRST_KEY:
                inputs.append(rng.randrange(FIRST_KEY, vocab_size))
            else:
                inputs.append(token)
        else:
            labels.append(PAD)
            inputs.append(UNK if unk_ratio and rng.random() < unk_ratio else token)

    # Ensure at least one token is masked (optional)
    if ensure_one and not any(labels):
        known = [i for i, tok in enumerate(seq, start=1) if tok != UNK]
        if known:
            pos = rng.choice(known)
            labels[pos], inputs[pos] = inputs[pos], MASK

    return inputs, labels


def predict_rng(seq: Sequence[int], seed: int = 1234) -> random.Random:
    return random.Random(zlib.crc32(bytes(str(list(seq)), "ascii")) ^ seed)


def collate(pairs: Sequence[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad to the longest sequence in the batch (the original's dynamic padding)."""
    length = max(len(inputs) for inputs, _ in pairs)
    x = torch.full((len(pairs), length), PAD, dtype=torch.long)
    y = torch.full((len(pairs), length), PAD, dtype=torch.long)
    for row, (inputs, labels) in enumerate(pairs):
        x[row, : len(inputs)] = torch.tensor(inputs, dtype=torch.long)
        y[row, : len(labels)] = torch.tensor(labels, dtype=torch.long)
    return x, y
