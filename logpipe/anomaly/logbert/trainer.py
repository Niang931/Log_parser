"""LogBERT training: masked log-key prediction plus the hypersphere objective.

Adapted from ``bert_pytorch/trainer/pretrain.py`` and ``bert_pytorch/train_log.py``
in https://github.com/HelenGuohx/logbert (MIT, Copyright (c) 2023 Helennnnn).

Kept from the original:
* The loss: NLL on masked keys (``ignore_index=0``) + 0.1 x MSE of the
  ``[SOS]`` embedding to the hypersphere centre.
* The centre: the mean ``[SOS]`` embedding over train + valid, recomputed
  every epoch.
* The radius: the (1 - nu) quantile of distances, with nu = 0.25.
* Early stopping on validation loss.

Changes:
* Adam without the warm-up scheduler.  On the CPU-sized configs used here,
  warm-up made no measurable difference and doubled the epoch count.
* The best weights are kept in memory and returned.  The caller saves
  ``state_dict`` + config, not a pickled model (see :mod:`..scorer`).
* There is no pandas/seaborn loss logging; a list of per-epoch losses is
  returned instead.
* Sequences longer than ``max_len - 1`` are split into chunks rather than
  truncated, so no events are silently dropped from training.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .dataset import collate, mask_sequence
from .model import LogBERT, LogBERTConfig


@dataclass
class TrainConfig:
    mask_ratio: float = 0.5
    unk_ratio: float = 0.02
    epochs: int = 30
    patience: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    hypersphere_weight: float = 0.1
    nu: float = 0.25
    seed: int = 1234
    threads: int | None = None


@dataclass
class TrainResult:
    model: LogBERT
    center: torch.Tensor
    radius: float
    history: list[dict] = field(default_factory=list)


def chunk(seq: Sequence[int], max_tokens: int) -> list[list[int]]:
    return [list(seq[i : i + max_tokens]) for i in range(0, len(seq), max_tokens)] or [[]]


@torch.no_grad()
def _sos_embeddings(
    model: LogBERT, sequences: Sequence[Sequence[int]], batch_size: int
) -> torch.Tensor:
    model.eval()
    outputs = []
    for start in range(0, len(sequences), batch_size):
        batch = [
            mask_sequence(sequence=s, vocab_size=0, mask_ratio=0.0, rng=random.Random(0), predict=True)
            for s in sequences[start : start + batch_size]
        ]
        x, _ = collate(batch)
        outputs.append(model(x)[1])
    return torch.cat(outputs) if outputs else torch.zeros(0, model.config.hidden)


def _epoch(
    model: LogBERT,
    sequences: Sequence[Sequence[int]],
    center: torch.Tensor,
    cfg: TrainConfig,
    rng: random.Random,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, list[float]]:
    training = optimizer is not None
    model.train(training)
    order = list(range(len(sequences)))
    if training:
        rng.shuffle(order)
    nll = nn.NLLLoss(ignore_index=0)
    total, batches, dists = 0.0, 0, []
    vocab_size = model.config.vocab_size
    for start in range(0, len(order), cfg.batch_size):
        batch = [
            mask_sequence(
                sequence=sequences[i],
                vocab_size=vocab_size,
                mask_ratio=cfg.mask_ratio,
                rng=rng,
                predict=False,
                unk_ratio=cfg.unk_ratio if training else 0.0,
            )
            for i in order[start : start + cfg.batch_size]
        ]
        x, y = collate(batch)
        with torch.set_grad_enabled(training):
            logprobs, sos = model(x)
            loss = nll(logprobs.transpose(1, 2), y) if (y > 0).any() else logprobs.sum() * 0
            dist = ((sos - center) ** 2).sum(dim=1)
            loss = loss + cfg.hypersphere_weight * dist.mean()
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += loss.item()
        batches += 1
        dists.extend(dist.detach().tolist())
    return total / max(batches, 1), dists


def train_logbert(
    train_seqs: Sequence[Sequence[int]],
    valid_seqs: Sequence[Sequence[int]],
    model_config: LogBERTConfig,
    cfg: TrainConfig | None = None,
    log: Callable[[str], None] = print,
) -> TrainResult:
    """Train on vocab-encoded sequences of *normal* behaviour."""
    cfg = cfg or TrainConfig()
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    max_tokens = model_config.max_len - 1
    train = [c for s in train_seqs for c in chunk(s, max_tokens) if c]
    valid = [c for s in valid_seqs for c in chunk(s, max_tokens) if c] or train[
        : max(1, len(train) // 10)
    ]
    if not train:
        raise ValueError("no training sequences")

    model = LogBERT(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    best_loss, best_state, best_center, best_radius, stale = float("inf"), None, None, 0.0, 0
    history: list[dict] = []
    for epoch in range(cfg.epochs):
        center = _sos_embeddings(model, train + valid, cfg.batch_size * 4).mean(dim=0)
        train_loss, train_dist = _epoch(model, train, center, cfg, rng, optimizer)
        valid_loss, valid_dist = _epoch(model, valid, center, cfg, rng, None)
        radius = float(np.quantile(np.sqrt(train_dist + valid_dist), 1 - cfg.nu))
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "radius": radius}
        )
        log(f"epoch {epoch}: train={train_loss:.4f} valid={valid_loss:.4f} radius={radius:.4f}")
        if valid_loss < best_loss:
            best_loss, stale = valid_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_center, best_radius = center.clone(), radius
        else:
            stale += 1
            if stale >= cfg.patience:
                log(f"early stop at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model=model, center=best_center, radius=best_radius, history=history)
