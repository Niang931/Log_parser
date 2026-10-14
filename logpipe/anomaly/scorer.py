"""Scoring windows of template ids with a trained LogBERT model.

A window is anomalous (following LogBERT's ``compute_anomaly``) when more
than ``miss_ratio`` of its masked keys are missing from the model's top-g
predictions, or, optionally, when its ``[SOS]`` embedding lies outside the
hypersphere radius.  On top of that, ``novelty_ratio``
(the share of templates the model has never seen) is a separate signal.
A never-seen event type is often the clearest anomaly on a device, and
masking it, as the original blueprint did, would hide it.

``unknown_mode`` exists for the evaluation ablation:

* ``unk``: unknown templates become ``[UNK]`` and are never scored (default).
* ``mask``: unknown templates become ``[MASK]`` and are never scored (the
  original blueprint).
* ``drop``: unknown templates are removed from the window.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import torch

from .logbert.dataset import collate, mask_sequence, predict_rng
from .logbert.model import LogBERT, LogBERTConfig
from .logbert.trainer import TrainResult, chunk
from .vocab import MASK, UNK, TemplateVocab

UnknownMode = Literal["unk", "mask", "drop"]


@dataclass
class Thresholds:
    """Defaults flag a window when over half its masked keys are missed, or when it
    holds any template the model has never seen (fleet-wide rollouts are
    suppressed by the detector).  Tune on labelled data to do better."""

    miss_ratio: float = 0.5
    novelty_ratio: float = 0.0
    use_distance: bool = False


@dataclass(frozen=True)
class WindowScore:
    length: int
    masked: int
    undetected: int
    unknown: int
    distance: float

    @property
    def miss_ratio(self) -> float:
        return self.undetected / self.masked if self.masked else 0.0

    @property
    def novelty_ratio(self) -> float:
        return self.unknown / self.length if self.length else 0.0


class Scorer(Protocol):
    """What the streaming detector needs from a window scorer."""

    version: str
    vocab: TemplateVocab

    def score(
        self, windows: Sequence[Sequence[int]], familiar: AbstractSet[int] = ...
    ) -> list[WindowScore]: ...

    def reasons(
        self, score: WindowScore, thresholds: Thresholds | None = None
    ) -> tuple[str, ...]: ...


@dataclass
class ScorerSettings:
    num_candidates: int = 6
    mask_ratio: float = 0.5
    unknown_mode: UnknownMode = "unk"
    batch_size: int = 128
    thresholds: Thresholds = field(default_factory=Thresholds)


class LogBERTScorer:
    def __init__(
        self,
        model: LogBERT,
        vocab: TemplateVocab,
        center: torch.Tensor,
        radius: float,
        settings: ScorerSettings | None = None,
        version: str = "logbert",
    ):
        self.model = model.eval()
        self.vocab = vocab
        self.center = center
        self.radius = radius
        self.settings = settings or ScorerSettings()
        self.version = version

    @classmethod
    def from_training(cls, result: TrainResult, vocab: TemplateVocab, **kwargs) -> LogBERTScorer:
        return cls(result.model, vocab, result.center, result.radius, **kwargs)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.model.state_dict(), "center": self.center},
            directory / "model.pt",
        )
        settings = asdict(self.settings)
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "model": self.model.config.to_dict(),
                    "vocab": self.vocab.to_json(),
                    "radius": self.radius,
                    "settings": settings,
                    "version": self.version,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> LogBERTScorer:
        meta = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        weights = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        model = LogBERT(LogBERTConfig(**meta["model"]))
        model.load_state_dict(weights["state_dict"])
        settings = dict(meta["settings"])
        settings["thresholds"] = Thresholds(**settings["thresholds"])
        return cls(
            model,
            TemplateVocab.from_json(meta["vocab"]),
            weights["center"],
            meta["radius"],
            ScorerSettings(**settings),
            meta.get("version", directory.name),
        )

    def _encode(self, window: Sequence[int], familiar: AbstractSet[int]) -> tuple[list[int], int]:
        encoded = self.vocab.encode(window)
        unknown = sum(1 for tid in window if tid not in self.vocab and tid not in familiar)
        # TODO: should make this en enum rather than literal
        if self.settings.unknown_mode == "drop":
            encoded = [tok for tok in encoded if tok != UNK]
        return encoded, unknown

    @torch.no_grad()
    def score(
        self, windows: Sequence[Sequence[int]], familiar: AbstractSet[int] = frozenset()
    ) -> list[WindowScore]:
        """Score windows of registry template ids (deterministic for a given window).

        ``familiar`` template ids are unknown to the model but should not count
        as novel, e.g. templates a firmware rollout introduced fleet-wide.
        """
        s = self.settings
        max_tokens = self.model.config.max_len - 1
        vocab_size = len(self.vocab)
        items: list[tuple[int, list[int], list[int]]] = []
        lengths, unknowns = [], []
        for w_index, window in enumerate(windows):
            encoded, unknown = self._encode(window, familiar)
            lengths.append(len(window))
            unknowns.append(unknown)

            # Masking the input
            for part in chunk(encoded, max_tokens):
                inputs, labels = mask_sequence(
                    sequence=part,
                    vocab_size=vocab_size,
                    mask_ratio=s.mask_ratio,
                    rng=predict_rng(part),
                    predict=True,
                    ensure_one=True,
                )

                # convert the UNK keys to MASK instead if mask mode enabled
                if s.unknown_mode == "mask":
                    inputs = [MASK if tok == UNK else tok for tok in inputs]
                items.append((w_index, inputs, labels))

        masked = [0] * len(windows)
        undetected = [0] * len(windows)
        distance = [0.0] * len(windows)

        # batch inference
        for start in range(0, len(items), s.batch_size):
            batch = items[start : start + s.batch_size]
            # collate pad the its inputs to the largest size within the elements
            x, y = collate([(inputs, labels) for _, inputs, labels in batch])

            # get the log probabilities and start of sequence (summary vector)
            logprobs, sos = self.model(x)

            # evaluate if true masked token appeared in top-k
            topk = logprobs.topk(min(s.num_candidates, vocab_size), dim=-1).indices
            hits = (topk == y.unsqueeze(-1)).any(dim=-1)

            # create a mask boolean tensor (not a simple boolean)
            scored = y > 0

            # euclidian distance for centroid distance calculation
            dists = ((sos - self.center) ** 2).sum(dim=1).sqrt()
            # including the window index here to ensure ordering even after batch shuffling
            for row, (w_index, _, _) in enumerate(batch):
                masked[w_index] += int(scored[row].sum())
                # take tokens that were suposed to be predict `scored[row]`
                # and find where the model failed to guess the right token `~hits[row]`
                undetected[w_index] += int((scored[row] & ~hits[row]).sum())
                distance[w_index] = max(distance[w_index], float(dists[row]))

        return [
            WindowScore(
                length=lengths[i],
                masked=masked[i],
                undetected=undetected[i],
                unknown=unknowns[i],
                distance=distance[i],
            )
            for i in range(len(windows))
        ]

    def reasons(self, score: WindowScore, thresholds: Thresholds | None = None) -> tuple[str, ...]:
        t = thresholds or self.settings.thresholds
        found = []
        if score.masked and score.miss_ratio > t.miss_ratio:
            found.append("sequence")
        if t.use_distance and score.distance > self.radius:
            found.append("hypersphere")
        if score.novelty_ratio > t.novelty_ratio:
            found.append("novelty")
        return tuple(found)


@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def to_dict(self) -> dict:
        return {
            "TP": self.tp,
            "FP": self.fp,
            "TN": self.tn,
            "FN": self.fn,
            "P": round(self.precision, 4),
            "R": round(self.recall, 4),
            "F1": round(self.f1, 4),
        }


def evaluate(
    scorer: LogBERTScorer,
    normal: Sequence[WindowScore],
    abnormal: Sequence[WindowScore],
    thresholds: Thresholds,
) -> Metrics:
    fp = sum(1 for s in normal if scorer.reasons(s, thresholds))
    tp = sum(1 for s in abnormal if scorer.reasons(s, thresholds))
    return Metrics(tp=tp, fp=fp, tn=len(normal) - fp, fn=len(abnormal) - tp)


def tune_thresholds(
    scorer: LogBERTScorer,
    normal: Sequence[WindowScore],
    abnormal: Sequence[WindowScore],
    *,
    use_novelty: bool = True,
    use_distance: tuple[bool, ...] = (False, True),
) -> tuple[Thresholds, Metrics]:
    """Grid search for the best F1 (LogBERT's ``find_best_threshold``, extended).

    Run this on a *validation* split.  The original tuned on the test set,
    which overstates F1.  The evaluation harness reports both numbers.
    In short, this thing is spamming different config to pray for best result.
    """
    novelty_grid = [0.0, 0.05, 0.1, 0.2, 0.3, float("inf")] if use_novelty else [float("inf")]
    best: tuple[Thresholds, Metrics] | None = None
    for miss in np.arange(0.0, 1.0, 0.05):
        for novelty in novelty_grid:
            for distance in use_distance:
                t = Thresholds(float(round(miss, 2)), novelty, distance)
                m = evaluate(scorer, normal, abnormal, t)
                if best is None or m.f1 > best[1].f1:
                    best = (t, m)
    assert best is not None
    return best
