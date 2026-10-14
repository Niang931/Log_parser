"""LogBERT evaluation on labelled sessions (HDFS blocks, BGL windows).

The protocol mirrors production rather than the original repo:

1. The registry is calibrated **only on the normal training sessions**, the
   way it would be on device history.  Templates first seen at test time
   therefore come out of the frozen parser as ``UNK``, just as a PENDING
   line would at ingest.
2. LogBERT trains on the normal training sessions (10% held out for early
   stopping).
3. Thresholds are tuned on a separate validation split (normal +
   abnormal) and applied to the test split.  The original tunes on the test
   set, so that "oracle" F1 is reported alongside for comparability.
4. Ablations: unknown handling (``unk`` / ``mask`` / ``drop``) x novelty
   signal (on / off).
"""
from __future__ import annotations

import json
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import get_args

import numpy as np

from ..anomaly.logbert import LogBERTConfig, TrainConfig, train_logbert
from ..anomaly.scorer import (
    LogBERTScorer,
    ScorerSettings,
    Thresholds,
    UnknownMode,
    evaluate,
    tune_thresholds,
)
from ..anomaly.vocab import TemplateVocab
from ..calibrate import calibrate
from ..enrich.llm import LLMAdapter
from ..parsing.frozen_parser import FrozenParser
from ..progress import progress, write
from .datasets import Sessions


@dataclass
class LogBERTEvalConfig:
    hidden: int = 64
    layers: int = 2
    attn_heads: int = 4
    max_len: int = 128
    train: TrainConfig = field(default_factory=TrainConfig)
    num_candidates: int = 6
    mask_ratio: float = 0.5
    min_len: int = 10
    train_share: float = 0.4
    tune_normal_share: float = 0.1
    tune_abnormal_share: float = 0.2
    seed: int = 42
    unknown_modes: tuple[UnknownMode, ...] = get_args(UnknownMode)


def _split(keys: list[str], shares: Sequence[float], rng: random.Random) -> list[list[str]]:
    keys = sorted(keys)
    rng.shuffle(keys)
    out, start = [], 0
    for share in shares:
        n = int(len(keys) * share)
        out.append(keys[start : start + n])
        start += n
    out.append(keys[start:])
    return out


def parse_sessions(parser: FrozenParser, sessions: Sessions, keys: Sequence[str],
                   latency: list[int] | None = None,
                   memo: dict[str, int] | None = None) -> dict[str, list[int]]:
    """Template ids per session.  ``memo`` caches by line (BGL windows overlap 5x)."""
    memo = {} if memo is None else memo
    seqs: dict[str, list[int]] = {}
    for key in progress(keys, "parsing sessions", unit="session"):
        ids = []
        for line in sessions.lines[key]:
            tid = memo.get(line)
            if tid is None:
                start = time.perf_counter_ns()
                tid = memo[line] = int(parser.parse(line).template_id)
                if latency is not None and len(latency) < 300_000:
                    latency.append(time.perf_counter_ns() - start)
            ids.append(tid)
        seqs[key] = ids
    return seqs


def run_logbert_eval(
    sessions: Sessions,
    adapter: LLMAdapter,
    cfg: LogBERTEvalConfig | None = None,
    out_dir: Path | None = None,
    log=write,
) -> dict:
    cfg = cfg or LogBERTEvalConfig()
    rng = random.Random(cfg.seed)
    train_k, tune_n, test_n = _split(sessions.normal, [cfg.train_share, cfg.tune_normal_share], rng)
    tune_a, test_a = _split(sessions.abnormal, [cfg.tune_abnormal_share], rng)
    log(f"[{sessions.name}] sessions: train={len(train_k)} tune={len(tune_n)}+{len(tune_a)} "
        f"test={len(test_n)}+{len(test_a)}")

    started = time.perf_counter()
    train_lines = [line for k in train_k for line in sessions.lines[k]]
    cal = calibrate(train_lines, adapter)
    log(f"[{sessions.name}] calibrated on {len(train_lines)} lines: {cal.report.templates} templates, "
        f"holdout coverage {cal.report.holdout_coverage:.3f} ({time.perf_counter() - started:.0f}s)")
    parser = FrozenParser(cal.snapshot)

    latency: list[int] = []
    memo: dict[str, int] = {}
    groups = {"train": train_k, "tune_n": tune_n, "tune_a": tune_a, "test_n": test_n, "test_a": test_a}
    seqs = {}
    for name, keys in groups.items():
        parsed = parse_sessions(parser, sessions, keys, latency, memo)
        seqs[name] = [s for s in parsed.values() if len(s) >= cfg.min_len]
    unknown_share = {
        name: round(float(np.mean([np.mean([t == 1 for t in s]) for s in v])), 4) if v else 0.0
        for name, v in seqs.items()
    }
    log(f"[{sessions.name}] after min_len={cfg.min_len}: " +
        ", ".join(f"{k}={len(v)}" for k, v in seqs.items()) + f"; unknown share {unknown_share}")

    vocab = TemplateVocab.build(seqs["train"])
    encoded = [vocab.encode(s) for s in seqs["train"]]
    rng.shuffle(encoded)
    n_valid = max(1, len(encoded) // 10)
    model_cfg = LogBERTConfig(len(vocab), cfg.hidden, cfg.layers, cfg.attn_heads, cfg.max_len)
    started = time.perf_counter()
    trained = train_logbert(encoded[n_valid:], encoded[:n_valid], model_cfg, cfg.train,
                            log=lambda m: log(f"[{sessions.name}] {m}"))
    train_seconds = time.perf_counter() - started

    results = []
    scorer = LogBERTScorer.from_training(
        trained, vocab,
        settings=ScorerSettings(num_candidates=cfg.num_candidates, mask_ratio=cfg.mask_ratio),
        version=f"logbert-{sessions.name.lower()}",
    )
    for mode in progress(cfg.unknown_modes, "unknown-mode ablations", unit="mode"):
        scorer.settings.unknown_mode = mode
        scores = {name: scorer.score(seqs[name]) for name in ("tune_n", "tune_a", "test_n", "test_a")}
        for novelty in (True, False):
            thresholds, tune_metrics = tune_thresholds(
                scorer, scores["tune_n"], scores["tune_a"], use_novelty=novelty
            )
            test_metrics = evaluate(scorer, scores["test_n"], scores["test_a"], thresholds)
            _, oracle = tune_thresholds(scorer, scores["test_n"], scores["test_a"], use_novelty=novelty)
            row = {
                "dataset": sessions.name, "unknown_mode": mode, "novelty": novelty,
                "thresholds": asdict(thresholds), "validation": tune_metrics.to_dict(),
                "test": test_metrics.to_dict(), "oracle_test": oracle.to_dict(),
            }
            results.append(row)
            log(f"[{sessions.name}] mode={mode} novelty={novelty}: test F1={test_metrics.f1:.4f} "
                f"(P={test_metrics.precision:.3f} R={test_metrics.recall:.3f}) oracle F1={oracle.f1:.4f}")

    scorer.settings.unknown_mode = "unk"
    best = max((r for r in results if r["unknown_mode"] == "unk"), key=lambda r: r["validation"]["F1"])
    scorer.settings.thresholds = Thresholds(**best["thresholds"])
    us = np.asarray(latency) / 1000 if latency else np.zeros(1)
    summary = {
        "dataset": sessions.name,
        "sessions": {k: len(v) for k, v in groups.items()},
        "sequences_after_min_len": {k: len(v) for k, v in seqs.items()},
        "unknown_share": unknown_share,
        "calibration": cal.report.to_json() | {"tuning": []},
        "vocab_size": len(vocab),
        "model": model_cfg.to_dict(),
        "train": {"seconds": round(train_seconds, 1), "epochs": len(trained.history),
                  "history": trained.history, "radius": trained.radius},
        "parse_latency_us": {f"p{q}": round(float(np.percentile(us, q)), 2) for q in (50, 90, 99)},
        "results": results,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        scorer.save(out_dir / "model")
        (out_dir / "results.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
