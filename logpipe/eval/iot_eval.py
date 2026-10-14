"""End-to-end evaluation on the synthetic IoT fleet.

1. Calibrate registry v1 on the normal history.
2. Train LogBERT on per-device sliding windows of the history.
3. Replay the validation stream (anomalies, no rollout) through the
   detector and tune thresholds on its windows.
4. Replay the test stream (anomalies + firmware rollout) through the full
   pipeline (frozen parser, store, enrichment loop, detector) and score
   the windows against the injected labels.

Each test variant is an ablation of one design decision from the review:

* ``full``: [UNK] token, novelty signal, fleet suppression, enrichment.
* ``no_fleet_suppression``: rollout templates count as novel on every device.
* ``mask_unknowns``: the original blueprint's [MASK] for unknown logs, no novelty.
* ``no_enrichment``: unknown lines stay PENDING (no closing of the loop).
"""
from __future__ import annotations

import bisect
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from ..anomaly.detector import DetectorConfig, ScoredWindow, sliding_windows
from ..anomaly.logbert import LogBERTConfig, TrainConfig, train_logbert
from ..anomaly.scorer import (
    LogBERTScorer,
    Metrics,
    ScorerSettings,
    Thresholds,
    UnknownMode,
    tune_thresholds,
)
from ..anomaly.vocab import TemplateVocab
from ..calibrate import calibrate
from ..enrich.llm import LLMAdapter
from ..parsing.frozen_parser import FrozenParser
from ..pipeline import Pipeline
from ..registry import SchemaRegistry
from ..progress import progress, write
from ..storage.memory import MemoryStore
from .synth_iot import SynthEvent, as_log_events, generate


@dataclass
class IoTEvalConfig:
    devices_per_kind: int = 5
    history_hours: float = 12
    validation_hours: float = 3
    test_hours: float = 6
    anomaly_rate: float = 0.01
    seed: int = 7
    window: int = 20
    stride: int = 5
    enrich_every: int = 2_000
    hidden: int = 64
    layers: int = 2
    attn_heads: int = 4
    candidate_grid: tuple[int, ...] = (1, 2, 3, 6)
    train: TrainConfig | None = None


@dataclass(frozen=True)
class Variant:
    name: str
    unknown_mode: UnknownMode = "unk"
    novelty: bool = True
    fleet_min_devices: int = 2
    enrichment: bool = True


VARIANTS = (
    Variant("full"),
    Variant("no_fleet_suppression", fleet_min_devices=0),
    Variant("mask_unknowns", unknown_mode="mask", novelty=False),
    Variant("no_enrichment", enrichment=False),
)


class _Labels:
    """Window labels from injected anomalies: (device, timestamp) → kind."""

    def __init__(self, events: list[SynthEvent]):
        self.times: dict[str, list] = defaultdict(list)
        self.kinds: dict[str, list[str]] = defaultdict(list)
        for e in events:
            if e.anomaly:
                self.times[e.device_id].append(e.timestamp)
                self.kinds[e.device_id].append(e.anomaly)

    def kinds_in(self, window: ScoredWindow) -> set[str]:
        """Returns anomolies window in in proviided time range."""
        times = self.times.get(window.device_id, [])
        # binary search to find low, high given time window
        lo = bisect.bisect_left(times, window.start)
        hi = bisect.bisect_right(times, window.end)
        return set(self.kinds[window.device_id][lo:hi])


def _metrics(windows: list[ScoredWindow], labels: _Labels, vocab: TemplateVocab | None = None,
             registry: SchemaRegistry | None = None) -> dict:
    tp = fp = tn = fn = 0
    by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_reason: Counter[str] = Counter()
    fp_reasons: Counter[str] = Counter()
    fp_unknown: Counter[int] = Counter()
    for w in windows:
        kinds, flagged = labels.kinds_in(w), bool(w.reasons)
        by_reason.update(w.reasons)
        # true positive: anomoly exists and detector flag it
        if kinds and flagged:
            tp += 1
        # false negative: anomoly exists and detector missed it
        elif kinds:
            fn += 1
        # false positive: no anomoly, detector falsely flagged it
        elif flagged:
            fp += 1
            fp_reasons.update(w.reasons)
            if vocab is not None:
                fp_unknown.update({t for t in w.template_ids if t not in vocab})
        # true negative: no anomoly, detector correctly shut up
        else:
            tn += 1

        # track total occureneces and caught occurences for each anomoly type
        for kind in kinds:
            by_kind[kind][0] += 1
            by_kind[kind][1] += int(flagged)

    m = Metrics(tp, fp, tn, fn)
    return m.to_dict() | {
        "windows": len(windows),
        "recall_by_kind": {k: round(hit / n, 4) for k, (n, hit) in sorted(by_kind.items())},
        "flags_by_reason": dict(by_reason),
        "false_positive_reasons": dict(fp_reasons),
        "false_positive_unknown_templates": {
            _template_text(registry, tid): n for tid, n in fp_unknown.most_common(8)
        },
    }


def _template_text(registry: SchemaRegistry | None, tid: int) -> str:
    entry = registry.current.templates.get(tid) if registry is not None else None
    return " ".join(entry.tokens[2:]) if entry else f"#{tid} (pending)"


def _device_sequences(parser: FrozenParser, events: list[SynthEvent]) -> dict[str, list[int]]:
    seqs: dict[str, list[int]] = defaultdict(list)
    for e in events:
        seqs[e.device_id].append(int(parser.parse(e.line).template_id))
    return seqs


def _detector_config(variant: Variant, cfg: IoTEvalConfig) -> DetectorConfig:
    return DetectorConfig(window=cfg.window, stride=cfg.stride,
                          fleet_min_devices=variant.fleet_min_devices, record_windows=True)


def run_iot_eval(adapter: LLMAdapter, cfg: IoTEvalConfig | None = None,
                 out_dir: Path | None = None, log=write) -> dict:
    cfg = cfg or IoTEvalConfig()
    fleet = generate(cfg.devices_per_kind, cfg.history_hours, cfg.validation_hours,
                     cfg.test_hours, cfg.anomaly_rate, seed=cfg.seed)
    if out_dir is not None:
        fleet.write(out_dir / "fleet")
    log(f"[iot] events: history={len(fleet.history)} validation={len(fleet.validation)} "
        f"test={len(fleet.test)} (anomalous test events: {sum(1 for e in fleet.test if e.anomaly)})")

    cal = calibrate([e.line for e in fleet.history], adapter)
    log(f"[iot] calibration: {cal.report.templates} templates, "
        f"holdout coverage {cal.report.holdout_coverage}")
    parser = FrozenParser(cal.snapshot)

    windows = [w for seq in _device_sequences(parser, fleet.history).values()
               for w in sliding_windows(seq, cfg.window, cfg.stride)]
    vocab = TemplateVocab.build(windows)
    encoded = [vocab.encode(w) for w in windows]
    random.Random(cfg.seed).shuffle(encoded)
    n_valid = max(1, len(encoded) // 10)
    trained = train_logbert(
        encoded[n_valid:], encoded[:n_valid],
        LogBERTConfig(len(vocab), cfg.hidden, cfg.layers, cfg.attn_heads, max_len=cfg.window + 1),
        cfg.train or TrainConfig(epochs=40, patience=6),
        log=lambda m: log(f"[iot] {m}"),
    )
    validation_labels, test_labels = _Labels(fleet.validation), _Labels(fleet.test)

    results = {}
    for variant in progress(VARIANTS, "iot variants", unit="variant"):
        scorer = LogBERTScorer.from_training(
            trained, vocab, settings=ScorerSettings(unknown_mode=variant.unknown_mode),
            version="logbert-iot",
        )
        # Tune top-g and thresholds on validation windows (labels from the injected
        # anomalies).  g matters: with a ~20-template vocabulary, LogBERT's HDFS
        # setting g=6 accepts almost any key and the sequence signal goes quiet.
        best = None
        for g in progress(cfg.candidate_grid, "tuning top-g", unit="g"):
            scorer.settings.num_candidates = g
            # Detector only: enrichment here would mutate the registry between g values.
            val = Pipeline.assemble(SchemaRegistry(cal.snapshot), MemoryStore(), scorer=scorer,
                                    detector_config=_detector_config(variant, cfg))
            val.run(as_log_events(fleet.validation), total=len(fleet.validation))
            assert val.detector is not None
            val_windows = val.detector.scored
            normal = [w.score for w in val_windows if not validation_labels.kinds_in(w)]
            abnormal = [w.score for w in val_windows if validation_labels.kinds_in(w)]
            t, m = tune_thresholds(scorer, normal, abnormal, use_novelty=variant.novelty)
            if best is None or m.f1 > best[2].f1:
                best = (g, t, m)
        assert best is not None
        scorer.settings.num_candidates, thresholds, val_metrics = best
        scorer.settings.thresholds = thresholds

        registry = SchemaRegistry(cal.snapshot)
        store = MemoryStore()
        pipeline = Pipeline.assemble(
            registry, store, adapter=adapter if variant.enrichment else None, scorer=scorer,
            detector_config=_detector_config(variant, cfg), enrich_every=cfg.enrich_every,
        )
        handle, detector = pipeline.handle, pipeline.detector
        assert detector is not None
        stats = pipeline.run(as_log_events(fleet.test), total=len(fleet.test))
        lags = [lag for c in stats.cycles for lag in c.resolve_lag_sec]
        counts = store.count_by_status()
        results[variant.name] = {
            "variant": asdict(variant),
            "thresholds": asdict(thresholds),
            "num_candidates": scorer.settings.num_candidates,
            "validation": val_metrics.to_dict(),
            "test": _metrics(detector.scored, test_labels, vocab, registry),
            "pipeline": stats.summary(),
            "enrichment": {
                "cycles": len(stats.cycles),
                "final_rows": counts.get("FINAL", 0),
                "pending_rows_at_end": counts.get("PENDING", 0),
                "pending_share_at_end": round(counts.get("PENDING", 0) / max(1, sum(counts.values())), 5),
                "resolve_lag_s": (
                    {f"p{q}": round(float(np.percentile(lags, q)), 1) for q in (50, 90, 99)}
                    if lags else {}
                ),
                "registry_version": handle.version,
                "rejected": {k: v for c in stats.cycles for k, v in c.rejected.items()},
                "learned_templates": [
                    " ".join(registry.current.templates[i].tokens)
                    for c in stats.cycles for i in c.published
                ],
                "aliases": {str(k): v for c in stats.cycles for k, v in c.aliased.items()},
            },
        }
        t = results[variant.name]["test"]
        log(f"[iot] {variant.name} (g={scorer.settings.num_candidates}): test F1={t['F1']} "
            f"P={t['P']} R={t['R']} flags={t['flags_by_reason']} "
            f"FP reasons={t['false_positive_reasons']} | pending at end="
            f"{results[variant.name]['enrichment']['pending_rows_at_end']}")

    summary = {
        "config": asdict(replace(cfg, train=None)) | {"train": asdict(cfg.train or TrainConfig(epochs=40, patience=6))},
        "events": {"history": len(fleet.history), "validation": len(fleet.validation),
                   "test": len(fleet.test)},
        "calibration": cal.report.to_json(),
        "vocab_size": len(vocab),
        "train": {"epochs": len(trained.history), "history": trained.history},
        "variants": results,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        tuned = LogBERTScorer.from_training(trained, vocab, settings=ScorerSettings(
            num_candidates=results["full"]["num_candidates"],
            thresholds=Thresholds(**results["full"]["thresholds"])), version="logbert-iot")
        tuned.save(out_dir / "model")
    return summary
