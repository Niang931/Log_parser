import random
from typing import get_args

import pytest
import torch

from logpipe.anomaly.logbert import LogBERT, LogBERTConfig, TrainConfig, train_logbert
from logpipe.anomaly.logbert.dataset import collate, mask_sequence, predict_rng
from logpipe.anomaly.scorer import (
    LogBERTScorer,
    ScorerSettings,
    Thresholds,
    UnknownMode,
    evaluate,
    tune_thresholds,
)
from logpipe.anomaly.vocab import FIRST_KEY, MASK, SOS, UNK, TemplateVocab

CYCLE = [2, 3, 4, 5, 6, 7]


def test_vocab_maps_unknown_and_reserved_ids_to_unk():
    vocab = TemplateVocab.build([[2, 2, 3, 1, 0]])
    assert len(vocab) == FIRST_KEY + 2
    assert vocab.encode([2, 3, 99, 1]) == [FIRST_KEY, FIRST_KEY + 1, UNK, UNK]
    assert TemplateVocab.from_json(vocab.to_json()) == vocab


def test_masking_never_masks_unknowns_and_is_deterministic_for_prediction():
    seq = [5, UNK, 6, 7, UNK, 8]
    inputs, labels = mask_sequence(sequence=seq, vocab_size=10, mask_ratio=1.0, rng=random.Random(0), predict=True)
    assert inputs[0] == SOS
    assert [labels[i + 1] for i, tok in enumerate(seq) if tok == UNK] == [0, 0]
    assert all(inputs[i + 1] == MASK for i, tok in enumerate(seq) if tok != UNK)
    first = mask_sequence(sequence=seq, vocab_size=10, mask_ratio=0.5, rng=predict_rng(seq), predict=True)
    assert first == mask_sequence(sequence=seq, vocab_size=10, mask_ratio=0.5, rng=predict_rng(seq), predict=True)


def test_ensure_one_masks_short_windows():
    inputs, labels = mask_sequence(
        sequence=[5, 6], vocab_size=10, mask_ratio=0.0, rng=random.Random(0), predict=True,
        ensure_one=True,
    )
    assert sum(1 for x in labels if x) == 1 and MASK in inputs


def test_model_shapes_and_padding_mask():
    model = LogBERT(LogBERTConfig(vocab_size=12, hidden=16, layers=1, attn_heads=2, max_len=16))
    x, _ = collate([([SOS, 5, 6], [0, 5, 0]), ([SOS, 7], [0, 0])])
    logprobs, sos = model.eval()(x)
    assert logprobs.shape == (2, 3, 12) and sos.shape == (2, 16)
    assert torch.allclose(logprobs.exp().sum(-1), torch.ones(2, 3), atol=1e-5)


@pytest.fixture(scope="module")
def trained():
    rng = random.Random(0)
    normal = []
    for _ in range(300):
        start = rng.randrange(len(CYCLE))
        normal.append([CYCLE[(start + i) % len(CYCLE)] for i in range(12)])
    vocab = TemplateVocab.build(normal)
    encoded = [vocab.encode(s) for s in normal]
    result = train_logbert(
        encoded[30:], encoded[:30],
        LogBERTConfig(len(vocab), hidden=32, layers=1, attn_heads=2, max_len=16),
        TrainConfig(epochs=12, patience=12, batch_size=32, unk_ratio=0.0),
        log=lambda _: None,
    )
    return result, vocab


def _abnormal(rng):
    seq = [CYCLE[i % len(CYCLE)] for i in range(12)]
    rng.shuffle(seq)
    return seq


@pytest.mark.slow
def test_trained_model_separates_normal_from_shuffled_sequences(trained):
    result, vocab = trained
    scorer = LogBERTScorer.from_training(result, vocab, settings=ScorerSettings(num_candidates=1))
    rng = random.Random(1)
    normal = scorer.score([[CYCLE[(s + i) % 6] for i in range(12)] for s in range(6)] * 5)
    abnormal = scorer.score([_abnormal(rng) for _ in range(30)])
    assert sum(s.miss_ratio for s in normal) / 30 < sum(s.miss_ratio for s in abnormal) / 30
    thresholds, metrics = tune_thresholds(scorer, normal, abnormal, use_novelty=False)
    assert metrics.precision == 1.0 and metrics.f1 > 0.5
    assert evaluate(scorer, normal, abnormal, thresholds).f1 == metrics.f1


@pytest.mark.slow
def test_unknown_templates_drive_the_novelty_signal(trained):
    result, vocab = trained
    scorer = LogBERTScorer.from_training(result, vocab)
    window = [2, 3, 4, 5, 99, 7, 2, 3, 4, 5, 6, 7]
    (score,) = scorer.score([window])
    assert score.unknown == 1 and score.novelty_ratio == pytest.approx(1 / 12)
    assert "novelty" in scorer.reasons(score, Thresholds(miss_ratio=1.0, novelty_ratio=0.0))
    (familiar,) = scorer.score([window], familiar=frozenset({99}))
    assert familiar.unknown == 0


@pytest.mark.slow
def test_unknown_modes_change_the_model_input_not_the_novelty_count(trained):
    result, vocab = trained
    window = [2, 3, 99, 5, 6, 7, 2, 3, 99, 5, 6, 7]
    counts = {}
    for mode in get_args(UnknownMode):
        scorer = LogBERTScorer.from_training(result, vocab, settings=ScorerSettings(unknown_mode=mode))
        (score,) = scorer.score([window])
        counts[mode] = (score.unknown, score.length)
    assert set(counts.values()) == {(2, 12)}


@pytest.mark.slow
def test_save_load_round_trip_gives_identical_scores(trained, tmp_path):
    result, vocab = trained
    scorer = LogBERTScorer.from_training(
        result, vocab, settings=ScorerSettings(thresholds=Thresholds(0.3, 0.1, True)), version="v1"
    )
    scorer.save(tmp_path / "m")
    loaded = LogBERTScorer.load(tmp_path / "m")
    windows = [[2, 3, 4, 5, 6, 7] * 2, [7, 6, 5, 4, 3, 2] * 2]
    assert loaded.score(windows) == scorer.score(windows)
    assert loaded.settings.thresholds == Thresholds(0.3, 0.1, True)
    assert loaded.version == "v1"
