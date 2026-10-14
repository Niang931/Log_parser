from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from datetime import UTC, datetime, timedelta

from logpipe.anomaly.detector import DetectorConfig, StreamingDetector, sliding_windows
from logpipe.anomaly.scorer import Thresholds, WindowScore
from logpipe.anomaly.vocab import TemplateVocab

T0 = datetime(2026, 4, 13, tzinfo=UTC)


class FakeScorer:
    """Flags windows containing template 66 as a sequence anomaly; counts unknowns."""

    version = "fake"

    def __init__(self):
        self.vocab = TemplateVocab((2, 3, 4, 66))
        self.calls: list[tuple[tuple[int, ...], frozenset[int]]] = []

    def score(
        self, windows: Sequence[Sequence[int]], familiar: AbstractSet[int] = frozenset()
    ) -> list[WindowScore]:
        out = []
        for w in windows:
            self.calls.append((tuple(w), frozenset(familiar)))
            unknown = sum(1 for t in w if t not in self.vocab and t not in familiar)
            out.append(WindowScore(len(w), 1, int(66 in w), unknown, 0.0))
        return out

    def reasons(self, score: WindowScore, thresholds: Thresholds | None = None) -> tuple[str, ...]:
        t = thresholds or Thresholds(miss_ratio=0.5, novelty_ratio=0.0)
        reasons = []
        if score.miss_ratio > t.miss_ratio:
            reasons.append("sequence")
        if score.novelty_ratio > t.novelty_ratio:
            reasons.append("novelty")
        return tuple(reasons)


def _detector(scorer: FakeScorer | None = None, **cfg) -> StreamingDetector:
    config = DetectorConfig(window=4, stride=2, min_len=2, batch_size=1,
                            lateness=timedelta(0), **cfg)
    return StreamingDetector(scorer or FakeScorer(), config)


def test_sliding_windows_cover_the_tail():
    assert sliding_windows([1, 2, 3, 4, 5], 3, 2) == [[1, 2, 3], [3, 4, 5]]
    assert sliding_windows([1, 2, 3, 4, 5, 6], 3, 2) == [[1, 2, 3], [3, 4, 5], [4, 5, 6]]
    assert sliding_windows([1, 2], 3, 2) == [[1, 2]]


def test_windows_are_emitted_every_stride_and_alert_on_anomalies():
    scorer = FakeScorer()
    detector = _detector(scorer)
    alerts = []
    for i, tid in enumerate([2, 3, 4, 2, 66, 3]):
        alerts += detector.observe("d1", tid, T0 + timedelta(seconds=i))
    alerts += detector.flush()
    windows = [c[0] for c in scorer.calls]
    assert windows[0] == (2, 3, 4, 2)
    assert (4, 2, 66, 3) in windows
    assert alerts and alerts[0].reasons == ("sequence",) and alerts[0].device_id == "d1"


def test_reorder_buffer_restores_event_time_order():
    config = DetectorConfig(window=3, stride=3, min_len=1, batch_size=100,
                            lateness=timedelta(seconds=5), fleet_min_devices=0)
    scorer = FakeScorer()
    detector = StreamingDetector(scorer, config)
    for offset, tid in [(2, 4), (0, 2), (1, 3)]:  # arrives out of order
        detector.observe("d1", tid, T0 + timedelta(seconds=offset))
    detector.flush()
    assert scorer.calls[0][0] == (2, 3, 4)


def test_devices_are_windowed_independently():
    scorer = FakeScorer()
    detector = _detector(scorer)
    for i in range(4):
        detector.observe("a", 2, T0 + timedelta(seconds=i))
        detector.observe("b", 3, T0 + timedelta(seconds=i))
    windows = {c[0] for c in scorer.calls}
    assert windows == {(2, 2, 2, 2), (3, 3, 3, 3)}


def test_fleet_wide_recurring_templates_become_familiar_but_bursts_do_not():
    detector = _detector(fleet_min_devices=2, fleet_min_minutes=3)
    for minute in range(3):  # rollout: template 50 on two devices across three minutes
        for device in ("a", "b"):
            detector.observe(device, 50, T0 + timedelta(minutes=minute))
    for i in range(20):  # burst: template 60 many times within seconds on two devices
        for device in ("a", "b"):
            detector.observe(device, 60, T0 + timedelta(minutes=5, seconds=i))
    assert detector.familiar() == frozenset({50})


def test_pending_lines_are_re_resolved_at_scoring_time():
    scorer = FakeScorer()
    detector = _detector(scorer, fleet_min_devices=0)
    detector.resolver = lambda line: 3 if "learned" in line else 1
    for i, (tid, line) in enumerate([(2, None), (1, "learned later"), (1, "still unknown"), (4, None)]):
        detector.observe("d1", tid, T0 + timedelta(seconds=i), line)
    detector.flush()
    assert scorer.calls[0][0] == (2, 3, 1, 4)


def test_recorded_windows_keep_scores():
    detector = _detector(record_windows=True)
    for i, tid in enumerate([2, 3, 4, 66]):
        detector.observe("d1", tid, T0 + timedelta(seconds=i))
    detector.flush()
    assert detector.scored and detector.scored[0].score.length == 4


def test_resolved_lines_count_spread_at_their_own_time_not_per_window():
    # One fault line per device, re-resolved in several overlapping windows,
    # must not look like a template recurring across many minutes.
    config = DetectorConfig(window=4, stride=1, min_len=2, batch_size=1, lateness=timedelta(0),
                            fleet_min_devices=2, fleet_min_minutes=3)
    detector = StreamingDetector(FakeScorer(), config, resolver=lambda line: 77)
    for device in ("a", "b"):
        for i in range(8):
            tid, line = (1, "fault") if i == 3 else (2, None)
            detector.observe(device, tid, T0 + timedelta(minutes=i), line)
    detector.flush()
    assert 77 not in detector.familiar()
