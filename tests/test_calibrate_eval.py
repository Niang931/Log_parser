import json

import pytest
from deepparse.dataset_loader import _create_demo_dataset
from deepparse.evaluation.eval_runner import EvaluationRunner

from logpipe.calibrate import MEASURE_MASK, GroundTruth, calibrate, with_predefined_masks
from logpipe.enrich.llm import OfflineAdapter


@pytest.fixture
def demo(tmp_path):
    path = tmp_path / "data" / "DemoTiny"
    _create_demo_dataset(path)
    return path


def test_report_without_labels_has_holdout_coverage(fixture_lines):
    report = calibrate(fixture_lines * 3, OfflineAdapter()).report
    assert report.ga is None and report.holdout_lines > 0
    assert 0.0 <= report.holdout_coverage <= 1.0
    assert "MEASURE" in report.masks


def test_ground_truth_gives_deepparse_metrics_and_tuning(demo):
    lines, truth = GroundTruth.from_deepparse_dir(demo)
    report = calibrate(lines, OfflineAdapter(), sample_size=50, ground_truth=truth, tune=True).report
    assert report.ga is not None and report.pa is not None and report.baseline_ga is not None
    assert report.tuning and max(r["GA"] for r in report.tuning) == report.ga
    assert report.ga >= report.baseline_ga


def test_baseline_matches_deepparse_evaluation_runner(demo, tmp_path):
    """The calibration baseline is DeepParse's own GA/PA for the same masks and data."""
    base = tmp_path / "base.yaml"
    base.write_text(
        f"seed: 1337\nk: 50\nmode: offline\nstrict: false\n"
        f"dataset_dir: {tmp_path / 'data'}\nmask_dir: {tmp_path / 'masks'}\n"
        f"output_dir: {tmp_path / 'out'}\nlog_dir: {tmp_path / 'logs'}\n", encoding="utf-8")
    config = tmp_path / "eval.yaml"
    config.write_text(
        f"base_config: {base}\ndatasets: [DemoTiny]\noutput_csv: {tmp_path / 'm.csv'}\n",
        encoding="utf-8")
    row = EvaluationRunner(config).evaluate_dataset("DemoTiny")

    lines, truth = GroundTruth.from_deepparse_dir(demo)
    report = calibrate(lines, OfflineAdapter(), sample_size=50, ground_truth=truth).report
    assert (report.baseline_ga, report.baseline_pa) == (row["GA"], row["PA"])


def test_ground_truth_must_align(demo):
    lines, truth = GroundTruth.from_deepparse_dir(demo)
    with pytest.raises(ValueError, match="aligned"):
        calibrate(lines[:-1], OfflineAdapter(), ground_truth=truth)


def test_measure_mask_goes_after_timestamps():
    from deepparse.utils.regex_library import canonical_masks

    masks = with_predefined_masks(canonical_masks())
    labels = [m.label for m in masks]
    assert labels.index(MEASURE_MASK.label) == labels.index("TIMESTAMP") + 1


def test_from_deepparse_dir_rejects_misaligned_labels(tmp_path):
    (tmp_path / "raw.log").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "templates.json").write_text(json.dumps({"entries": [{"template": "a", "cluster_id": 0}]}))
    with pytest.raises(ValueError):
        GroundTruth.from_deepparse_dir(tmp_path)
