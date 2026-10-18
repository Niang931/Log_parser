"""The nightly regression gate: collect, compare, baseline round trip."""

from __future__ import annotations

import json
from pathlib import Path

from logpipe.eval.gate import collect, compare, run_gate, write_baseline


def _write_results(root: Path, ga: float, fp: int, f1: float) -> None:
    (root / "parsing.json").write_text(
        json.dumps(
            [
                {"system": "A", "GA": ga, "PA": 0.5},
                {"system": "B", "GA": ga, "PA": 0.7},
            ]
        ),
        encoding="utf-8",
    )
    (root / "alias.json").write_text(
        json.dumps(
            {
                "pairs": 28,
                "deciders": {
                    "rules_offline": {"precision": 1.0, "recall": 0.6, "TP": 6, "FP": fp, "FN": 4}
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "iot").mkdir(exist_ok=True)
    (root / "iot" / "results.json").write_text(
        json.dumps(
            {
                "variants": {"full": {"test": {"P": 0.6, "R": 0.5, "F1": f1}}},
            }
        ),
        encoding="utf-8",
    )


def test_collect_reads_the_three_benchmarks(tmp_path: Path) -> None:
    _write_results(tmp_path, ga=0.7, fp=0, f1=0.55)
    got = collect(tmp_path)
    assert got["parsing"] == {"systems": 2, "mean_GA": 0.7, "mean_PA": 0.6}
    assert got["alias"]["false_merges"] == 0 and got["alias"]["decider"] == "rules_offline"
    assert got["iot"]["F1"] == 0.55


def test_small_drop_within_tolerance_passes(tmp_path: Path) -> None:
    _write_results(tmp_path, ga=0.70, fp=0, f1=0.55)
    baseline_path = write_baseline(collect(tmp_path), tmp_path / "baseline.json")
    _write_results(tmp_path, ga=0.69, fp=0, f1=0.52)  # -0.01 GA, -0.03 F1
    assert run_gate(tmp_path, baseline_path) == 0


def test_regression_fails_and_names_the_check(tmp_path: Path, capsys) -> None:
    _write_results(tmp_path, ga=0.70, fp=0, f1=0.55)
    baseline_path = write_baseline(collect(tmp_path), tmp_path / "baseline.json")
    _write_results(tmp_path, ga=0.60, fp=1, f1=0.55)  # GA -0.10 and one wrong merge
    assert run_gate(tmp_path, baseline_path) == 1
    out = capsys.readouterr().out
    assert "[FAIL] parsing mean GA" in out and "[FAIL] alias wrong merges" in out
    assert "[PASS] IoT fleet F1" in out


def test_missing_results_fail_rather_than_pass_silently(tmp_path: Path) -> None:
    _write_results(tmp_path, ga=0.70, fp=0, f1=0.55)
    baseline = collect(tmp_path)
    (tmp_path / "iot" / "results.json").unlink()
    checks = compare(collect(tmp_path), baseline)
    iot = next(c for c in checks if c.name.startswith("IoT"))
    assert not iot.ok and "missing" in iot.detail


def test_gate_without_baseline_returns_2(tmp_path: Path) -> None:
    _write_results(tmp_path, ga=0.7, fp=0, f1=0.5)
    assert run_gate(tmp_path, tmp_path / "nope.json") == 2
