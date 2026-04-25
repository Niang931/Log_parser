"""End-to-end integration test against real LogHub-2k corrected data.

This test runs the full DeepParse pipeline (offline mask synthesis +
Drain parsing + GA/PA computation) on a single LogHub-2k system that
the user must fetch first via::

    python -m deepparse.tools.fetch_loghub --systems Apache

The test is automatically skipped when ``artifacts/data/Apache`` is
absent so that:

* ``pytest -q`` always passes on a fresh checkout.
* ``pytest -q -m integration`` (CI integration job) exercises the full
  pipeline once data is present.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from deepparse.dataset_loader import load_dataset
from deepparse.drain.drain_engine import DrainEngine
from deepparse.io_paths import build_paths
from deepparse.metrics import grouping_accuracy, parsing_accuracy
from deepparse.synth.r1_deepseek_stub import synthesize_offline
from deepparse.utils.regex_library import validate_regexes

ROOT = Path(__file__).resolve().parents[1]
APACHE_DIR = ROOT / "artifacts" / "data" / "Apache"


@pytest.mark.integration
@pytest.mark.skipif(
    not (APACHE_DIR / "raw.log").exists(),
    reason="Run `python -m deepparse.tools.fetch_loghub --systems Apache` first",
)
def test_apache_offline_pipeline_meets_floor():
    """Full pipeline on real Apache LogHub-2k should at least clear a
    weak floor (GA >= 0.5, PA >= 0.5).  The offline stub uses only the
    four core variable classes, so this floor is intentionally loose —
    the LLM-trained adapter is required for the paper's 0.97 PA.
    """
    paths = build_paths(
        dataset_dir=str(ROOT / "artifacts" / "data"),
        mask_dir=str(ROOT / "artifacts" / "masks"),
        output_dir=str(ROOT / "artifacts" / "outputs"),
        log_dir=str(ROOT / "artifacts" / "outputs" / "logs"),
    )
    dataset = load_dataset("Apache", paths)
    assert len(dataset.logs) == 2000

    masks = synthesize_offline(dataset.logs[:50])
    validate_regexes([m.pattern for m in masks])

    engine = DrainEngine(masks=masks)
    pairs = engine.parse_with_ids(dataset.logs)
    pred_ids = [cid for cid, _ in pairs]
    pred_templates = [tpl for _, tpl in pairs]

    # Ground truth comes from the corrected LogHub-2k templates.json that
    # `fetch_loghub` writes alongside raw.log.
    import json

    payload = json.loads((APACHE_DIR / "templates.json").read_text(encoding="utf-8"))
    entries = payload["entries"]
    gt_ids = [int(e["cluster_id"]) for e in entries]
    gt_templates = [e["template"] for e in entries]

    ga = grouping_accuracy(gt_ids, pred_ids)
    pa = parsing_accuracy(gt_templates, pred_templates)

    # Apache is a structurally simple corpus where the offline stub
    # already does well.  We assert a floor rather than the exact
    # number so this test stays stable across LogHub-2.0 mirror updates.
    assert ga >= 0.5, f"Apache GA={ga:.3f} below floor"
    assert pa >= 0.5, f"Apache PA={pa:.3f} below floor"
