import json

from logpipe.cli import main
from tests.conftest import FIXTURES


def test_calibrate_then_replay(tmp_path, capsys):
    registry = tmp_path / "registry.json"
    main(["calibrate", "--logs", str(FIXTURES), "--registry", str(registry)])
    main(["replay", "--logs", str(FIXTURES), "--registry", str(registry), "--quiet"])
    out = capsys.readouterr().out
    assert "registry v1" in out and "holdout coverage" in out
    assert "0 PENDING" in out
    report = json.loads(registry.with_suffix(".report.json").read_text())
    assert report["templates"] > 0


def test_train_then_run_full_pipeline(tmp_path, capsys):
    registry, model = tmp_path / "registry.json", tmp_path / "model"
    main(["calibrate", "--logs", str(FIXTURES), "--registry", str(registry)])
    main(["train-logbert", "--logs", str(FIXTURES), "--registry", str(registry), "--out", str(model),
          "--window", "4", "--stride", "2", "--epochs", "1", "--hidden", "16"])
    assert (model / "config.json").exists() and (model / "model.pt").exists()
    capsys.readouterr()
    main(["run", "--logs", str(FIXTURES), "--registry", str(registry), "--model", str(model),
          "--store", "memory", "--window", "4", "--stride", "2"])
    out = capsys.readouterr().out
    summary = json.loads(out[: out.rindex("}") + 1])
    assert summary["lines"] == 17 and summary["pending_at_ingest"] == 0


def test_eval_alias_writes_results(tmp_path, capsys):
    main(["eval", "alias", "--out", str(tmp_path)])
    result = json.loads((tmp_path / "alias.json").read_text())
    assert result["deciders"]["rules_offline"]["FP"] == 0
