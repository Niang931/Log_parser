import json

from deepparse.tools import retrain


def _system(root, name, lines, templates):
    d = root / name
    d.mkdir(parents=True)
    (d / "raw.log").write_text("\n".join(lines) + "\n")
    ids = {t: i for i, t in enumerate(dict.fromkeys(templates))}
    entries = [{"cluster_id": ids[t], "template": t} for t in templates]
    (d / "templates.json").write_text(json.dumps({"entries": entries}))


LINES = [f"user u{i} logged in from 10.0.0.{i}" for i in range(6)] + ["system halted"] * 2
TEMPLATES = ["user <*> logged in from <*>"] * 6 + ["system halted"] * 2


def test_data_leaves_holdout_out_of_every_source(tmp_path, monkeypatch):
    data = tmp_path / "data"
    for name in ("Apache", "OpenSSH"):
        _system(data, name, LINES, TEMPLATES)
    _system(data, retrain.IOT, LINES, TEMPLATES)
    monkeypatch.setattr(retrain, "DEFAULT_SYSTEMS", ["Apache", "OpenSSH"])
    monkeypatch.setattr(retrain, "fetch_systems", lambda systems, out: None)
    manifest = retrain.build_data(tmp_path / "out", ["OpenSSH"], data, data / retrain.IOT,
                                  None, 2, 4, 50)
    assert "loghub2k/Apache" in manifest["per_source"]
    assert not any("OpenSSH" in k for k in manifest["per_source"])
    assert manifest["examples"] == len((tmp_path / "out/train.jsonl").read_text().splitlines())


def test_eval_scores_bundles_and_marks_holdout(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _system(data, "Apache", LINES, TEMPLATES)
    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / "Apache.json").write_text(json.dumps(
        [{"label": "USER", "pattern": r"(?<=user )\S+", "justification": ""},
         {"label": "IPV4", "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "justification": ""}]))
    monkeypatch.setattr(retrain, "DEFAULT_SYSTEMS", ["Apache"])
    results = retrain.evaluate({"v2": masks}, data, tmp_path / "none")
    assert results["Apache"]["v2"] == (1.0, 1.0)
    assert "Apache (held out)" in retrain._report(results, ["Apache"])
