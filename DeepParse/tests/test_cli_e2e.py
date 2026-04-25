import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)


def test_cli_demo_pipeline(tmp_path):
    (ROOT / "artifacts" / "outputs").mkdir(parents=True, exist_ok=True)
    run(["python", "-m", "deepparse.cli", "synth", "--dataset", "DemoTiny", "--mode", "offline", "--out", "artifacts/masks/DemoTiny.json"])
    result = run(["python", "-m", "deepparse.cli", "eval", "--config", "configs/demo_small.yaml"])
    golden = (ROOT / "tests" / "golden" / "demo_eval.txt").read_text(encoding="utf-8").strip()
    assert golden in result.stdout
