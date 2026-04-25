#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python -m deepparse.cli synth --dataset DemoTiny --mode offline --out artifacts/masks/DemoTiny.json
python -m deepparse.cli eval --config configs/demo_small.yaml --deterministic
python -m deepparse.cli table --inputs artifacts/outputs/demo_metrics.csv --out artifacts/outputs/tables/
