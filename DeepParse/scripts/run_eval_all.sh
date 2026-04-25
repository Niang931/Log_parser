#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python -m deepparse.cli synth --dataset ALL --config configs/eval_16_datasets.yaml --mode offline
python -m deepparse.cli eval --config configs/eval_16_datasets.yaml --deterministic
python -m deepparse.cli table --inputs artifacts/outputs/table_I_ga_pa.csv artifacts/outputs/table_II_timing.csv --out artifacts/outputs/tables/
