#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python -m deepparse.cli table --inputs artifacts/outputs/*.csv --out artifacts/outputs/tables/
