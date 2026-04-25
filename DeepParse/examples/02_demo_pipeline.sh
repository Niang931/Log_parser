#!/usr/bin/env bash
# DeepParse Tier A — bundled demo, end to end.  Reaches GA = PA = 1.000.
#
# Usage:
#   ./examples/02_demo_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/prepare_paths.sh
deepparse synth --dataset DemoTiny --mode offline --out artifacts/masks/DemoTiny.json
deepparse eval  --config configs/demo_small.yaml --deterministic
deepparse table --inputs artifacts/outputs/demo_metrics.csv \
                --out artifacts/outputs/tables/

echo
echo "=== artifacts/outputs/demo_metrics.csv ==="
cat artifacts/outputs/demo_metrics.csv
