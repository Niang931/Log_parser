#!/usr/bin/env bash
# DeepParse Tier B — fetch real LogHub-2k corrected data and evaluate the
# offline-stub mask bundle on every system.  No GPU required.
#
# Usage:
#   ./examples/03_loghub_offline_eval.sh                     # all 14 systems
#   ./examples/03_loghub_offline_eval.sh Apache HDFS         # subset

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -gt 0 ]]; then
    SYSTEMS=("$@")
else
    SYSTEMS=(Apache BGL HDFS HPC Hadoop HealthApp Linux Mac OpenSSH OpenStack
             Proxifier Spark Thunderbird Zookeeper)
fi

./scripts/prepare_paths.sh
# Fetch the log files and template.json and write manifest file for further processing
python -m deepparse.tools.fetch_loghub --systems "${SYSTEMS[@]}"

# Build a one-shot eval config covering the requested systems.
TMPCFG="$(mktemp -t deepparse_eval_XXXXXX.yaml)"
{
    echo "base_config: configs/default.yaml"
    echo "datasets:"
    for s in "${SYSTEMS[@]}"; do echo "  - $s"; done
    echo "output_csv: artifacts/outputs/loghub_eval.csv"
    echo "timing_csv: artifacts/outputs/loghub_timing.csv"
} > "$TMPCFG"

deepparse eval  --config "$TMPCFG" --deterministic
# The table command just format the csv to be better looking
deepparse table --inputs artifacts/outputs/loghub_eval.csv \
                --out artifacts/outputs/tables/

echo
echo "=== artifacts/outputs/loghub_eval.csv ==="
cat artifacts/outputs/loghub_eval.csv
