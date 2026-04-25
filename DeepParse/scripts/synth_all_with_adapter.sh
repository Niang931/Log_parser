#!/usr/bin/env bash
# Synthesise per-system mask bundles with a fine-tuned DeepParse adapter.
#
# Usage:
#   scripts/synth_all_with_adapter.sh artifacts/checkpoints/deepparse-r1-8b
#
# Iterates over every directory in artifacts/data/ (each containing raw.log)
# and produces artifacts/masks/<NAME>.json by invoking the LLM via the
# adapter.  Skips DemoTiny since it has its own bundle.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <adapter_dir>" >&2
    exit 1
fi

ADAPTER="$1"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d "$ADAPTER" ]]; then
    echo "Adapter directory not found: $ADAPTER" >&2
    exit 1
fi

mkdir -p artifacts/masks
for system_dir in artifacts/data/*/; do
    system="$(basename "$system_dir")"
    [[ "$system" == "DemoTiny" ]] && continue
    [[ ! -f "${system_dir}raw.log" ]] && continue
    out="artifacts/masks/${system}.json"
    echo "[synth] $system -> $out"
    deepparse synth \
        --dataset "$system" \
        --mode hf \
        --adapter "$ADAPTER" \
        --out "$out"
done
echo "[synth] done."
