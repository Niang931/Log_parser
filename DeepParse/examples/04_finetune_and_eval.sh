#!/usr/bin/env bash
# DeepParse Tier C — fine-tune DeepSeek-R1-Distill-Llama-8B with LoRA, then
# evaluate the trained adapter on every LogHub-2k system.
#
# Requires a CUDA or ROCm GPU with >= 24 GB of VRAM (~32 GB if you keep the
# default fp32 mode; ~16 GB with --bf16).  Runtime: 30-90 min on a single
# A100/MI300A.
#
# Usage:
#   ./examples/04_finetune_and_eval.sh                       # paper defaults
#   ./examples/04_finetune_and_eval.sh --small               # CPU smoke run
#                                                              # using Qwen2.5-0.5B
#   ./examples/04_finetune_and_eval.sh --epochs 5 --bf16     # custom

set -euo pipefail
cd "$(dirname "$0")/.."

CHECKPOINT_DIR="artifacts/checkpoints/deepparse-r1-8b"

./scripts/prepare_paths.sh
python -m deepparse.tools.fetch_loghub --out artifacts/data
# It builds out training sets in the format instruction: , input: , output: 
# So receiving a log line, it output regex that match variable parts
# The template.json file that goes with each dataset only has <*> placeholder
# This module is supposed to convert those placeholders into actual regex expressions to fine tune the LLM
python -m deepparse.tools.build_training_set \
    --entropy-k 50 \
    --out artifacts/training/train_paper.jsonl
python -m deepparse.training.finetune \
    --train artifacts/training/train_paper.jsonl \
    --output-dir "$CHECKPOINT_DIR" \
    "$@"

# TODO: check on this LLM adapter module
./scripts/synth_all_with_adapter.sh "$CHECKPOINT_DIR"
deepparse eval  --config configs/eval_16_datasets.yaml --deterministic
deepparse table --inputs artifacts/outputs/table_I_ga_pa.csv \
                --out artifacts/outputs/tables/

echo
echo "=== artifacts/outputs/table_I_ga_pa.csv ==="
cat artifacts/outputs/table_I_ga_pa.csv
