#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT_DIR/artifacts/data" "$ROOT_DIR/artifacts/masks" "$ROOT_DIR/artifacts/outputs/logs" "$ROOT_DIR/artifacts/outputs/tables"

echo "[prepare_paths] Ensured artifacts directories exist."
