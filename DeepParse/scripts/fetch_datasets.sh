#!/usr/bin/env bash
# Fetch the corrected LogHub-2k benchmark from the public mirror used in the
# DeepParse paper (Khan et al., adopted by logpai/loghub-2.0).  Downloads the
# `*_structured_corrected.csv` and `*_templates_corrected.csv` files for each
# of the 16 systems and converts them to the directory layout expected by
# DeepParse: `artifacts/data/<NAME>/{raw.log, templates.json, manifest.json}`.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/artifacts/data"
mkdir -p "${DATA_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[fetch_datasets] python3 is required" >&2
    exit 1
fi

cd "${ROOT_DIR}"
python3 -m deepparse.tools.fetch_loghub --out "${DATA_DIR}" "${@}"
