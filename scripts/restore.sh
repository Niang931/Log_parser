#!/usr/bin/env bash
# Restore one backup tag made by scripts/backup.sh (docs/runbook-restore.md):
# load its day of ClickHouse data into the existing tables, then fetch the
# registry and the LogBERT model of that tag to REGISTRY_PATH and MODEL_PATH.
#
# Usage: scripts/restore.sh <tag> [B2|S3]    (default B2)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/backup_lib.sh

TAG=${1:?usage: scripts/restore.sh <tag> [B2|S3]}
DAY=${TAG%%-*}
REGISTRY_PATH=${REGISTRY_PATH:-artifacts/registry.json}
MODEL_PATH=${MODEL_PATH:-artifacts/logbert}
use_target "${2:-B2}" || die "target ${2:-B2} is not configured"

TABLES=(logs alerts logs_dead_letter)
day_column() { if [[ $1 == alerts ]]; then echo window_start; else echo timestamp; fi; }
day_rows() { ch -q "SELECT count() FROM $1 WHERE toYYYYMMDD($(day_column "$1")) = $DAY"; }

# alerts and logs_dead_letter are plain MergeTree: loading a day twice duplicates it.
for table in "${TABLES[@]}"; do
    rows=$(day_rows "$table")
    [[ $rows == 0 ]] || die "$table already has $rows rows of $DAY, restore into an empty day only"
done

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
for table in "${TABLES[@]}"; do
    s3 cp "$(url "clickhouse/$TAG/$table.sql.gz")" - >"$work/$table.sql.gz"
    # A day without rows is a file holding only ";", which clickhouse-client rejects.
    if zgrep -q -m1 '^INSERT' "$work/$table.sql.gz"; then
        gunzip -c "$work/$table.sql.gz" | ch
    fi
    echo "restore: $table $(day_rows "$table") rows of $DAY"
done

# Download next to the target, then rename: a reader never sees half a file.
fetch() {
    mkdir -p "$(dirname "$2")"
    s3 cp "$(url "$1")" - >"$2.part"
    mv "$2.part" "$2"
}

fetch "artifacts/$TAG/registry.json" "$REGISTRY_PATH"
if s3 ls "$(url "artifacts/$TAG/registry.report.json")" >/dev/null; then
    fetch "artifacts/$TAG/registry.report.json" "${REGISTRY_PATH%.json}.report.json"
fi
echo "restore: registry -> $REGISTRY_PATH"

if ! s3 ls "$(url "artifacts/$TAG/model.txt")" >/dev/null; then
    echo "restore: tag $TAG has no model" >&2
    exit 0
fi
hash=$(s3 cp "$(url "artifacts/$TAG/model.txt")" -)
# Restoring several tags in a row downloads the model only when it changes.
local_hash=$(cat "$MODEL_PATH/model.pt" "$MODEL_PATH/config.json" 2>/dev/null \
    | sha256sum | cut -c1-16) || true
if [[ $hash != "$local_hash" ]]; then
    fetch "models/$hash/model.pt" "$MODEL_PATH/model.pt"
    fetch "models/$hash/config.json" "$MODEL_PATH/config.json"
fi
echo "restore: model $hash -> $MODEL_PATH"
