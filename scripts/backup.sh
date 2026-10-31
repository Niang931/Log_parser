#!/usr/bin/env bash
# Nightly backup, called by cron and `make backup` (docs/runbook-restore.md).
#
# For every configured target (B2, S3), under <bucket>/<BACKUP_PREFIX>:
#   clickhouse/<tag>/<table>.sql.gz  one day of logs, alerts, logs_dead_letter as INSERT statements
#   artifacts/<tag>/                 registry.json, registry.report.json, model.txt (model hash)
#   models/<hash>/                   the LogBERT model, uploaded only when its hash is new
# tag = <day>-r<registry version>. A failing target does not stop the others;
# the exit code is non-zero if any target failed.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/backup_lib.sh

# Rows of yesterday can still be PENDING; after 2 days they no longer change.
DAY=${DAY:-$(date -u -d '2 days ago' +%Y%m%d)}
REGISTRY_PATH=${REGISTRY_PATH:-artifacts/registry.json}
MODEL_PATH=${MODEL_PATH:-artifacts/logbert}

targets=$(enabled_targets)
[[ -n $targets ]] || die "set BACKUP_B2_BUCKET or BACKUP_S3_BUCKET, nothing was backed up"
[[ -f $REGISTRY_PATH ]] || die "no registry at $REGISTRY_PATH (make calibrate)"

version=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["version"])' \
    "$REGISTRY_PATH") || die "cannot read the registry version from $REGISTRY_PATH"
TAG="$DAY-r$version"

model_hash=""
if [[ -f $MODEL_PATH/model.pt && -f $MODEL_PATH/config.json ]]; then
    model_hash=$(cat "$MODEL_PATH/model.pt" "$MODEL_PATH/config.json" | sha256sum | cut -c1-16)
else
    echo "backup: no model in $MODEL_PATH, skipping it" >&2
fi

backup_to() {
    use_target "$1" || return 1
    local table column final
    for table in logs alerts logs_dead_letter; do
        column=timestamp final=""
        [[ $table == alerts ]] && column=window_start
        [[ $table == logs ]] && final=FINAL  # latest version of each row only
        # Query on stdin, not argv: it holds the secret.
        ch <<SQL || return 1
INSERT INTO FUNCTION s3('$ENDPOINT/$BUCKET/${PREFIX}clickhouse/$TAG/$table.sql.gz',
                        '$KEY_ID', '$SECRET', 'SQLInsert')
SELECT * FROM $table $final WHERE toYYYYMMDD($column) = $DAY
SETTINGS output_format_sql_insert_table_name = '$table', s3_truncate_on_insert = 1
SQL
    done

    local report=${REGISTRY_PATH%.json}.report.json
    s3 cp - "$(url "artifacts/$TAG/registry.json")" <"$REGISTRY_PATH" || return 1
    if [[ -f $report ]]; then
        s3 cp - "$(url "artifacts/$TAG/registry.report.json")" <"$report" || return 1
    fi

    [[ -n $model_hash ]] || return 0
    # config.json goes last, so its presence means the model upload is complete.
    if ! s3 ls "$(url "models/$model_hash/config.json")" >/dev/null; then
        s3 cp - "$(url "models/$model_hash/model.pt")" <"$MODEL_PATH/model.pt" || return 1
        s3 cp - "$(url "models/$model_hash/config.json")" <"$MODEL_PATH/config.json" || return 1
    fi
    echo "$model_hash" | s3 cp - "$(url "artifacts/$TAG/model.txt")"
}

failed=()
for target in $targets; do
    # Subshell: each target sets its own credentials.
    if (backup_to "$target"); then
        echo "backup: $target ok, tag $TAG"
    else
        echo "backup: $target FAILED, tag $TAG" >&2
        failed+=("$target")
    fi
done
(( ${#failed[@]} == 0 )) || die "failed targets: ${failed[*]}"
