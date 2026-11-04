# Shared by scripts/backup.sh and scripts/restore.sh. Settings come from the
# environment (.env.example). Docker runs clickhouse-client and the AWS CLI, so
# the host needs neither; set CLICKHOUSE_CLIENT to use a local clickhouse-client.

AWS_CLI_IMAGE=${AWS_CLI_IMAGE:-amazon/aws-cli}

die() {
    echo "$(basename "$0"): $*" >&2
    exit 1
}

# B2 and/or S3: a target is enabled when its bucket is set.
enabled_targets() {
    [[ -n ${BACKUP_B2_BUCKET:-} ]] && echo B2
    [[ -n ${BACKUP_S3_BUCKET:-} ]] && echo S3
    return 0
}

# Sets ENDPOINT, BUCKET, KEY_ID, SECRET, PREFIX and the AWS CLI credentials.
use_target() {
    local name field
    for field in ENDPOINT BUCKET KEY_ID SECRET; do
        name="BACKUP_${1}_$field"
        [[ -n ${!name:-} ]] || { echo "$name is empty" >&2; return 1; }
        printf -v "$field" %s "${!name}"
    done
    ENDPOINT=${ENDPOINT%/}
    PREFIX=${BACKUP_PREFIX:-}
    # https://s3.<region>.backblazeb2.com or https://s3.<region>.amazonaws.com
    AWS_DEFAULT_REGION=$(sed -nE 's#^https?://s3\.([^.]+)\..*#\1#p' <<<"$ENDPOINT")
    AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
    export AWS_ACCESS_KEY_ID=$KEY_ID AWS_SECRET_ACCESS_KEY=$SECRET AWS_DEFAULT_REGION
}

# s3://<bucket>/<prefix><path> of the current target.
url() {
    echo "s3://$BUCKET/$PREFIX$1"
}

# aws s3 <args> against the current target; credentials go through the environment.
s3() {
    docker run --rm -i -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION \
        "$AWS_CLI_IMAGE" --endpoint-url "$ENDPOINT" s3 "$@"
}

# clickhouse-client on CLICKHOUSE_DB; reads queries from stdin or -q.
ch() {
    # shellcheck disable=SC2086 # CLICKHOUSE_CLIENT is a command line
    ${CLICKHOUSE_CLIENT:-docker run --rm -i --network host -e CLICKHOUSE_PASSWORD \
        --entrypoint clickhouse-client clickhouse/clickhouse-server:25.8} \
        --host "${CLICKHOUSE_HOST:-127.0.0.1}" --port "${CLICKHOUSE_NATIVE_PORT:-9000}" \
        --user "${CLICKHOUSE_USER:-default}" --database "${CLICKHOUSE_DB:-default}" "$@"
}
