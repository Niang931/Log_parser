#!/usr/bin/env bash
# Run the load generator against ClickHouse + Kafka while one fault is injected.
#
#   scripts/chaos.sh worker|llm|clickhouse|kafka|none
#
# The firmware rollout starts with the fault, so new (PENDING) templates arrive during it.
# none = no fault; with DURATION=86400 this is the 24 h soak (`make soak`).
# PASS = every generated line is in `logs` (uniqExact(log_id)) and nothing stays PENDING.
# Settings (env): DEVICES RATE DURATION FAULT_AT FAULT_FOR DRAIN_TIMEOUT LLM ENV_FILE.
# Each run gets its own database, topic and group, so older runs cannot disturb the count.

set -euo pipefail
cd "$(dirname "$0")/.."

die() { echo "chaos.sh: $*" >&2; exit 1; }

SCENARIO=${1:-}
[[ $SCENARIO =~ ^(worker|llm|clickhouse|kafka|none)$ ]] || die "usage: chaos.sh worker|llm|clickhouse|kafka|none"

DEVICES=${DEVICES:-100}
RATE=${RATE:-500}
DURATION=${DURATION:-300}
FAULT_AT=${FAULT_AT:-60}
FAULT_FOR=${FAULT_FOR:-45}
DRAIN_TIMEOUT=${DRAIN_TIMEOUT:-300}
LLM=${LLM:-offline}
LOGPIPE=${LOGPIPE:-uv run ${ENV_FILE:+--env-file $ENV_FILE} logpipe}
REGISTRY=${REGISTRY_PATH:-artifacts/registry.json}
[[ -f $REGISTRY ]] || die "$REGISTRY not found: run make calibrate first"

ID="$SCENARIO-$(date +%Y%m%d-%H%M%S)"
DIR=artifacts/chaos/$ID
mkdir -p "$DIR"
export CLICKHOUSE_DB="chaos_${ID//-/_}" KAFKA_TOPIC="chaos.$ID" KAFKA_GROUP="chaos-$ID"
export LLM_DOWN_FILE="$PWD/$DIR/llm_down"

ch() {
    docker exec -i logpipe_clickhouse clickhouse-client ${CLICKHOUSE_PASSWORD:+--password "$CLICKHOUSE_PASSWORD"} "$@"
}

PIDS=()
# `uv run` is a wrapper: kill its python child too, or the process lives on as an orphan.
killtree() { pkill -"$1" -P "$2" 2>/dev/null || true; kill -"$1" "$2" 2>/dev/null || true; }
cleanup() { for pid in "${PIDS[@]}"; do killtree TERM "$pid"; done; }
trap cleanup EXIT

docker compose up -d --wait
ch -q "CREATE DATABASE $CLICKHOUSE_DB"

start_worker() {  # $1 = name; own registry copy so workers never write the same file
    [[ -f $DIR/$1.json ]] || cp "$REGISTRY" "$DIR/$1.json"
    $LOGPIPE worker --queue kafka --store clickhouse --llm "$LLM" --registry "$DIR/$1.json" \
        >>"$DIR/$1.log" 2>&1 &
    PIDS+=($!)
    eval "PID_$1=$!"
}

start_worker w1
start_worker w2
$LOGPIPE load --no-enrich --store clickhouse --queue kafka --registry "$REGISTRY" \
    --devices "$DEVICES" --rate "$RATE" --duration "$DURATION" --rollout-at "$FAULT_AT" \
    --metrics-out "$DIR/metrics.jsonl" >"$DIR/load.log" 2>&1 &
LOAD=$!
PIDS+=($LOAD)
echo "run $ID: db=$CLICKHOUSE_DB topic=$KAFKA_TOPIC logs in $DIR"

inject() {
    echo "fault: $SCENARIO at ${FAULT_AT}s for ${FAULT_FOR}s"
    case $SCENARIO in
        worker)
            killtree KILL "$PID_w1"; sleep 10; start_worker w1 ;;
        llm)
            touch "$LLM_DOWN_FILE"; sleep "$FAULT_FOR"; rm -f "$LLM_DOWN_FILE" ;;
        clickhouse | kafka)
            docker stop "logpipe_$SCENARIO"; sleep "$FAULT_FOR"; docker start "logpipe_$SCENARIO"
            docker compose up -d --wait ;;
    esac
}
if [[ $SCENARIO != none ]]; then
    sleep "$FAULT_AT"
    inject
fi
wait "$LOAD" || die "load failed, see $DIR/load.log"

GENERATED=$(sed -n 's/.*"generated": \([0-9]*\).*/\1/p' "$DIR/load.log" | tail -1)
[[ -n $GENERATED ]] || die "no generated count in $DIR/load.log"

STORED=0 PENDING=-1
deadline=$((SECONDS + DRAIN_TIMEOUT))
while ((SECONDS < deadline)); do
    STORED=$(ch -d "$CLICKHOUSE_DB" -q "SELECT uniqExact(log_id) FROM logs")
    PENDING=$(ch -d "$CLICKHOUSE_DB" -q "SELECT count() FROM logs FINAL WHERE status = 'PENDING'")
    [[ $STORED -eq $GENERATED && $PENDING -eq 0 ]] && break
    sleep 5
done

verdict=FAIL
[[ $STORED -eq $GENERATED && $PENDING -eq 0 ]] && verdict=PASS
echo "RESULT scenario=$SCENARIO generated=$GENERATED stored=$STORED pending=$PENDING $verdict" | tee "$DIR/result.txt"
[[ $verdict == PASS ]]
