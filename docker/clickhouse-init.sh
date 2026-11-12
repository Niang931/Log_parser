#!/bin/bash
# First start of `clickhouse` only (the entrypoint runs this on an empty data dir): a read-only user
# for the Query page's SQL and Ask's reads. It may SELECT the two log tables and nothing else: no
# control_docs (keys, password hash), no system tables, no url()/file()/remote() sources, and its
# profile pins readonly=2 (settings can change, readonly can't) with a 30 s time limit.
set -euo pipefail
: "${CLICKHOUSE_READONLY_USER:?set CLICKHOUSE_READONLY_USER in .env}"
: "${CLICKHOUSE_READONLY_PASSWORD:?set CLICKHOUSE_READONLY_PASSWORD in .env}"
DB="${CLICKHOUSE_DB:-default}"
clickhouse client --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" --multiquery <<SQL
CREATE SETTINGS PROFILE IF NOT EXISTS logpipe_readonly SETTINGS readonly = 2 CONST, max_execution_time = 30;
CREATE USER IF NOT EXISTS ${CLICKHOUSE_READONLY_USER}
  IDENTIFIED WITH sha256_password BY '${CLICKHOUSE_READONLY_PASSWORD}' SETTINGS PROFILE 'logpipe_readonly';
GRANT SELECT ON ${DB}.iot_logs TO ${CLICKHOUSE_READONLY_USER};
GRANT SELECT ON ${DB}.anomaly_alerts TO ${CLICKHOUSE_READONLY_USER};
SQL
