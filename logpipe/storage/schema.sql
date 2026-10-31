-- Run by create_schema() at startup. {placeholders} come from the environment.
-- IF NOT EXISTS: changing a TTL later needs a manual ALTER TABLE ... MODIFY TTL.

-- A backfill inserts a FINAL row with the same log_id and a higher version;
-- merges keep the highest version. log_id in ORDER BY keeps rows of the same millisecond.
CREATE TABLE IF NOT EXISTS logs (
    log_id UUID,
    timestamp DateTime64(3, 'UTC'),
    device_id LowCardinality(String),
    template_id Int64,
    status Enum8('FINAL' = 1, 'PENDING' = 2),
    registry_version UInt32,
    payload Map(String, String),
    -- heaviest column, expires first
    raw_message String TTL toDateTime(timestamp) + INTERVAL {raw_ttl_days} DAY,
    version UInt32,
    -- ingest time: PENDING age counts from here, so replayed old logs still get time to resolve
    inserted_at DateTime('UTC') DEFAULT now(),
    INDEX status_idx status TYPE set(2) GRANULARITY 4,
    INDEX inserted_idx inserted_at TYPE minmax GRANULARITY 1
)
ENGINE = ReplacingMergeTree(version)
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (device_id, timestamp, log_id)
-- Old PENDING rows are deleted 1 hour after the view below copies them.
-- The hour must stay longer than the view's 10 minute refresh.
TTL toDateTime(timestamp) + INTERVAL {log_ttl_days} DAY,
    inserted_at + INTERVAL {pending_max_age_hours} HOUR + INTERVAL 1 HOUR
        DELETE WHERE status = 'PENDING';

CREATE TABLE IF NOT EXISTS alerts (
    alert_id UUID,
    device_id LowCardinality(String),
    window_start DateTime64(3, 'UTC'),
    window_end DateTime64(3, 'UTC'),
    template_ids Array(Int64),
    miss_ratio Float64,
    novelty_ratio Float64,
    distance Nullable(Float64),
    reasons Array(String),
    had_unknowns Bool,
    registry_version UInt32,
    model_version String
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(window_start)
ORDER BY (device_id, window_start)
TTL toDateTime(window_end) + INTERVAL {alert_ttl_days} DAY;

CREATE TABLE IF NOT EXISTS logs_dead_letter (
    log_id UUID,
    timestamp DateTime64(3, 'UTC'),
    device_id LowCardinality(String),
    template_id Int64,
    status Enum8('FINAL' = 1, 'PENDING' = 2),
    registry_version UInt32,
    payload Map(String, String),
    raw_message String,
    version UInt32,
    dead_lettered_at DateTime('UTC'),
    reason String
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (device_id, timestamp, log_id)
TTL dead_lettered_at + INTERVAL {log_ttl_days} DAY;

-- Every 10 minutes: copy rows still PENDING after pending_max_age into logs_dead_letter.
-- Inner query has no FINAL so the skip indexes work (docs/clickhouse.md).
CREATE MATERIALIZED VIEW IF NOT EXISTS logs_dead_letter_mv
REFRESH EVERY 10 MINUTE APPEND TO logs_dead_letter
AS SELECT
    log_id, timestamp, device_id, template_id, status, registry_version, payload,
    raw_message, version, now() AS dead_lettered_at, 'pending_max_age' AS reason
FROM logs FINAL
WHERE status = 'PENDING'
  AND (device_id, timestamp, log_id) IN (
      SELECT device_id, timestamp, log_id FROM logs
      WHERE status = 'PENDING'
        AND inserted_at < now() - INTERVAL {pending_max_age_hours} HOUR
        AND inserted_at >= now() - INTERVAL {pending_max_age_hours} HOUR - INTERVAL 1 HOUR
  )
  AND log_id NOT IN (
      SELECT log_id FROM logs_dead_letter
      WHERE dead_lettered_at >= now() - INTERVAL 2 HOUR
  );
