-- Parsed rows. A backfill re-inserts a PENDING row with the same sort key and a higher
-- ``version`` (the registry version that resolved it); ReplacingMergeTree keeps the newest,
-- so read with FINAL.
CREATE TABLE IF NOT EXISTS iot_logs (
    log_id           UUID,
    timestamp        DateTime64(3, 'UTC'),
    device_id        LowCardinality(String),
    template_id      UInt32,
    log_status       Enum8('PENDING' = 0, 'FINAL' = 1),
    registry_version UInt32,
    version          UInt32,
    payload          JSON,
    raw_message      String
) ENGINE = ReplacingMergeTree(version)
ORDER BY (device_id, timestamp, log_id);

CREATE TABLE IF NOT EXISTS anomaly_alerts (
    alert_id         UUID,
    device_id        LowCardinality(String),
    window_start     DateTime64(3, 'UTC'),
    window_end       DateTime64(3, 'UTC'),
    template_ids     Array(UInt32),
    miss_ratio       Float32,
    novelty_ratio    Float32,
    distance         Nullable(Float32),
    reasons          Array(LowCardinality(String)),
    had_unknowns     UInt8,
    registry_version UInt32,
    model_version    String
) ENGINE = MergeTree
ORDER BY (device_id, window_start);
