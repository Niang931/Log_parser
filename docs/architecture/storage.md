# Storage

Two layers, each behind an interface:

| Layer | Interface | Backends | Used by |
|---|---|---|---|
| **Write path** | `storage/base.py:LogStore` | `MemoryStore`, `ClickHouseStore`, `MongoStore` | pipeline, worker back-fill, detector alerts |
| **Read path** | `server/stores/base.py:StoreReads` + `Compiler` | `MemoryReads` (SQLite copy), `ClickHouseReads`, `MongoReads` | UI, Query builder, Ask/RCA, MCP |
| **Control plane** | `server/control.py:ControlStore` | in memory, ClickHouse `control_docs`, MongoDB | agents, keys, streams, jobs, drafts, plans, settings |

## Rows: versioned upserts

```sql
CREATE TABLE iot_logs (
    log_id UUID, timestamp DateTime64(3,'UTC'), device_id LowCardinality(String),
    template_id UInt32, log_status Enum8('PENDING'=0,'FINAL'=1),
    registry_version UInt32, version UInt32, payload JSON, raw_message String,
    service LowCardinality(String), source LowCardinality(String), tags Array(LowCardinality(String))
) ENGINE = ReplacingMergeTree(version)
ORDER BY (device_id, timestamp, log_id);
```

A back-fill inserts the same `log_id` and sort key with a higher `version` (the registry
version that resolved it). ClickHouse reads use `FINAL`, MongoDB keeps one document per
`log_id` (`_id = log_id`, replaced only by an equal or higher version), and the memory store mimics
ReplacingMergeTree in Python. Redelivered Kafka batches and repeated back-fills are therefore
no-ops. `device_id` holds the agent's host.

`anomaly_alerts` stores windows over threshold (`miss_ratio`, `novelty_ratio`, `distance`,
`reasons`, `template_ids`, model and registry versions). `server_metrics` (ClickHouse) keeps
the server's own 10-second samples for 30 days.

## Control plane

`control_docs`: one JSON document per `(kind, id)`, newest `version` wins, `deleted` hides
it. Kinds: `agent`, `key` (sha256 only), `auth` (the admin password's scrypt hash), `remote`,
`source`, `template` (counters, links), `stream`, `group`, `warning`, `incident`, `batch` /
`draft`, `job`, `ask_plan`, `ask_shutdown`, `saved_query`, `settings`, `metric_rule` /
`metric_alert`, `bench`, `meta`. The registry and the models live on disk in
`--data-dir` (default `artifacts/server`): `registry.json`, `models/<stream>/`,
`review/<batch>.json`.

## Reads and queries

The read path gives each backend one adapter with two jobs:

- **UI reads**: Log Explorer pages, facets, histograms, `between()`, template samples, the
  learning jobs' samples.
- **Compiler**: the visual builder's spec and Ask's reads become SQL (ClickHouse, or SQLite over a
  read-only copy of the memory store, cached until the next insert) or a MongoDB aggregation
  pipeline (allowed stages only).

Levels are computed, not stored, from one rule table written in the regex subset Python, RE2
and PCRE share, so every backend agrees.

## Choosing a backend

| | memory | ClickHouse | MongoDB |
|---|---|---|---|
| Setup | none | `make docker-up` | `--profile mongo` ([guide](../guides/mongodb.md)) |
| Survives restart | no (only `--data-dir`) | yes | yes |
| Query p95 / percentiles | SQLite | native | `$percentile` |
| Read-only guard | SQLite `query_only` on a copy | `readonly=2` + RO user + table-function block | `check()` + RO URI |
| Ingest measured (`logpipe bench`) | ~10k lines/s (one core, GIL-bound) | ~5k lines/s, insert p99 ~55 ms | — |

To add a backend, see [Replacing components](../guides/replace-components.md#a-storage-backend).
