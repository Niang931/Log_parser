# Unresolved queue on Kafka

Lines the fast parser does not recognise are stored as `PENDING` and handed to the learning worker through the *unresolved queue*. Until now that queue was a Python `deque` inside the one `logpipe run` process. It can now be a Kafka topic, which gives two things the design (technical.md §3, §8) asks for:

- **durability**: a restart or crash replays the work instead of losing it;
- **several workers**: `logpipe worker` processes form a consumer group and share
  the topic's partitions.

`QUEUE=memory` (the default) keeps the old single-process behaviour; nothing changes for tests, `make demo` or the evaluation.

## How it works

```
logpipe run --queue kafka --no-enrich      (fast path, producer group)
    parse -> store PENDING -> produce to logpipe.unresolved  (never blocks)

logpipe worker --queue kafka               (learning workers, consumer group)
    consume batch -> cluster -> LLM once per cluster -> publish registry v.N+1
    -> backfill FINAL rows -> re-queue / dead-letter the rest -> commit offsets
```

| Piece | Where | What it does |
|---|---|---|
| `logpipe/unresolved/base.py` | `UnresolvedQueue` protocol, `Message`, `QueueStats` | one interface for both queues |
| `logpipe/unresolved/memory.py` | `MemoryQueue` | in-process queue, also the test double |
| `logpipe/unresolved/kafka.py` | `KafkaQueue` | confluent-kafka producer + consumer |
| `logpipe/unresolved/codec.py` | `encode_row` / `decode_row`, `partition_key` | JSON wire format, partition key |
| `logpipe/enrich/worker.py` | `EnrichmentWorker.run_cycle` | consume → learn → backfill → commit |
| `logpipe/cli.py` | `run --queue`, `worker` | commands |

**Partitioning.** The key is the line's *shape* (tokens with digits collapsed, hashed), so all lines of one not-yet-known template land in the same partition and one worker sees the whole cluster. `KAFKA_KEY_BY=device` keys by device id instead.

**Never block the fast path.** `publish` hands the message to librdkafka's send buffer and returns. When the broker is down, deliveries time out (30 s) or the buffer fills; those rows are parked in a bounded local buffer (`spilled` in the stats), one  alert is logged, and the buffer drains when deliveries succeed again. If even the local buffer overflows, the oldest rows are dropped (`dropped`): they are still `PENDING` in the store and `logpipe run --rehydrate` re-queues them.

**At least once.** Workers run with `enable.auto.commit=false`. A cycle commits its
offsets only after the resolved rows are written and flushed. A crash before that
redelivers the batch; the backfill inserts the same `log_id` with the same version,
which the ReplacingMergeTree collapses, so a redelivery changes nothing.

**Dead letters.** A row whose cluster fails validation (`template_problem`) counts
one attempt; after `--max-attempts` (default 3) it goes to `<topic>.dlq` with
`attempts` and `reason` headers. Rows in clusters the cycle did not look at (too
small, over the per-cycle budget) keep their count.

**Lag.** `QueueStats.lag` is the sum over the worker's partitions of
`high watermark − position`; `logpipe worker` prints it after every non-empty cycle,
and `logpipe run` prints the producer-side counters in its summary.

## Running it locally

```bash
make docker-up
```

starts ClickHouse and a one-broker Kafka (KRaft, no ZooKeeper) on `localhost:9092`.
Then, in two terminals:

```bash
uv run logpipe worker --queue kafka --store memory --llm offline
```

```bash
uv run logpipe run --queue kafka --no-enrich --logs artifacts/fleet/test --store memory
```

With `--store memory` each process has its own store, so the worker's backfill is
only visible in its process; use `STORE=clickhouse` once that backend lands for a
shared store. `make worker` and `make run-fast` wrap the two commands.

Settings (`.env.example`): `QUEUE`, `KAFKA_BOOTSTRAP`, `KAFKA_TOPIC`, `KAFKA_GROUP`,
`KAFKA_DLQ_TOPIC`, `KAFKA_KEY_BY`.

## Tests

- `tests/test_unresolved_queue.py`: codec, `MemoryQueue`, and the worker's
  commit-after-backfill, re-queue and dead-letter behaviour (no broker).
- `tests/test_kafka_queue.py`: `KafkaQueue` against fake clients: spill, alert,
  headers, dead letters (no broker).
- `tests/test_kafka_integration.py` (`-m integration`): round trip and a full worker
  cycle on a real broker; skipped when nothing listens on `KAFKA_BOOTSTRAP`.

Run them with `make test` and `make test-integration`.

## CI

The integration job needs a Kafka service next to ClickHouse. Add this under
`services:` of the `integration` job in `.github/workflows/ci.yml` and
`KAFKA_BOOTSTRAP: 127.0.0.1:9092` to its `env:` once that workflow is on `main`:

```yaml
      kafka:
        image: apache/kafka:3.9.1
        ports:
          - 9092:9092
        env:
          KAFKA_NODE_ID: "1"
          KAFKA_PROCESS_ROLES: broker,controller
          KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
          KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
          KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
          KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
          KAFKA_CONTROLLER_QUORUM_VOTERS: 1@localhost:9093
          KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
          KAFKA_NUM_PARTITIONS: "3"
          KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: "1"
          KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: "1"
          KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: "1"
          KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS: "0"
        options: >-
          --health-cmd "/opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list"
          --health-interval 10s
          --health-timeout 10s
          --health-retries 10
```

## Known gaps

- A fast-path process does not reload the registry file a worker wrote, so templates
  learned by a separate worker only reach it on restart. Hot reload across processes
  belongs with the API / containerisation work.
- Consumer lag is printed, not exported; the Prometheus card will expose it as a gauge.
