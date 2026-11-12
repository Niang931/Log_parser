# Run it

## 1. Interactive start

```sh
uv run logpipe            # same as `logpipe start` or `make start`
```

It asks what to run (browser demo, your own logs, or the server for agents), which queue
(in-memory, or Kafka, which it starts in Docker for you), who names keys and confirms
re-associations (offline rules, a signed-in web chat, or a hosted API), and the few settings
that matter. Then it prints the equivalent commands and runs them. Answers are saved in
`.logpipe/start.json` and become next time's defaults.

## 2. Browser demo: the pipeline on a synthetic IoT fleet

```sh
uv run logpipe demo                       # asks queue, store, LLM, masks, size, port
uv run logpipe --browser chrome demo --llm llm:web:gemini   # flags skip the questions
```

The demo builds a small synthetic fleet, calibrates, trains a tiny LogBERT and tunes its
thresholds (about 10 s), then generates the test stream live and runs it through the whole
system.

- **System** tab: every stage with live counters (frozen Drain → hit/miss → unresolved queue
  → shadow Drain → LLM → validate → alias → registry vN+1 → hot-swap/back-fill, plus
  LogBERT → alerts). Click an event or a line to light up the path it took.
- **Timeline**: true positives, false positives and missed anomalies, firmware-2 template
  shifts and whether each was re-associated, requeues and dead letters.
- **Controls**: pause, step, speed. You can trigger the firmware rollout, inject an anomaly
  into a device's next episode, or restart with a different queue, store or LLM, either from
  calibration or keeping what was learned so far.

Calibration is cached in `artifacts/demo/` per mask backend, fleet size and seed, so only
LogBERT retrains on a restart. `--recalibrate` redoes it.

| flag | effect |
|---|---|
| `--queue kafka` | Kafka (started in Docker if needed, fresh topic per session) |
| `--store clickhouse` | also write rows to ClickHouse (`logpipe_demo`, emptied per session) |
| `--llm llm:web:gemini` | key names and alias confirmation by your web chat |
| `--registry demo_registry.json` | load masks made by the fine-tuned model (see `notebooks/`) |
| `--rate`, `--devices`, `--test-hours`, `--epochs`, `--port` | size and speed |

## 3. Showcase: the product, guided

```sh
uv run logpipe showcase           # asks store and LLM; --no-ask for memory + offline
uv run logpipe showcase --auto    # does every act itself
```

It starts the server, a generated web shop's logs (nginx, a Java API with stack traces,
Postgres) and a real `logpipe-agent` built from `agent/` (run as you, no sudo). Then it walks
through eight acts: connect the agent, draft templates, review them and write a parsing rule,
calibrate and train, explore, a payment-gateway outage that LogBERT flags, and a new service
whose templates are drafted for review automatically. Each act waits for you to do it in the
UI; Enter lets it do the act instead. Files go to `artifacts/showcase/`.

## 4. The platform: agents + server + UI

```sh
cp .env.example .env
docker compose --profile platform up --build
# open http://localhost:8080 — the first start prints the admin password:
docker compose logs server | grep -i password
```

The `platform` profile runs ClickHouse, the server, a log generator and an agent. Follow the
setup guide in the UI: API key → install agents → track files → sample size → streams → review
templates → train. `--profile platform-mongo` runs the same on MongoDB.

Without Docker:

```sh
uv run logpipe server --store memory           # API + UI on :8080
cd web && npm ci && npm run dev                # UI dev server on :5173, proxies /api
cd agent && go build -o logpipe-agent . && ./logpipe-agent -h
```

Install an agent on a real host: [Install the agent](agent.md) (container or Go binary, no root
needed). Anything that speaks OTLP/HTTP logs can send to `POST /v1/logs` too.
[Server and learning lifecycle →](../agent-platform.md)

## 5. Batch: a directory of log files

```sh
# LOGS_PATH in .env; each file is one device
make calibrate      # masks + templates + key names from history → registry v1 (+ .report.json)
make train          # LogBERT on per-device windows of the history
make run            # parse, store, learn PENDING lines, detect
```

End to end on the synthetic fleet:

```sh
uv run logpipe gen-iot --out artifacts/fleet
uv run logpipe calibrate --logs artifacts/fleet/history
uv run logpipe train-logbert --logs artifacts/fleet/history
uv run logpipe run --logs artifacts/fleet/test             # --store clickhouse after make docker-up
```

### With Kafka: a fast path and separate learning workers

```sh
make docker-up
uv run logpipe worker --queue kafka --store memory --llm offline         # 1+ terminals
uv run logpipe run --queue kafka --no-enrich --logs artifacts/fleet/test --store memory
```

[Unresolved queue on Kafka →](../kafka.md)

## 6. Connect an AI assistant (MCP)

1. In the UI, **Settings → API keys**, create a key with use **AI assistant**.
2. Add it to Claude Code:

   ```sh
   claude mcp add --transport http logpipe https://<your-server>/mcp \
     --header "Authorization: Bearer lp_..."
   ```

3. Ask "investigate the latest open incident", or use the `investigate_incident` /
   `review_batch` prompts. [MCP →](../architecture/mcp.md)

## 7. Benchmarks

```sh
make eval                                            # parsing, alias, IoT, HDFS, BGL + report
make eval-rcaeval DATASET=re3-ob                     # RCAEval, rules only
make eval-rcaeval DATASET=re3-ob LLM=llm:web:gemini BROWSER=chromium
make eval-rcaeval DATASET=re3-ob VIA=mcp             # Claude Code over /mcp
uv run logpipe bench                                 # throughput of one server
```

[Benchmarks →](../benchmarks/index.md)
