# Code structure

```
Log_parser/
├── logpipe/                 the product (Python package, CLI `logpipe`)
├── DeepParse/               mask synthesis, Drain, chat providers, web-chat scraper (package `deepparse`)
├── agent/                   logpipe-agent, the Go collector
├── web/                     operator UI (React + TypeScript + Vite), built into logpipe/server/static
├── tests/                   pytest for logpipe (DeepParse has DeepParse/tests)
├── eval/baseline.json       numbers the nightly gate compares against
├── docs/                    this site (MkDocs)
├── notebooks/               Colab: fine-tune / evaluate the mask model on a GPU
├── docker/                  ClickHouse / Mongo init scripts
├── docker-compose.yaml      ClickHouse, Mongo, Kafka, server, log generator, agent (profiles)
├── Dockerfile               server image (builds web/ then the Python package)
├── Makefile                 every task; `make help`
└── artifacts/               generated: registries, models, datasets, eval results (git-ignored)
```

## `logpipe/`

| Path | Responsibility | Key types / functions |
|---|---|---|
| `cli.py` | `logpipe` subcommands | `main()` |
| `start.py` | interactive `logpipe start` | `start()` |
| `settings.py` | every env setting and tuning number | `get_settings()`, `RcaSettings`, `AskSettings`, `RuntimeSettings` |
| `pipeline.py` | the ingest loop: parse → store → detect, periodic enrichment | `Pipeline.process()`, `.enrich()`, `PipelineStats` |
| `registry.py` | template ids, aliases, masks, versions | `SchemaRegistry`, `RegistrySnapshot`, `NewTemplate` |
| `calibrate.py` | registry v1 from history, holdout coverage, GA/PA, tuning | `calibrate()`, `append_drafts()` |
| `parsing/` | hot path | `FrozenParser`, `ParserHandle`, `make_engine`, `grok.compile_rule`, `kv.extract` |
| `sources/` | `LogEvent`, file replay, timestamps from lines | `LogEvent`, `replay()`, `event_time()` |
| `storage/` | write path per backend, ClickHouse schema | `LogStore`, `LogRow`, `AnomalyAlert`, `MemoryStore`, `ClickHouseStore`, `MongoStore` |
| `unresolved/` | queue between fast path and learner | `UnresolvedQueue`, `MemoryQueue`, `KafkaQueue`, codec |
| `enrich/` | online learning and LLM guards | `EnrichmentWorker`, `LLMAdapter`, `get_adapter`, `find_alias`, `validate`, `sql_guard` |
| `anomaly/` | LogBERT model, training, scoring, streaming detector | `StreamingDetector`, `LogBERTScorer`, `Thresholds`, `tune_thresholds`, `train_logbert` |
| `server/` | `logpipe server` | see below |
| `ask/` | Ask and root-cause analysis | see below |
| `eval/` | every benchmark | `parsing_eval`, `alias_eval`, `iot_eval`, `logbert_eval`, `rca_eval`, `rcaeval_bench`, `report`, `gate` |
| `demo/` | `logpipe demo` (one-file app with its own page) | `Demo`, `serve()` |
| `showcase/` | `logpipe showcase` and the web-shop log generator | `ShopGenerator`, `FAULTS` |
| `bench.py` | `logpipe bench` | — |

### `logpipe/server/`

| File | Responsibility |
|---|---|
| `__init__.py` | `run()`: build stores, control plane, server, uvicorn; logging setup |
| `app.py` | the FastAPI app: `/intake/v1/*`, `/v1/logs`, `/api/*`, `/mcp`, auth middleware, static UI |
| `core.py` | `Server`: intake buffer → pipeline, sources/streams, jobs (draft, apply, retrain, reextract, ask), housekeeping thread, incidents, templates, metrics |
| `control.py` | `ControlStore`: JSON docs per (kind, id) on memory / ClickHouse / Mongo |
| `review.py` | draft batches: approve, edit, merge, reject, apply |
| `stores/` | read path per backend: `StoreReads` + `Compiler` (`base`, `sql`, `memory`, `clickhouse`, `mongo`) |
| `otlp.py` | OTLP/HTTP logs → `LogEvent`s |
| `auth.py` | admin password (scrypt), sessions, lockout |
| `mcp_tools.py` | MCP tools, prompts, allowlist |
| `metrics.py`, `drift.py`, `groups.py` | self-metrics and alert rules, PSI drift, stream groups |

### `logpipe/ask/`

| File | Responsibility |
|---|---|
| `spec.py` | `QuerySpec`, `validate()` (fail closed) |
| `router.py` | question → spec: regex fast path, else one LLM call |
| `query.py` | `Filters`, `Query` (one read) |
| `engine.py` | rounds: `first_round`, `advance`, `execute`, `run_unattended` |
| `rca.py` | anchor, window, baseline, ranking, evidence bundle |
| `hints.py` | value hints (where errors concentrate) |
| `synth.py` | LLM answer + citation check; offline leads |
| `moves.py` | typed follow-up reads the LLM may ask for |
| `index.py` | MiniLM text index, LogBERT neighbours, shutdown set |
| `models.py` | pydantic shapes of rows, report, answer |
| `service.py` | plans, approval, jobs, glue to the server |
| `activity.py` | live log of what the LLM is doing for a plan |

## `DeepParse/deepparse/`

| Path | Responsibility |
|---|---|
| `drain/` | `DrainEngine` (typed-placeholder masks), mask application |
| `synth/` | mask synthesis: offline stub, HF DeepSeek-R1 + LoRA, chat provider |
| `providers/` | `ChatProvider` protocol + Anthropic, OpenAI, Gemini, Groq, and `web` (a signed-in chat app) |
| `ai_scraper/` | Playwright driver for Gemini / Claude / ChatGPT web apps: sessions, sites as data, cookie rejection |
| `metrics/`, `evaluation/` | GA / PA and the paper's evaluation runner |
| `training/`, `tools/` | LoRA fine-tune, training-set builders, Loghub fetchers, the IoT fleet generator |

## `agent/` (Go)

| Path | Responsibility |
|---|---|
| `main.go` | CLI: `init`, `run`, `track` / `untrack` / `list`, `config get\|set\|edit\|show\|path`, `status`, `check`, `version` |
| `internal/config` | `agent.yaml` + `conf.d/*.yaml` (Datadog layout), remote config |
| `internal/source` | tail files (with rotation and offsets), journald, docker socket, syslog |
| `internal/process` | multiline aggregation, then exclude / include / mask rules in written order |
| `internal/ship` | OTLP/HTTP JSON batches, gzip, size-capped disk buffer while the server is unreachable |
| `internal/agent` | run loop, heartbeats, long-poll for config pushes, status |

## `web/src/`

`App.tsx` (routes, layout), `api.ts` (typed client for `/api`), `ask.tsx` (Ask panel, plan
approval, RCA report), `pages/` (Setup, Overview, Logs, Templates, Review, Query, Stats, Alerts,
Maintenance, Streams, Jobs, Agents, Keys, Settings, System, Bench), `charts.tsx`, `rules.tsx`
(parsing-rule editor), `shortcuts.tsx`, `notify.tsx`, `theme.ts`.

## Tests

| File(s) | Covers |
|---|---|
| `test_server.py`, `test_mcp.py`, `test_auth.py` | server routes, lifecycle, MCP allowlist, `ask` tool |
| `test_ask_*.py`, `test_rca_*.py` | router, synth, moves, service, RCA baseline, replay of recorded LLM replies |
| `test_unresolved_queue.py`, `test_kafka_*.py` | queue semantics, Kafka with fakes and live |
| `test_prompt_injection.py` | hostile log lines vs every LLM guard |
| `test_*_integration.py` | `-m integration`: ClickHouse, Mongo, Kafka |
| `test_rcaeval_bench.py` | RCAEval adapter on a hand-made case |

Markers: `integration` (needs services), `slow` (trains a model).
