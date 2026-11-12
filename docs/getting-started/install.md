# Install

## Requirements

| Tool | Version | Needed for |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | recent | everything in Python (it installs Python 3.13 itself) |
| Docker + Compose | recent | ClickHouse, MongoDB, Kafka, the platform profile |
| Go | 1.23+ | building `logpipe-agent` from `agent/` (the showcase does it for you) |
| Node | 22+ | developing the web UI in `web/` (the Docker image builds it for you) |
| Chromium or Camoufox | — | only for `llm:web:<site>`: driving your signed-in web chat |
| A GPU | — | only for `LLM_BACKEND=hf` (the fine-tuned mask model) |

CPU is enough for everything else, including LogBERT training on the demo and eval sizes.

## Python environment

```sh
git clone git@github.com:Niang931/Log_parser.git
cd Log_parser
make setup          # = uv sync --group dev, and creates .env from .env.example
```

`DeepParse/` is a second Python package in the same repo (mask synthesis, the Drain engine, chat
providers, the web-chat scraper). It's installed into the same environment as
`deepparse-artifact`, so nothing extra is needed.

## `.env`

`.env.example` lists what most installs change; every other setting has a default in
`logpipe/settings.py` and can be overridden by its prefix (see [Configuration](../guides/configuration.md)).

| Variable | Default | What it does |
|---|---|---|
| `LOGS_PATH` | `logs` | directory replayed by `calibrate` / `run` (one file = one device) |
| `REGISTRY_PATH` | `artifacts/registry.json` | template registry written by `calibrate` |
| `LLM_BACKEND` | `offline` | `offline`, `hf`, `llm:<provider>[:<model>]`, `llm:web:<site>` |
| `WEB_BROWSER` | `camoufox` | browser for `llm:web:*`: `camoufox` or `chromium` |
| `QUEUE` | `memory` | unresolved queue: `memory` or `kafka` |
| `CLICKHOUSE_*` | see file | store connection, plus a read-only user for Query/Ask |
| `MONGO_*` | see file | store connection, plus a read-only URI for free-form queries |
| `KAFKA_*` | see file | broker, topic, consumer group, partition key |
| `LOGPIPE_ADMIN_PASSWORD` | unset | UI password; unset = generated and printed on first start |

## Services

```sh
make docker-up                      # ClickHouse + Kafka, waits until healthy
docker compose --profile mongo up -d mongo   # MongoDB instead of ClickHouse
make docker-down
```

The memory store and memory queue need nothing running, and all unit tests use them.

## Optional: a web chat as the LLM

No API key needed: logpipe drives your own signed-in Gemini, Claude or ChatGPT tab.

```sh
make web-setup                                   # Camoufox once (or BROWSER=chromium to use yours)
make web-login SITE=gemini BROWSER=chromium      # sign in, in a visible window
make web-check SITE=gemini BROWSER=chromium      # headless check
make web-ask   SITE=gemini BROWSER=chromium PROMPT="Reply with OK."
```

The scraper answers cookie banners with **Reject all**. Gemini also answers signed out.
Runs through a web chat are not reproducible, because the site picks the model.

## Optional: hosted API

Set `LLM_BACKEND=llm:anthropic` (or `openai`, `gemini`, `groq`) and the provider's usual API-key
variable. Add a model with a third part: `llm:anthropic:<model>`.

## Check the install

```sh
make test-fast                    # unit tests without model training
uv run logpipe demo --port 8000   # any flag skips the questions; open http://localhost:8000
```
