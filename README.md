# Log Parser 💅💅💅

A log ingestion and parsing tool with anomaly detection and template extraction.

IoT log lines are parsed on a sub-millisecond hot path against a frozen Drain tree. Lines
that match no known template are stored as `PENDING`, and a background enrichment loop
learns templates for them and backfills the stored rows. Per-device sequences of template
ids are scored by LogBERT.

```
source ──► FrozenParser(vN) ──► hit:  template_id, payload, FINAL ──┬─► store (ClickHouse / memory)
           (match-only Drain)    miss: UNK, PENDING ───────────────┤
                                                                   ├─► detector: reorder buffer → windows → LogBERT
                                                                   └─► enrichment worker:
                       shadow Drain → validate → name keys → alias check → registry vN+1
                       → hot-swap parser → re-parse buffer → backfill rows (higher version)
```

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Docker (for ClickHouse; the in-memory store works without it)

## Setup

```sh
cp .env.example .env
# edit .env if needed
uv sync --group dev
```

## Quick start

```sh
uv sync --group dev
uv run logpipe        # or `logpipe start` / `make start`
```

It asks what to run (the browser demo, the pipeline on your own logs, or the server for
logpipe-agent hosts with its operator UI), which queue (in-memory,
or Kafka, which it starts in Docker for you), who names keys and confirms re-associations
(offline rules, a signed-in web chat, or a hosted API), and the few settings that matter, then
prints the equivalent commands and runs them. If you pick a web chat it can open the sign-in
window too. The answers are saved in `.logpipe/start.json` and become the defaults next time;
every question is still asked, so a repeat run is just pressing Enter.

## Running

Set `LOGS_PATH` in `.env` to point to the directory containing your log files (default: `logs/`).
Each file is treated as one device.

```sh
make calibrate  # learn templates + key names from history -> registry v1 (+ .report.json)
make train      # train LogBERT on per-device windows of the history
make run        # full pipeline: parse, store, enrich PENDING lines, detect anomalies
```

Try it end to end on a synthetic fleet with `make demo`, or step by step:

```sh
uv run logpipe gen-iot --out artifacts/fleet
uv run logpipe calibrate --logs artifacts/fleet/history
uv run logpipe train-logbert --logs artifacts/fleet/history
uv run logpipe run --logs artifacts/fleet/test              # STORE=clickhouse after make docker-up
```

### Browser demo

```sh
uv run logpipe demo        # asks: queue, store, LLM, masks, size, port; then open the page
```

Plain `logpipe demo` in a terminal asks its questions (your last answers are the defaults);
with any flag it runs straight away.

The calibration is saved in `artifacts/demo/` (one file per mask backend, fleet size and
seed) and reused on later runs, so only LogBERT is retrained at start-up (a few seconds);
`--recalibrate` redoes it. This matters with a web LLM, where calibrating asks the site to
name the keys of every template (about a minute each). Gemini answers signed out, so
`--llm llm:web:gemini` needs no login; `--browser chrome` uses the chromium on your PATH,
while camoufox needs `uv run camoufox fetch` once (and again after camoufox upgrades).

Builds a small synthetic fleet, calibrates, trains a tiny LogBERT and tunes its thresholds
(about 10 seconds), then generates the test stream live and runs it through the whole system.
The **System** tab draws every stage with live counters: frozen Drain → hit/miss → unresolved
queue → shadow Drain → LLM (key names, alias check) → validate → alias check → registry vN+1 →
hot-swap and backfill, plus LogBERT → alerts. Its timeline lists true positives, false positives
and missed anomalies, firmware-2 template shifts (with whether each was re-associated to an
existing template and why), requeues and dead-letters. Clicking an event or a line lights up
the path that line took. The header has pause, step and speed; the panel beside the graph
triggers the firmware rollout now, injects an anomaly into a device's next episode, and
restarts the session: the same stream or a new one replayed from the first line with an empty
store and queue, either from calibration or keeping the templates learned so far (so the
firmware-2 lines that missed the first time now hit). The restart also picks the queue
(memory or Kafka), the store (memory or ClickHouse) and the LLM that names keys and confirms
re-associations (offline rules, a web chat such as `llm:web:gemini`, or a hosted API), so the
same stream can be compared across setups; Kafka and ClickHouse are started in Docker when
picked. Calibration and LogBERT are kept across restarts; your rollout and inject clicks are
not replayed. The other tabs show
the stream with a per-line trace, the registry, a SQL query per template and charts.

| flag | effect |
|---|---|
| `--queue kafka` | the Kafka broker (started in Docker if needed, a fresh topic per session) instead of the in-memory queue |
| `--store clickhouse` | also write every row to ClickHouse (database `logpipe_demo`, emptied per session); the Query tab then runs on it |
| `--llm llm:web:gemini` | key names and alias confirmation by your signed-in web chat (default: offline rules, which never confirm a wording change) |
| `--registry demo_registry.json` | skip calibration and load masks made by the fine-tuned model on Colab (section 4 of `notebooks/colab_finetune_eval.ipynb`, i.e. `logpipe demo --llm hf --save-registry ...`) |

`--rate`, `--port`, `--devices`, `--test-hours` and `--epochs` change the setup; the charts load
Chart.js from a CDN.

The LLM backend is set by `LLM_BACKEND`:

- `offline` (default): deterministic, no network.
- `hf`: DeepSeek-R1-8B with the fine-tuned LoRA adapter (`HF_ADAPTER`, default
  `MThien213/deepparse-iot-r1-8b`) in 4-bit, for masks only; needs a GPU (Colab).
- `llm:<provider>[:<model>]`: a hosted provider through `deepparse.providers`, e.g.
  `llm:anthropic`. API keys come from the environment.
- `llm:web:<site>`: drives your own signed-in Claude, ChatGPT or Gemini web app in a browser,
  no API key. Runs are not reproducible, since the site picks the model.

Any make target takes `LLM=...` to override it for one run. To set up the web backend:

```sh
make web-setup                 # download Camoufox once
make web-login SITE=gemini     # sign in in a visible window; claude | chatgpt | gemini
make web-check SITE=gemini     # confirm the saved login still works headless
make web-ask SITE=gemini PROMPT="Reply with OK."
make eval-iot LLM=llm:web:gemini
```

To drive the Chromium on your `PATH` instead of Camoufox, add `BROWSER=chromium` to each
command (or set `WEB_BROWSER=chromium` in `.env`). Each browser keeps its own profile, so sign
in with the same browser you evaluate with, e.g.
`make web-login BROWSER=chromium && make eval LLM=llm:web:gemini BROWSER=chromium`.

### Showcase: the whole product in one command

```sh
uv run logpipe showcase          # asks store/LLM like `logpipe start`; --no-ask for memory + offline
uv run logpipe showcase --auto   # does every act itself (hands-free demo)
```

It starts the server, a generated web shop's logs (nginx, a Java API with stack traces, Postgres)
and a real `logpipe-agent` (built from `agent/` with Go, run as you, no sudo), then walks through
eight acts, each telling you what to do in the UI and waiting until it happened (Enter lets it do
it instead): connect the agent, draft templates, review them and write a parsing rule, calibrate
and train, explore (Log Explorer, Templates, Query builder), a payment-gateway outage that LogBERT
flags, and a new service whose templates are drafted for review by themselves. Files go to
`artifacts/showcase/`; Ctrl+C stops everything.

### Agents, server and operator UI

Ship live logs from your hosts with the Go collector, and watch them in the React UI:

```sh
docker compose --profile platform up --build   # ClickHouse + server + demo agent, UI on :8080
```

See [docs/agent-platform.md](docs/agent-platform.md) for the learning lifecycle (setup guide,
template review, streams, maintenance, retraining) and [agent/README.md](agent/README.md) for the
agent's Datadog-style config and CLI.

### AI assistant (MCP)

The server exposes an MCP endpoint at `/mcp`. Claude Code and other MCP clients can use it to
search logs, run read-only queries, investigate incidents and pre-review template batches.

1. In the UI, open **Settings → API keys** and create a key with use **AI assistant**. Each person
   should get their own key, so the UI shows whose assistant wrote a note or decided a draft.
2. Connect Claude Code:

   ```sh
   claude mcp add --transport http logpipe https://<your-server>/mcp \
     --header "Authorization: Bearer lp_..."
   ```

3. Ask in plain words, e.g. "investigate the latest open incident". You can also use the
   `investigate_incident` and `review_batch` prompts.

The `ask` tool puts a question to logpipe's own root-cause analysis and returns its ranked
report: when it broke, the candidate lines scored by anomaly × rarity × proximity with the reason
for each score, and the evidence. Its queries are read-only, so the tool approves them itself.

An assistant key can only read, save queries, write incident notes, and edit, merge, approve or
reject drafts. It can't change an incident's status or apply a review batch: a person does that
in the UI. Serve `/mcp` over HTTPS only. On MongoDB, also set `MONGO_READONLY_URI` so that
`run_query` runs as a read-only database user.

## Calibration and evaluation

`logpipe calibrate` reports **holdout coverage** for every run: the share of the newest 20%
of lines that a registry mined from the older 80% resolves without going `PENDING`. With
`--dataset <DeepParse dataset dir>`, it also reports DeepParse's **GA/PA** together with the
plain DeepParse configuration as a baseline. `--tune` grid-searches Drain depth and
similarity threshold.

| command | what it measures |
|---|---|
| `make eval-parsing` | GA/PA on the 14 Loghub-2k systems through the calibration path, vs. DeepParse |
| `make eval-alias` | alias decisions on labelled template pairs (same event vs. opposite states) |
| `make eval-iot` | end-to-end on the synthetic fleet with a firmware rollout and injected anomalies, plus ablations |
| `make eval-hdfs` / `make eval-bgl` | LogBERT on CPU-sized HDFS/BGL subsets: `[UNK]` / `[MASK]` / drop, novelty on/off |
| `make eval-rcaeval` | root cause on [RCAEval](https://github.com/phamquiluan/RCAEval) failure cases, scored by RCAEval (AC@k, Avg@k) |

`make eval` runs all of them and then `make eval-report`. Results go to `artifacts/eval/`
(`EVAL_OUT=...` to change); the latest numbers and findings are in
[docs/evaluation.md](docs/evaluation.md). Training benchmarks take `EPOCHS=` and `THREADS=`.

`make eval-rcaeval DATASET=re3-ob` exports the dataset as RCAEval's blind task view, answers each
case and scores the answers with `rcaeval-agent evaluate`. RCAEval runs through `uvx`, so it isn't
a dependency. Each case gets a fresh server that calibrates on the logs before the fault, then
reads the rest. `VIA=direct` (the default) asks the RCA pipeline, with and without LogBERT and
semantic search. `VIA=mcp` serves the case and lets Claude Code investigate it over `/mcp`, with the
`ask` tool and without it; that path costs LLM tokens, so start with `LIMIT=3`.

```sh
make eval-rcaeval DATASET=re3-ob                                          # rules only
make eval-rcaeval DATASET=re3-ob LLM=llm:web:gemini BROWSER=chromium      # + Gemini (web)
make eval-rcaeval DATASET=re3-ob VIA=mcp                                  # Claude Code over /mcp
```

Runs are cached in `artifacts/eval/rcaeval/`: an answered case and a saved LLM reply are reused,
so a stopped or repeated run only does what's missing, and a failed case is retried.
`FRESH=1` redoes everything. Every report, LLM prompt and reply, and agent stream (tool calls
and results) is kept in `artifacts/eval/rcaeval/transcripts/`. Logs only: the datasets with logs are `re2-ob`, `re2-ss`, `re2-tt`,
`re3-ob` and `re3-tt`, and only the service level is scored (the indicator level needs metrics).
Resource faults (`re2`: cpu, mem, disk, delay, loss) leave few traces in logs; the code-level
faults of `re3` are the fair test. The cases hold about 10 minutes of logs before the fault, so
`LOGPIPE_RCA_BEFORE_MINUTES` and the other `LOGPIPE_RCA_*` settings change the window to fit.

Datasets go to `artifacts/datasets/`. The eval targets fetch what they need, or download
up front with `make data` (all), `make data-loghub`, `make data-hdfs` or `make data-bgl`.

## Other commands

```sh
make help              # every target
make setup             # uv sync + .env from the example
make test              # all tests except integration
make test-fast         # skip slow (model-training) tests
make test-integration  # ClickHouse tests (starts the container)
make lint              # ruff
make typecheck         # pyright
make check             # lint + typecheck + test
make docker-up         # start ClickHouse and wait until healthy
make docker-down
```

---

## Acknowledgements

This project incorporates code from [Deep Parse](https://github.com/NightBaRron1412/DeepParse.git)
by [NightBaRron1412], licensed under the Apache License.
Changes have been made to fit this project.

The anomaly detector is adapted from [LogBERT](https://github.com/HelenGuohx/logbert)
(MIT License, Copyright (c) 2023 Helennnnn), described in
Guo, Yuan and Wu, *LogBERT: Log Anomaly Detection via BERT*, IJCNN 2021
([arXiv:2103.04475](https://arxiv.org/abs/2103.04475)). Its model code derives from
[BERT-pytorch](https://github.com/codertimo/BERT-pytorch) (Apache License 2.0).
Evaluation data comes from [Loghub](https://github.com/logpai/loghub).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for licenses and the list of changes.
