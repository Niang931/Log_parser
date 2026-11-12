# Commands

## `logpipe`

`uv run logpipe [--browser camoufox|chrome] <command> …`. `--browser` is global, so it goes
before the command, and is used only by `llm:web:<site>` backends. Every command has `-h`.

| Command | What it does | Main options |
|---|---|---|
| `start` (default) | asks what to run, starts services, runs it | — |
| `calibrate` | build registry v1 from historical logs, with a holdout-coverage report | `--logs`, `--registry`, `--llm`, `--sample`, `--depth`, `--similarity`, `--dataset` (GA/PA), `--tune` |
| `replay` | parse files with the frozen registry only, with no learning | `--logs`, `--registry` |
| `run` | full pipeline: parse → store → enrich → detect | `--store`, `--queue`, `--llm`, `--model`, `--enrich-every`, `--no-enrich`, `--rehydrate`, `--window`, `--stride` |
| `worker` | learning worker on the unresolved queue (Kafka) | `--queue`, `--store`, `--llm`, `--batch`, `--poll`, `--max-attempts`, `--max-cycles` |
| `train-logbert` | train LogBERT on per-device windows | `--logs`, `--out`, `--window`, `--stride`, `--hidden`, `--layers`, `--heads`, `--epochs` |
| `gen-iot` | write the synthetic IoT fleet (history / validation / test) | `--out`, `--devices`, `--history-hours`, `--test-hours`, `--seed` |
| `demo` | the pipeline live in a browser | `--queue`, `--store`, `--llm`, `--registry`, `--recalibrate`, size flags |
| `server` | agent intake + operator UI + learning lifecycle + MCP | `--store`, `--queue`, `--llm`, `--data-dir`, `--epochs`, `--demo`, `--host`, `--port` |
| `showcase` | guided tour on a generated web shop with a real agent | `--no-ask`, `--auto` |
| `bench` | how many lines/s one server sustains, with CPU and memory | `--store`, `--max-rate`, `--step-seconds`, `--out` |
| `fetch-data` | download eval datasets up front | `loghub2k`, `HDFS`, `BGL` |
| `eval <name>` | `parsing`, `logbert`, `iot`, `alias`, `report`, `gate` | `--out`, `--llm`, `--dataset`, `--epochs`, `--write-baseline` |

Eval modules that aren't `logpipe eval` subcommands run as modules:

```sh
uv run python -m logpipe.eval.rca_eval [--llm …] [--record f.json | --replay f.json]
uv run python -m logpipe.eval.rcaeval_bench artifacts/eval/rcaeval/tasks/re3-ob [--via mcp] [--llm …] [--fresh]
```

## `make`

`make help` lists every target. Make variables override for one run: `LLM=…`, `EPOCHS=…`,
`THREADS=…`, `BROWSER=chromium`, `STORE=…`, `EVAL_OUT=…`, `DATASET=…`, `VIA=…`, `LIMIT=…`,
`VARIANTS=…`, `FRESH=1`, `SITE=…`, `PROMPT=…`.

=== "Setup & services"

    | target | |
    |---|---|
    | `setup` | `uv sync --group dev`, create `.env` |
    | `start` | interactive start |
    | `docker-up` / `docker-down` | ClickHouse + Kafka |
    | `clean` | caches only (keeps datasets, models, results) |

=== "Pipeline"

    | target | |
    |---|---|
    | `calibrate` | registry v1 from `LOGS_PATH` |
    | `train` | LogBERT on `LOGS_PATH` |
    | `run` | full pipeline |
    | `run-fast` | fast path only, unknown lines to Kafka |
    | `worker` | learning worker on Kafka |
    | `replay` | frozen registry only |
    | `gen-iot` / `demo` | synthetic fleet / end to end on it |

=== "Evaluation"

    | target | |
    |---|---|
    | `data`, `data-loghub`, `data-hdfs`, `data-bgl` | datasets |
    | `eval` | parsing, alias, IoT, HDFS, BGL, then the report |
    | `eval-parsing` / `eval-alias` / `eval-iot` / `eval-hdfs` / `eval-bgl` | one benchmark |
    | `eval-rcaeval` | RCAEval: `DATASET`, `VIA=direct\|mcp`, `LLM`, `LIMIT`, `VARIANTS`, `FRESH=1` |
    | `eval-report` / `eval-gate` / `eval-baseline` | report, CI gate, record baseline |
    | `bench` | throughput (`STORE=clickhouse` to include writes) |
    | `retrain-data` / `retrain-eval` | mask-model fine-tune data and scoring |

=== "Web LLM"

    | target | |
    |---|---|
    | `web-setup` | Camoufox once (`BROWSER=chromium` uses yours) |
    | `web-login SITE=gemini` | sign in, visible window |
    | `web-check SITE=gemini` | check headless |
    | `web-ask SITE=gemini PROMPT=…` | one prompt |

=== "Quality"

    | target | |
    |---|---|
    | `test` | all but integration |
    | `test-fast` | no model training, no integration |
    | `test-slow` | only model-training tests |
    | `test-integration` | ClickHouse + Kafka (starts them) |
    | `lint` / `format` / `typecheck` / `check` | ruff, ruff --fix, pyright, all |
    | `docs` / `docs-build` | serve / build this site |
