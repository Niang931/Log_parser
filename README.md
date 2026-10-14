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

## Running

Set `LOGS_PATH` in `.env` to point to the directory containing your log files (default: `logs/`).
Each file is treated as one device.

```sh
make calibrate  # learn templates + key names from history -> registry v1 (+ .report.json)
make train      # train LogBERT on per-device windows of the history
make run        # full pipeline: parse, store, enrich PENDING lines, detect anomalies
```

Try it end to end on a synthetic fleet:

```sh
uv run logpipe gen-iot --out artifacts/fleet
uv run logpipe calibrate --logs artifacts/fleet/history
uv run logpipe train-logbert --logs artifacts/fleet/history
uv run logpipe run --logs artifacts/fleet/test              # STORE=clickhouse after make docker-up
```

The LLM backend is set by `LLM_BACKEND`:

- `offline` (default): deterministic, no network.
- `hf`: local DeepSeek.
- `llm:<provider>[:<model>]`: a hosted provider through `deepparse.providers`, e.g.
  `llm:anthropic`. API keys come from the environment.

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

Results go to `artifacts/eval/`; the latest numbers and findings are in [docs/evaluation.md](docs/evaluation.md). Datasets are downloaded to `artifacts/datasets/` on first use.

## Other commands

```sh
make test              # unit tests
make test-integration  # ClickHouse tests (starts the container)
make lint              # lint with ruff
make docker-up         # start ClickHouse
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
