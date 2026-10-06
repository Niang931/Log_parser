# Log Parser 💅💅💅

A log ingestion and parsing tool with anomaly detection and template extraction.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Docker (for the database)

## Setup

```sh
cp .env.example .env
# edit .env if needed
uv sync --group dev
```

## Running

Set `LOGS_PATH` in `.env` to point to the directory containing your log files (default: `logs/`).

```sh
make calibrate  # learn templates + key names from historical logs -> registry v1
make run        # replay LOGS_PATH through the frozen parser (FINAL / PENDING per line)
```

Try it on the bundled fixtures:

```sh
uv run logpipe calibrate --logs tests/fixtures/iot
uv run logpipe replay --logs tests/fixtures/iot
```

## Other commands

```sh
make test       # run tests
make lint       # lint with ruff
make docker-up  # start ClickHouse
make docker-down
```

---

## Acknowledgements

This project incorporates code from [Deep Parse](https://github.com/NightBaRron1412/DeepParse.git)
by [NightBaRron1412], licensed under the Apache License.
Changes have been made to fit this project.
