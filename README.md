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

```sh
make run
```

Set `LOGS_PATH` in `.env` to point to the directory containing your log files (default: `logs/`).

## Other commands

```sh
make test       # run tests
make lint       # lint with ruff
make docker-up  # start postgres
make docker-down
```

---

## Acknowledgements

This project incorporates code from [Deep Parse](https://github.com/NightBaRron1412/DeepParse.git)
by [NightBaRron1412], licensed under the Apache License.
Changes have been made to fit this project.
