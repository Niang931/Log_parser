# MongoDB

logpipe can keep everything in MongoDB instead of ClickHouse: parsed lines, alerts, and the control
plane (agents, keys, streams, jobs, drafts, plans, settings). The UI, the Query page, Ask/RCA and MCP
read it through MongoDB aggregations. See [Storage](../architecture/storage.md) for how it compares.

## Setup

1. Set the passwords in `.env` (the URIs must use the same ones; URL-encode `@ : / ?`):

    ```sh
    MONGO_ROOT_PASSWORD=...
    MONGO_APP_PASSWORD=...     # read-write user `logpipe`, used by the server
    MONGO_RO_PASSWORD=...      # read-only user `logpipe_ro`, used by the Query page's free-form pipelines
    MONGO_URI=mongodb://logpipe:<app password>@127.0.0.1:27017/?authSource=logpipe
    MONGO_READONLY_URI=mongodb://logpipe_ro:<ro password>@127.0.0.1:27017/?authSource=logpipe
    ```

2. Start it, either the whole platform or only MongoDB with the server from source:

    ```sh
    docker compose --profile platform-mongo up -d    # MongoDB + server + demo generator + agent
    # or
    docker compose --profile mongo up -d mongo
    uv run logpipe server --store mongodb            # or STORE=mongodb in .env
    ```

MongoDB runs with auth on and listens on localhost only. `docker/mongo-init.js` creates the two users
on the **first** start of an empty volume; if you change the passwords later, run
`docker compose down -v` (this deletes the data) or change them with `mongosh`.

For your own MongoDB (Atlas or self-hosted), skip compose: create a `readWrite` user and a `read`
user on the `logpipe` database (`MONGO_DB`), and point `MONGO_URI` / `MONGO_READONLY_URI` at it.

## Datadog

Two ways to see MongoDB in Datadog, both optional:

- **APM**: with `--datadog` and `DD_APM=1` (see [Datadog](datadog.md#3-self-monitoring-with-the-agent)),
  every query logpipe makes is a span, and MongoDB shows up as a dependency of `logpipe`.
- **MongoDB check**: the Agent reads MongoDB's own metrics (connections, operations, locks, replication).

    ```sh
    # .env: DD_MONGO_PASSWORD=...
    set -a; . ./.env; set +a
    ./docker/datadog/mongo-user.sh                        # creates the read-only `datadog` user
    docker compose --profile datadog up -d datadog-agent  # docker/datadog/mongo.d turns the check on
    ```

    The metrics appear under `mongodb.*` and on Datadog's MongoDB dashboard. If MongoDB isn't
    running, the check only shows an error in `docker exec logpipe_datadog agent status`.

## Settings

| Variable | Default | |
|---|---|---|
| `STORE` | `memory` | `mongodb` to use it (or `--store mongodb`) |
| `MONGO_URI` | `mongodb://127.0.0.1:27017` | read-write connection |
| `MONGO_DB` | `logpipe` | |
| `MONGO_READONLY_URI` | | free-form Query pipelines run on this connection when set |
| `MONGO_ROOT_PASSWORD`, `MONGO_APP_PASSWORD`, `MONGO_RO_PASSWORD` | | compose only: the users created on first start |
| `DD_MONGO_PASSWORD` | | the Datadog Agent's `datadog` user |
