# Set up the server

The server takes in logs from agents (and from anything that sends OTLP/HTTP to `POST /v1/logs`),
runs the operator UI and API on port 8080 and trains a LogBERT model per stream. It stores data in
ClickHouse by default, or in MongoDB. You can run the image we ship or build it from source.

Either way, start with the repository and a `.env`:

```sh
git clone git@github.com:Niang931/Log_parser.git && cd Log_parser
cp .env.example .env    # change the change-me passwords
```

=== "Shipped container"

    CI builds `ghcr.io/niang931/logpipe-server` and publishes it on every push to `main` (`latest`)
    and every `v*` tag. The UI is built into the image.

    ```sh
    docker compose pull server
    docker compose up -d clickhouse server
    docker compose logs server | grep -i password    # the first start prints the admin password
    ```

    Open <http://localhost:8080>, sign in and follow the setup guide (API key → install agents →
    track files → sample size → streams → review templates → train).

    | Want | Run |
    |---|---|
    | Also a demo log generator and an agent | `docker compose --profile platform up -d` |
    | MongoDB instead of ClickHouse | `docker compose --profile platform-mongo up -d` |
    | Build the image from your checkout | add `--build` |

    **Your own ClickHouse.** Run the image on its own and point it at your ClickHouse. The `-e`
    flags override the values in `.env`:

    ```sh
    docker run -d --name logpipe-server --restart unless-stopped -p 8080:8080 \
      --env-file .env -e CLICKHOUSE_HOST=clickhouse.example -e CLICKHOUSE_PORT=8123 \
      -v logpipe-server:/app/artifacts \
      ghcr.io/niang931/logpipe-server:latest
    ```

    Templates and models live in `/app/artifacts`, and logs and the admin password in ClickHouse.
    Keep both volumes when you upgrade (`docker compose pull server && docker compose up -d server`).

=== "From source"

    Needs [uv](https://docs.astral.sh/uv/), Node 20+ (to build the UI) and Docker (for ClickHouse).
    See [Install](install.md) for details.

    ```sh
    make setup                                   # uv sync --group dev + .env
    cd web && npm ci && npm run build && cd ..   # builds the UI into logpipe/server/static
    docker compose up -d clickhouse              # or skip it and use --store memory
    uv run logpipe server --store clickhouse     # the first start prints the admin password
    ```

    The server reads ClickHouse settings (`CLICKHOUSE_HOST`, `CLICKHOUSE_PORT`, user and password)
    from `.env`. Useful flags:

    | Flag | Default | |
    |---|---|---|
    | `--store` | `memory` | `clickhouse` or `mongodb` to keep data across restarts |
    | `--port` / `--host` | `8080` / `0.0.0.0` | |
    | `--data-dir` | `artifacts/server` | registry and per-stream models |
    | `--llm` | `LLM_BACKEND` | masks, key names and alias checks |

    For UI work, run `cd web && npm run dev`. It serves on :5173 and proxies `/api` to the server.

## Admin password

If `LOGPIPE_ADMIN_PASSWORD` is unset, the first start generates a password, prints it once and
stores its hash in the store (ClickHouse or MongoDB). You can change it under **Settings → Password & session**.
If you forget it, set `LOGPIPE_ADMIN_PASSWORD` and restart: a password from the environment always
wins.

## Next

[Install the agent](agent.md) on the hosts whose logs you want, using an **Agent** key from
**Settings → API keys**. To connect an AI assistant, see [MCP](../architecture/mcp.md).
