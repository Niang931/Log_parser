# Datadog

If you're already on Datadog, logpipe reads your logs from it and writes what it finds back.
It can also send its own health to Datadog: metrics, logs, APM traces and LLM traces.

| Feature | What happens |
|---|---|
| **Logs in** | Every 15 s (`DD_PULL_EVERY_S`), the server pulls logs that match the query from Logs Search, one page of 1000 per tick, which keeps it under the search rate limit. The logs go through the same setup as an agent's: drafted, reviewed, trained. |
| **Anomalies out** | Each new incident becomes a Datadog event tagged `source:logpipe`, `host`, `service` and `logpipe_incident`. The event links to the logs around the incident in Datadog and to the incident in logpipe (`DD_PUBLIC_URL`). Incidents that existed when the server started are not posted. |
| **Root cause out** | The incident's root-cause analysis starts on its own. With **Settings → Ask requires approval** on (the default), its rounds wait for a person, as every Ask does. The answer follows as a second event with the same aggregation key, so Datadog groups the two together. |
| **Grok out** | **Templates → Export to Datadog** turns every template's parsing rule into a `logpipe (auto)` log pipeline: a grok parser, plus a date remapper when timestamps were found. See [Grok export](#grok-export). |
| **Monitor webhook** | A Datadog monitor alert runs a root-cause analysis around the alert's time and posts the answer as an event. |
| **Self-monitoring** | With `--datadog`: stage metrics (DogStatsD), anomaly logs, LLM Observability traces, and with `DD_APM=1` APM traces that draw logpipe on the Service Map. A dashboard and monitors to import are in `docs/datadog/`. |
| **MongoDB check** | The Agent can also watch logpipe's MongoDB (connections, operations, replication). See [MongoDB](mongodb.md#datadog). |

## Setup

### 1. Keys and site

1. In Datadog, **Organization Settings → API Keys**: create an API key (events, the Agent).
2. **Organization Settings → Application Keys**: create an application key (logs search, pipelines, dashboards).
3. Find your site in the browser's address bar: `app.datadoghq.com` is `datadoghq.com`,
   `us3.datadoghq.com` is `us3.datadoghq.com`, `app.datadoghq.eu` is `datadoghq.eu`.
4. Put them in `.env`:

```sh
DD_API_KEY=...
DD_APP_KEY=...
DD_SITE=us3.datadoghq.com
```

A wrong `DD_SITE` is the usual cause of `401 Unauthorized`: keys only work on their own site.

Then install the extra (`datadog`, `ddtrace`):

```sh
uv sync --extra datadog
```

### 2. Pull your logs and post incidents

```sh
uv run logpipe server --datadog-pull 'service:shop*'   # or DD_PULL_QUERY in .env
```

Any [Logs Search query](https://docs.datadoghq.com/logs/explorer/search_syntax/) works. Leave it out
and the server still posts incidents and root causes, it just doesn't pull. Set `DD_PUBLIC_URL` to
where the server can be reached (e.g. a tunnel: `cloudflared tunnel --url http://localhost:8080`),
so the links in events open logpipe.

### 3. Self-monitoring with the Agent

```sh
docker compose --profile datadog up -d datadog-agent   # reads DD_API_KEY and DD_SITE from .env
uv run logpipe server --datadog                        # also: logpipe run / worker --datadog
```

| Signal | Where in Datadog | Turned on by |
|---|---|---|
| Stage timings, lines, anomalies, errors (`logpipe.*` metrics) | Metrics, the dashboard | `--datadog` |
| logpipe's anomaly log lines (`.logpipe/dd-logs.jsonl`, tailed by the Agent) | Logs, `service:logpipe` | `--datadog` |
| Each LLM call: prompt, answer, tokens, latency | **LLM Observability → Traces**, app `logpipe` | `--datadog` |
| A span per request and per call it makes (MongoDB, ClickHouse, Datadog, HTTP LLMs) | **APM → Services → logpipe**, its Dependencies map | `--datadog` and `DD_APM=1` |

`DD_ENV` sets the `env` tag on the spans (e.g. `prod`). If the server itself runs in compose,
set `DD_AGENT_HOST=datadog-agent`.

### 4. Dashboard and monitors

`docs/datadog/` holds a dashboard and four monitors (anomaly spike, parse p99, pending ratio, stage
errors). Import them in **Dashboards → New → Import dashboard JSON** and **Monitors → New → Import**,
or let `logpipe showcase --datadog` create them. The dashboard has totals, the Service Map, request
and error rates and per-stage timings, with logpipe's incident events marked on the graphs.
`make demo-datadog` shows a store getting slow and then failing.

### 5. Monitor webhook

1. In logpipe: **Settings → API keys**, create an **Agent** (intake) key.
2. In Datadog: **Integrations → Webhooks → New**
    - URL: `https://<server>/intake/v1/datadog/webhook`
    - Custom headers: `{"X-API-Key": "lp_..."}`
    - Payload: `{"title": "$EVENT_TITLE", "date": "$DATE", "tags": "$TAGS", "host": "$HOSTNAME", "id": "$ALERT_ID"}`
3. Mention `@webhook-<name>` in a monitor's message.

If `host` or `service` (taken from the tags) isn't known to logpipe, the analysis covers every service.

## Grok export

**Templates → Export to Datadog** (needs `DD_APP_KEY`) writes one `logpipe (auto)` pipeline.
Exporting again replaces it, so export after the templates change (for example after a firmware
update rewords lines).

- **Matchers** Datadog lacks are translated: `notspace` → `notSpace`, `quoted` → `quotedString`, `path` → `notSpace`.
- **Timestamps** become `date("…")` with the exact format of the newest line the template parsed, and a date remapper makes them the log's official time. Without a sample line they fall back to a regex.
- **Filters** are added only when the sample value really is what the field name says: `useragent`, `url`, `querystring`, `json`, `keyvalue`.
- **Order**: Datadog uses the first rule that matches, so rules with the most fixed text come first.
- **Scope**: the pipeline's filter is limited to the services the templates came from, e.g. `service:(nginx OR shop-api)`.

## Showcase

`logpipe showcase --datadog` (add `--auto` for hands-free) is the normal guided tour plus the Datadog parts:

| Act | What you see in Datadog |
|---|---|
| 1b | The Datadog Agent (Docker, `docker/datadog/shop.d`) ships the shop's files as host `shop-web-1`. The `logpipe pipeline` dashboard and the monitors from `docs/datadog/` are created if they don't exist yet. The server runs with `DD_APM=1` and `DD_ENV=showcase`, so APM shows logpipe and what it calls. |
| 6 | The outage becomes an anomaly event, followed by its root-cause analysis once its rounds are approved. In `--auto` mode the tour approves them itself. **LLM Observability → Traces** shows each LLM step of the analysis. |
| 7b | **Templates → Export to Datadog** creates the `logpipe (auto)` pipeline. Lines in **Logs** now show the parsed fields. |
| 7c | The tour sends the payload a monitor's webhook would. The answer comes back as an event. |

When you stop it, the showcase deletes the dashboard, monitors and pipeline it created. Ones that
existed before are kept. Events stay.

## Settings

| Variable | Default | |
|---|---|---|
| `DD_API_KEY` | | events and the Agent; turns the Datadog sync on |
| `DD_APP_KEY` | | logs search, pipelines, dashboards |
| `DD_SITE` | `datadoghq.com` | your region, e.g. `datadoghq.eu`, `us3.datadoghq.com` |
| `DD_PULL_QUERY` | | same as `--datadog-pull` |
| `DD_PULL_EVERY_S` | `15` | |
| `DD_PUBLIC_URL` | | base of the links to logpipe in events |
| `DD_SERVICE` | `logpipe` | service name on metrics, logs and spans |
| `DD_AGENT_HOST` | `127.0.0.1` | `datadog-agent` if logpipe runs in compose |
| `DD_LOG_FILE` | `.logpipe/dd-logs.jsonl` | the file the Agent tails |
| `DD_LLMOBS_ML_APP` | `logpipe` | app name in LLM Observability |
| `DD_APM` | off | APM spans for the Service Map (needs `--datadog` and the Agent) |
| `DD_ENV` | | `env` tag on the spans |
| `DD_MONGO_PASSWORD` | | the Agent's MongoDB check, see [MongoDB](mongodb.md#datadog) |
