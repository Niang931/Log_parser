# Datadog

If you're already on Datadog, logpipe reads your logs from it and writes what it finds back.

```sh
# .env: DD_API_KEY, DD_APP_KEY, DD_SITE, and DD_PUBLIC_URL if the server is reachable from outside
uv sync --extra datadog
uv run logpipe server --datadog-pull 'service:shop*'
```

| | What happens |
|---|---|
| **In** | Every 15 s (`DD_PULL_EVERY_S`), the server pulls logs that match the query from Logs Search, one page of 1000 per tick, which keeps it under the search rate limit. The logs go through the same setup as an agent's: drafted, reviewed, trained. |
| **Anomalies out** | Each new incident becomes a Datadog event tagged `source:logpipe`, `host`, `service` and `logpipe_incident`. The event links to the logs around the incident in Datadog and to the incident in logpipe (`DD_PUBLIC_URL`). Incidents that existed when the server started are not posted. |
| **Root cause out** | The incident's root-cause analysis starts without waiting for approval, because its queries are read-only. The answer follows as a second event with the same aggregation key, so Datadog groups the two together. |
| **Grok out** | **Templates → Export to Datadog** writes every template's parsing rule into a `logpipe (auto)` log pipeline. Matchers Datadog lacks are translated (`notspace` → `notSpace`, `quoted` → `quotedString`, `path` → `notSpace`, `timestamp` → a regex). Exporting again replaces the whole pipeline. |
| **Monitor webhook** | A Datadog monitor alert runs a root-cause analysis around the alert's time and posts the answer as an event. |

## Monitor webhook

1. In logpipe: **Settings → API keys**, create an **Agent** (intake) key.
2. In Datadog: **Integrations → Webhooks → New**
    - URL: `https://<server>/intake/v1/datadog/webhook`
    - Custom headers: `{"X-API-Key": "lp_..."}`
    - Payload: `{"title": "$EVENT_TITLE", "date": "$DATE", "tags": "$TAGS", "host": "$HOSTNAME", "id": "$ALERT_ID"}`
3. Mention `@webhook-<name>` in a monitor's message.

If `host` or `service` (taken from the tags) isn't known to logpipe, the analysis covers every service.
To expose a server running on a laptop: `cloudflared tunnel --url http://localhost:8080`.

## Demo

`make demo-datadog-loop` (`NORMAL=` sets the seconds of normal traffic to learn, default 300) runs the whole loop against your org:

1. A web shop's logs (nginx, a Java API, Postgres) are written to `artifacts/datadog-loop/logs`. The Datadog Agent (`docker compose --profile datadog`, config in `docker/datadog/shop.d`) ships them to Datadog.
2. `logpipe server --datadog-pull 'service:(nginx OR shop-api OR postgres)' --datadog` pulls them back, drafts and approves templates, trains LogBERT, and exports the grok pipeline.
3. A payment-gateway outage starts. It shows up in Datadog's event explorer (`source:logpipe`) as an anomaly event, followed by its root cause.

## Self-monitoring

With `--datadog`, logpipe's own stage metrics, anomaly logs and LLM spans go to the local Agent. A dashboard and monitors to import are in `docs/datadog/`. `make demo-datadog` shows a store getting slow and then failing.

## Settings

| Variable | Default | |
|---|---|---|
| `DD_API_KEY` | | events; turns the Datadog sync on |
| `DD_APP_KEY` | | logs search and pipelines |
| `DD_SITE` | `datadoghq.com` | e.g. `datadoghq.eu`, `us5.datadoghq.com` |
| `DD_PULL_QUERY` | | same as `--datadog-pull` |
| `DD_PULL_EVERY_S` | `15` | |
| `DD_PUBLIC_URL` | | base of the links to logpipe in events |
