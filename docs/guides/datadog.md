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
| **Root cause out** | The incident's root-cause analysis starts on its own. With **Settings → Ask requires approval** on (the default), its rounds wait for a person, as every Ask does. The answer follows as a second event with the same aggregation key, so Datadog groups the two together. |
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

## Showcase

`logpipe showcase --datadog` (add `--auto` for hands-free) is the normal guided tour plus the Datadog parts:

| Act | What you see in Datadog |
|---|---|
| 1b | The Datadog Agent (Docker, `docker/datadog/shop.d`) ships the shop's files as host `shop-web-1`. The `logpipe pipeline` dashboard and the monitors from `docs/datadog/` are created if they don't exist yet. |
| 6 | The outage becomes an anomaly event, followed by its root-cause analysis once its rounds are approved. In `--auto` mode the tour approves them itself. |
| 7b | **Templates → Export to Datadog** creates the `logpipe (auto)` pipeline. |
| 7c | The tour sends the payload a monitor's webhook would. The answer comes back as an event. |

When you stop it, the showcase deletes the dashboard, monitors and pipeline it created. Events stay.

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
