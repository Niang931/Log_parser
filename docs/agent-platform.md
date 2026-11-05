# Agent platform

Three parts turn logpipe from a replay tool into a live service:

| Part | Where | What it does |
|---|---|---|
| `logpipe-agent` | `agent/` (Go) | tails files, journald, docker containers and syslog on each host; multiline aggregation, filters and masking on the host; ships gzip batches with a disk buffer |
| `logpipe server` | `logpipe/server/` (FastAPI) | agent intake, API keys, remote config, the setup → template review → per-stream LogBERT lifecycle, maintenance, jobs |
| Operator UI | `web/` (React + Vite) | setup guide, overview, log explorer, templates, template review, query, analytics, anomaly alerts, maintenance, streams, jobs, agents, API keys |

The developer demo (`logpipe demo`) is unchanged and is also mounted at `/demo/` by
`logpipe server --demo`.

## Try it

```bash
cp .env.example .env
docker compose --profile platform up --build
# open http://localhost:8080 and follow the setup guide
```

The `platform` profile starts ClickHouse, the server, a log generator (an app log with Java
stack traces plus an nginx access log) and an agent that tracks the app log. The agent uses
`LOGPIPE_BOOTSTRAP_KEY` (default `lp_compose_demo_key`), which the server accepts as an API key.
To see the new-source flow, push `/logs/access.log` to `compose-agent` from the Agents page after
setup: it is drafted for review under Maintenance once it has 500 lines.

Without Docker:

```bash
uv run logpipe server                             # asks store/queue/LLM/port, like `logpipe demo`
uv run logpipe server --store memory              # or pass flags: API on :8080
cd web && npm ci && npm run dev                   # UI on :5173 (proxies /api) or `npm run build`
cd agent && go build -o logpipe-agent . && ./logpipe-agent -h
```

The agent's config format and CLI are documented in [agent/README.md](../agent/README.md).

## Learning lifecycle

Nothing reaches the parser without a person approving it, and nothing that needs data starts
before there is enough of it.

1. **Setup** (UI wizard, one step per screen): API key → install agents → track files →
   sample size → streams → review templates → train. Logs are stored right away as `PENDING`
   (visible in the Log Explorer), but nothing is learned. Drafting needs at least 50 lines.
2. **Drafts.** For each source (`host | service | source`), up to `sample_cap` lines (default
   5000) are picked by entropy-greedy priority sampling from the 4× most recent lines, so a chatty
   source can't crowd out the rest. The first time, `calibrate()` mines masks, templates and field
   names; later, `append_drafts()` mines only lines the current registry can't match. Either way
   the result is a **batch of drafts**, not a registry: each draft shows how many of the sources'
   recent lines it matches and five examples.
3. **Review.** Per draft: approve, reject, edit the template text (`<*>` for a variable; the
   match count is re-checked live against the recent lines), rename its fields, or merge drafts
   of the same length (differing words become variables). *Apply* needs a decision on every
   draft: approved ones are published (existing ids never change), stored rows are back-filled,
   and the affected streams retrain. A later batch can also be skipped and left to the online
   learner.
4. **Streams.** A stream is a named list of `host`/`service`/`source` glob selectors. All streams
   share one template registry (stable ids); each trains its own LogBERT on capped per-source
   sequences of canonical template ids. `All logs` always exists.
5. **New sources** after setup collect lines, then are drafted automatically once they have
   `draft_lines` lines (default 500) or after `draft_after_minutes` (default 60) with at least 50.
   The batch waits under **Maintenance → Review templates**. Alternatives: *Draft now*, or *same as
   another source* (it joins that source's streams).
6. **Online templates.** Lines no template matches are still learned by the enrichment worker so
   parsing never waits; those templates go live at once and are listed under Maintenance, where
   they can be kept, corrected (text and field names, same id) or merged into another template.
7. **New to a model.** A stream whose model doesn't know a template gets a *retrain* item in
   Maintenance. This is a novelty signal, never an anomaly alert: thresholds are tuned with
   `novelty_ratio = inf`, so the model keeps scoring known events and only sequence outliers alert
   (those are the only things on the Anomaly alerts page).
8. **Training waits for data.** A model needs 8 windows (20 consecutive parsed lines from one
   source). A stream with fewer is marked *waiting for data* with a have/need meter instead of
   failing; housekeeping recounts every minute and trains it as soon as it has enough. Retraining
   also runs on request, or automatically when a model is older than its `retrain_days`
   (default 7) **and** has templates it hasn't seen.
9. **Jobs** (draft, apply, retrain) run one at a time. The Jobs page lists them with status,
   target, progress, duration and a step-by-step log; queued jobs can be cancelled and failed ones
   retried. What is waiting for data is shown above them.

Without labelled incidents, a stream's alert threshold is the 99th percentile of the miss ratio on
its held-out normal windows (at least 0.3).

## Parsing rules and linked templates

A template can carry a hand-written rule, written like Datadog's parsing rules:

```
%{timestamp:ts} UTC [%{integer:pid}] LOG: duration: %{number:duration_ms} ms statement: %{data:statement}
```

Matchers: `word`, `notspace`, `integer`, `number`, `ip`, `hostname`, `path`, `uuid`, `quoted`,
`timestamp`, `data` (anything, lazy) and `%{regex("…"):name}`. `%{matcher}` without a name matches
without extracting; whitespace matches any run of whitespace; the rule must match the whole line.
Every draft and template starts with a rule generated from its mined text. The editor re-runs the
rule on the recent lines as you type and shows a table: each sample line with what every field
extracted, plus the template's lines the rule would miss. Templates with a rule are tried before
the Drain tree. Saving a rule on a live template re-extracts its stored lines (a `reextract` job).

Templates can be **linked** as the same event (registry aliases): the anomaly models count a linked
template as the one it points to; each keeps its own rule and fields. Links come from a person
(Templates → Related templates, Maintenance → Same event as…) or from the enrichment worker's LLM
same-event check; the UI shows which, lists linked templates under the one they count as, shows them
in the Log Explorer's side panel, and can unlink them.

A source that appears after setup is held: its lines stay `PENDING` (visible in the explorer) and
are not handed to the online learner until its drafted templates are reviewed, so the review isn't
pre-empted. Lines no approved template matches are learned online after that.

## Query

The Query page has a **builder** (time range, filters on log attributes or extracted `@fields`, a
measure: count, distinct, avg/min/max/sum, p95 on ClickHouse; up to two group-bys; over time), which
the server turns into read-only SQL for ClickHouse or the memory store's SQLite and charts. It also
has a schema sidebar, **saved queries** (stored on the server, shared), **Ask in English** (the
configured LLM writes SQL you review before it runs; needs a non-offline `--llm`), and plain SQL.

## Storage

- Logs: `iot_logs` gains `service`, `source`, `tags` (added with `ALTER … IF NOT EXISTS`);
  `device_id` holds the agent's host.
- Control plane: `control_docs`, one JSON document per `(kind, id)` (agents, API keys stored as
  sha256, remote configs, sources, templates' counters, streams, warnings, draft batches and
  drafts, jobs, settings),
  `ReplacingMergeTree` keyed on a nanosecond version.
- Registry and models: `--data-dir` (default `artifacts/server`): `registry.json`,
  `models/<stream>/` and `review/<batch>.json` (the recent lines a batch's counts use).

With `--store memory` everything except the data dir is lost on restart.

## Intake API (agents)

All requests carry `X-API-Key`.

| Endpoint | Body / answer |
|---|---|
| `GET /intake/v1/ping` | `{ok, key}`: used by `logpipe-agent check` |
| `POST /intake/v1/logs` | gzip JSON `{"events": [{ts, host, service, source, tags, message}]}` → 202; 429 when the ingest buffer is full (the agent buffers and retries) |
| `POST /intake/v1/heartbeat` | agent status (version, sources with counters, effective config, shipper stats, `remote_hash`) → `{remote_hash}` |
| `GET /intake/v1/wait?hostname=&remote_hash=&timeout=25` | long-poll every agent keeps open: answers `{remote_hash}` the moment the server's remote config differs (the agent then heartbeats and applies it), else after `timeout`. While it's open the agent is online; once it closes and isn't reopened within 1s, offline |
| `GET /intake/v1/config?hostname=` | `{yaml}`: written to `conf.d/remote.yaml` when the hashes differ |

`ts` is the agent's read time with its local UTC offset: a timestamp without a zone inside the
line is read as that host's wall-clock time.

Heartbeats (every 15s) carry the counters; presence and config pushes ride the long-poll, so the
UI sees an agent come up, go down or apply a pushed config within about a second. The UI listens
on `GET /api/events` (server-sent events) and refetches on each change; polling stays as the
fallback. Agents older than the long-poll fall back to heartbeats (online for 45s after the last).

The UI's API is browsable at `/api/docs`.
