# Agent platform

Three parts turn logpipe from a replay tool into a live service:

| Part | Where | What it does |
|---|---|---|
| `logpipe-agent` | `agent/` (Go) | tails files, journald, docker containers and syslog on each host; multiline aggregation, filters and masking on the host; ships gzip batches with a disk buffer |
| `logpipe server` | `logpipe/server/` (FastAPI) | agent intake, API keys, remote config, the setup → calibration → per-stream LogBERT lifecycle, warnings, jobs |
| Operator UI | `web/` (React + Vite) | setup guide, overview, live logs, templates, query, statistics, alerts, warnings, streams, jobs, agents, API keys |

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
calibrating.

Without Docker:

```bash
uv run logpipe server --store memory              # API on :8080
cd web && npm ci && npm run dev                   # UI on :5173 (proxies /api) or `npm run build`
cd agent && go build -o logpipe-agent . && ./logpipe-agent -h
```

The agent's config format and CLI are documented in [agent/README.md](../agent/README.md).

## Learning lifecycle

1. **Setup.** Logs from every agent are stored right away as `PENDING` (visible in Live logs), but
   nothing is learned. The setup guide walks through: API key → install agents → track files →
   streams → calibrate.
2. **Calibration** (once). For each source (`host | service | source`), up to `sample_cap` lines
   (default 5000) are picked by entropy-greedy priority sampling from the 4× most recent lines, so
   a chatty source can't crowd out the rest. `calibrate()` builds registry v1, every stored
   `PENDING` row is back-filled, the rest go to online enrichment, and every stream trains its
   LogBERT.
3. **Streams.** A stream is a named list of `host`/`service`/`source` glob selectors. All streams
   share one template registry (stable ids); each trains its own LogBERT on capped per-source
   sequences of canonical template ids. `All logs` always exists.
4. **New sources** after calibration raise a `new_source` warning. Actions:
   - *Recalibrate*: sample the source, mine only the lines the current registry can't match and
     **append** their templates (`append_calibrate`; existing ids never change), back-fill, then
     **retrain** (replace) the models of the streams it feeds;
   - *Same as another source*: it joins that source's streams;
   - *Mark resolved*.
5. **New templates** (found online by the enrichment worker or by recalibration) get a stable id,
   a candidate state, an occurrence count, a first-seen time, and their rows stay in the store as
   training data. A stream whose model doesn't know a template raises a `novel_template` warning.
   This is a novelty signal, never an anomaly alert: thresholds are tuned with
   `novelty_ratio = inf`, so the model keeps scoring known events and only sequence or hypersphere
   outliers alert. Actions: retrain now, *same as another template* (registry alias), mark
   resolved.
6. **Retraining** runs on request, or automatically when a stream's model is older than its
   `retrain_days` (per stream, default from settings: 7) **and** it has open unseen-template
   warnings. Jobs run one at a time with a progress bar (Jobs page).

Without labelled incidents, a stream's alert threshold is the 99th percentile of the miss ratio on
its held-out normal windows (at least 0.3).

## Storage

- Logs: `iot_logs` gains `service`, `source`, `tags` (added with `ALTER … IF NOT EXISTS`);
  `device_id` holds the agent's host.
- Control plane: `control_docs`, one JSON document per `(kind, id)` (agents, API keys stored as
  sha256, remote configs, sources, templates' counters, streams, warnings, jobs, settings),
  `ReplacingMergeTree` keyed on a nanosecond version.
- Registry and models: `--data-dir` (default `artifacts/server`): `registry.json` and
  `models/<stream>/`.

With `--store memory` everything except the data dir is lost on restart.

## Intake API (agents)

All requests carry `X-API-Key`.

| Endpoint | Body / answer |
|---|---|
| `GET /intake/v1/ping` | `{ok, key}`: used by `logpipe-agent check` |
| `POST /intake/v1/logs` | gzip JSON `{"events": [{ts, host, service, source, tags, message}]}` → 202; 429 when the ingest buffer is full (the agent buffers and retries) |
| `POST /intake/v1/heartbeat` | agent status (version, sources with counters, effective config, shipper stats, `remote_hash`) → `{remote_hash}` |
| `GET /intake/v1/config?hostname=` | `{yaml}`: written to `conf.d/remote.yaml` when the hashes differ |

`ts` is the agent's read time with its local UTC offset: a timestamp without a zone inside the
line is read as that host's wall-clock time.

The UI's API is browsable at `/api/docs`.
