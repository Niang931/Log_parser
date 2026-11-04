# Features

## Feature list (each → persona problem)
Agent (Go, `agent/`)
1. Config loader: agent.yaml + conf.d/*.yaml, `${ENV}` expansion, validation → configure hosts with standard tooling
2. File source: globs, rotation/truncation, persisted offsets → no lost/duplicate lines across restarts
3. journald source (`journalctl -o json -f` subprocess) → systemd services
4. Docker source (docker socket, container stdout/stderr, label filter) → containerised apps
5. TCP/UDP syslog listener → devices that only speak syslog (IoT)
6. Processing rules: multi_line, auto multiline, exclude/include_at_match, mask_sequences → stack traces stay one event; secrets never leave host
7. Shipper: batched gzip HTTP POST, `X-API-Key`, retry+backoff, bounded on-disk buffer → survives server outages
8. Heartbeat + config report + remote config poll → `conf.d/remote.yaml`, hot reload → fleet visibility + push config
9. CLI: run, track/untrack/list, config get/set/edit (validate on save), status, check → manage without hand-editing YAML (install-service cut)
10. Docker image (scratch/distroless, multi-arch) + one-liner → fast trial

Server (Python, `logpipe/server/`)
11. `logpipe server` FastAPI: `/intake/v1/logs`, `/intake/v1/heartbeat`, `/intake/v1/config` → receive logs
12. Intake → `Pipeline.process` (LogEvent gains host/service/source/tags) → existing parsing/templates/anomaly
13. Schema: host/service/source/tags columns on iot_logs; `agents`, `agent_remote_configs`, `api_keys` tables (ClickHouse + memory store)
14. API keys: create/revoke, hashed → only known agents ingest
15. `/api/*` for UI: overview, logs (filters), templates, query (read-only SQL), stats, alerts, agents, agent config, keys
16. Serve React build at `/`, demo `index.html` at `/demo`

Web (React, `web/`)
17. Overview, Live logs (filter host/service/status), Templates, Query, Statistics, Alerts
18. Agents list (status, version, lines/s, errors) + detail (tracked sources, effective config, remote-config YAML editor) + API keys page

Learning lifecycle (user-specified)
21. Setup guide (UI wizard): API key → install agents (one-liner) → confirm agents + tracked sources reporting → set per-source cap (default 5k) → define streams → start calibration. No calibration before the user finishes the guide; logs arriving meanwhile are stored raw (PENDING/UNK) and backfilled after.
22. Calibration: per-source priority sampling (`entropy_greedy_sample`) up to the cap → existing `calibrate()` → registry v1 → backfill. Progress bar.
23. Log streams: named sets of (host, service, source) selectors, default stream "all"; one global Drain registry, one LogBERT per stream trained on capped per-source windows.
24. New source after calibration → warning "needs recalibration". Actions: Recalibrate (Drain **append**: only lines the frozen parser can't match are mined, new templates published with new IDs; LogBERT for affected streams **overwritten**), Mark resolved, "Same as source X" (joins X's streams).
25. Novel templates: stable ID, candidate state, occurrence count, first-seen, buffered sequences for training, flagged unseen-by-LogBERT. Shown as a separate novelty signal (existing `novelty_ratio`), never auto-flagged as anomalous. Listed in Warnings; actions: Mark resolved, alias to existing template.
26. Retraining: manual button, or scheduled every N days per stream (default 7) only when unseen templates are pending. Jobs table with progress shown in UI.

Ops
19. Server Dockerfile; compose: `server` + sample `agent` tailing `examples/logs`; GH Actions builds/pushes agent+server images to GHCR, Go tests in CI
20. Docs (agent README with config reference), Trello cards → Done
