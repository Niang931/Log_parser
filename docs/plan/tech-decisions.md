# Technical decisions

## Technical decisions
| Decision | Choice | Why | Rejected |
|---|---|---|---|
| Agent language | Go module `agent/`, deps: `gopkg.in/yaml.v3`, `github.com/spf13/cobra`, `github.com/fsnotify/fsnotify`; docker via Engine HTTP API over the unix socket (stdlib, no docker SDK); journald via `journalctl -o json -f` subprocess | static binary, ARM gateways, tiny image | Python (runtime on hosts), Rust (slow iteration) |
| Config | `/etc/logpipe-agent/agent.yaml` + `conf.d/*.yaml` (`logs:` list), `${ENV}` expansion; server-managed sources only in `conf.d/remote.yaml` | Datadog shape, config-mgmt friendly, no merge fights | single file |
| Multiline | per-source `multi_line` rule (start-pattern regex) + `auto_multi_line_detection` (score first lines against a fixed set of timestamp regexes, pick best); flush on next start line or 1s timeout; max 256KB event | Java traces stay one event | — |
| Offsets | JSON registry `run/positions.json` keyed by path+inode; rotation = inode change or size < offset | restart-safe, no dupes | — |
| Transport | `POST /intake/v1/logs` gzip JSON `{events:[{ts,host,service,source,tags,message}]}`, `X-API-Key`; batch 500 lines / 1s; retry exp. backoff; disk buffer `buffer/` capped (default 100MB, drop oldest) | proxy-friendly, survives outages | OTLP/gRPC, raw syslog |
| Heartbeat / remote config | every 15s `POST /intake/v1/heartbeat` with version, uptime, effective sources, per-source stats; response carries remote-config hash; agent `GET /intake/v1/config` on change → writes `remote.yaml` → hot reload | one round-trip, no push socket | websockets |
| Server | `logpipe/server/` FastAPI + uvicorn, `logpipe server` subcommand; intake → `LogEvent` (gains host/service/source/tags, `device_id`=host) → existing `Pipeline.process` on one ingest thread fed by a bounded queue (backpressure → 429) | reuse whole parsing/learning stack | extending stdlib demo server |
| Storage | ClickHouse only: ALTER `iot_logs` add `host, service, source, tags`; new tables `agents`, `agent_remote_configs`, `api_keys` (sha256), `streams`, `warnings`, `jobs`, `templates_meta` (count, first_seen, seen_by_model per stream); `MemoryStore` mirrors for tests | per earlier infra decision | Postgres/SQLite |
| Jobs | in-server background thread, one job at a time, progress written to `jobs`; UI polls | single-VM deployment | Celery/separate worker |
| Calibration | sample per source via `deepparse.utils` `entropy_greedy_sample` up to cap → `calibrate()`; recalibration = FrozenParser on new sample, mine only unmatched lines, `SchemaRegistry.publish` (append) → backfill | stable IDs, no overwrite | full re-mine |
| LogBERT | per stream: `train_logbert` + `tune_thresholds` on that stream's capped windows, saved `artifacts/models/<stream>/`; `StreamingDetector` per stream; scheduler thread checks hourly for streams due (N days + pending unseen templates) | per user spec | auto-train always |
| Web | Vite + React + TS + TanStack Query (poll 2s) + Recharts + react-router; hand-written CSS tokens, light/dark; build → copied into `logpipe/server/static` | small, no SSR | Next.js, shadcn |
| Routes | `/` React (Setup, Overview, Live logs, Templates, Query, Stats, Alerts, Warnings, Streams, Agents, Agent detail, API keys, Jobs), `/demo` existing `index.html` + its `/api/*` stays on `logpipe demo` (server links to it) | demo untouched | port demo |
| Query | read-only: only `SELECT`/`WITH`, ClickHouse `readonly=1` setting, row limit | safe SQL console | — |
| Docker | agent: multi-stage → `gcr.io/distroless/static`, multi-arch; server: Python image + built web; compose adds `server` + `agent` (tails `examples/logs`); GH Actions: Go test + vet, web build, push both to GHCR on main/tag | per CI/CD decision | — |
| Git | `feat/agent-platform` off `feat/interactive-setup`; sub-branches `feat/agent-collector`, `feat/agent-server`, `feat/agent-web`, `feat/agent-docker` merged back; commit per working step; no Claude attribution | user | — |

## Implementation order
1. Branch; write `docs/plan/{persona,features,tech-decisions}.md`; commit.
2. Agent: config + CLI (track/untrack/list, config get/set/edit, check) → file tailer + offsets → rules (multiline, auto, include/exclude, mask) → shipper + buffer → heartbeat/remote config → journald, docker, syslog → `status`. Go tests per step (`go test ./...`), commit each.
3. Server: schema migrations + memory mirrors → API keys → intake + heartbeat + config endpoints → `/api` read endpoints → streams/warnings/jobs/calibration/recalibration/retraining scheduler. pytest per step.
4. Web: shell + nav + warning badge → Setup guide → Overview/Live/Templates → Query/Stats/Alerts → Agents/detail/remote-config editor/API keys → Streams/Warnings/Jobs.
5. Docker + compose + CI workflow + agent README.
6. End-to-end check; merge sub-branches into umbrella; Trello cards (green infra: agent/docker/CI; yellow: server/UI; orange: calibration/streams; blue: LogBERT per stream) → Done with branch/commit comments.

## Verification
- `cd agent && go test ./...` (multiline, rotation, offsets, rules, config round-trip, CLI edits).
- `uv run pytest -m "not integration"` (intake, auth, API, calibration append, warnings, scheduler) + `-m integration` with ClickHouse.
- `docker compose up` → open `http://localhost:8080`, run the setup guide with the compose agent, append a Java stack trace to `examples/logs/app.log` → appears as one event in Live logs; add a new tracked file via `logpipe-agent track` → warning appears; push remote config from the UI → agent writes `remote.yaml` and status shows the new source; `/demo` link opens the existing demo.
- `npm run build && npm run lint` in `web/`.
