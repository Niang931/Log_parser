# Datadog in/out loop: technical decisions

## Technical decisions
| Decision | Choice | Why | Rejected |
|---|---|---|---|
| DD HTTP client | `httpx` (already installed through fastapi/mcp) in one new module `logpipe/datadog_api.py`: `search_logs(query, from, cursor)`, `post_event(...)`, `upsert_pipeline(name, rules)` | ~4 calls, no new dependency | `datadog-api-client` (heavy) |
| Settings | Add `DatadogSettings(BaseSettings)` to `logpipe/settings.py` with the `_env("DD_")` pattern used by the other classes. Fields: `service`, `site`, `agent_host`, `dogstatsd_port=8125`, `log_file=.logpipe/dd-logs.jsonl`, `llmobs_ml_app`, `api_key`, `app_key`, `pull_query`, `pull_every_s=15`, `public_url`. Add `datadog` to `Settings`. | fixes the ImportError; follows the codebase's single settings module | ad-hoc `os.environ` |
| Intake (F1) | `logpipe server --datadog-pull 'service:shop*'`. A step in `_housekeeping_loop` (`logpipe/server/core.py:2047`) every `pull_every_s` pages POST `/api/v2/logs/events/search` from a cursor time saved in `self.control` ("datadog" doc) and maps attributes `{message, host, service, source, tags, timestamp}` → the existing `ingest()` dict. | the ingest dict already matches DD fields; housekeeping already runs periodic tasks | DD-format intake endpoint plus Agent dual-ship (needs HTTPS reachable from the Agent, so too risky for the deadline) |
| Push (F2/F3) | Same housekeeping step: for each `self.incidents()` with no `dd_event` in its control doc, call `explain_incident(id, require_approval=False)`, then POST `/api/v1/events` (title, host/service tags, `aggregation_key=incident id`, links to `public_url/incidents/<id>` and a DD Logs Explorer URL for the window), and store `dd_event`. When the incident's `ask_plan` is done and `dd_rca` is unset, post the answer as a second event with the same aggregation_key. | incidents are computed on the fly, so an idempotent "posted" flag in the control store is the simplest hook; the API key alone is enough | DD Incidents API (needs Incident Management) |
| Grok export (F4) | `logpipe datadog export-pipeline` (CLI) + `POST /api/datadog/pipeline` (UI button optional). For each approved template, translate the `rule` from `rule_from_template` (`logpipe/parsing/grok.py:116`): rename matchers (`notspace→notSpace`, `quoted→quotedString`, `timestamp→data`, `path→notSpace`; `word/integer/number/ip/hostname/uuid/data/regex` stay the same) and emit `rule_<template_id> <pattern>`. Then upsert one pipeline named `logpipe (auto)` with a grok-parser processor. | rules are already in DD grammar; only matcher names differ | a new grammar generator |
| Webhook (F5) | `POST /api/datadog/webhook`, authenticated with a logpipe intake key in the header (existing `bearer`/`check_key`). The body is a DD webhook template (`$EVENT_TITLE`, `$HOSTNAME`, `$SERVICE`, `$DATE`). Creates an incident from it and runs auto-RCA through the F3 path. Exposed with cloudflared during the demo. | reuses auth + explain_incident | polling DD monitors |
| Seed data (F6) | Reuse `logpipe/showcase/generator.py` (web shop with a payment outage), have it write files that the existing `datadog-agent` compose profile tails, then the server pulls them back. | no new generator | a direct-intake pusher |
| Tests | One test per feature using `httpx.MockTransport` (offline CI), extending `tests/test_observability.py` style; plus the manual live run. | demo-grade but no silent breakage | live-only |

Reuse: `ServerCore.ingest`, `incidents()`, `explain_incident`, `self.control` get/put, `check_key`, `rule_from_template`/`compile_rule`, and the showcase generator.
Shortcuts (marked with `shortcut:` comments): the cursor is a timestamp (it can duplicate lines in the same millisecond), there are no retries/backoff on DD 429 (the next tick tries again), and the pipeline upsert replaces the whole pipeline.

## 4. Implementation order
1. Write `docs/plan/{persona,features,tech-decisions}.md` (from sections 1–3).
2. F0 `DatadogSettings` → `uv run python -c "import logpipe.observability"` passes and the existing tests are green.
3. `logpipe/datadog_api.py` + tests (search pagination, event payload, matcher translation).
4. F1 pull in housekeeping + the `--datadog-pull` flag (`logpipe/cli.py`, `logpipe/server/__init__.py`).
5. F2/F3 push in the same step.
6. F4 CLI + endpoint.
7. F5 webhook route in `logpipe/server/app.py`.
8. F6 Makefile target, `.env.example` (DD_APP_KEY, DD_PULL_QUERY, LOGPIPE public URL), README section.

## Verification
- `make check` (lint + typecheck + tests), with the new mocked tests passing.
- Live: `docker compose --profile datadog up -d datadog-agent` → `logpipe showcase --datadog` → in DD: showcase logs appear in Logs, then a "logpipe: anomaly on shop/payment" Event appears, followed by an RCA event; `logpipe datadog export-pipeline` creates the "logpipe (auto)" pipeline and DD Logs shows parsed attributes; a DD monitor with a webhook to the cloudflared URL produces an incident in the logpipe UI and an RCA event in DD.
