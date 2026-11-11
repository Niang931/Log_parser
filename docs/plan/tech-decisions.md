# MCP: technical decisions

## Technical decisions
| Decision | Choice | Why | Rejected |
|---|---|---|---|
| Depth | Production-grade, lean | teammates will use it over the internet | prototype |
| SDK | official `mcp` package (FastMCP), `stateless_http=True`, `json_response=True`, `streamable_http_path="/"`, mounted at `/mcp`; FastAPI `lifespan` runs `mcp.session_manager.run()` | standard; stateless works with any worker count | hand-rolled JSON-RPC |
| Where | new `logpipe/server/mcp.py` with `build_mcp(app) -> FastMCP`, mounted in `create_app` (`logpipe/server/app.py:50`) | one deploy, same URL and TLS as the UI | separate service or stdio |
| Tool implementation | each tool calls the app's own `/api` route through `httpx.AsyncClient(transport=ASGITransport(app))` and forwards the caller's `Authorization` header | reuses all route glue (`validate_sql`, template listing, query validation) and works on all three stores for free | calling `Server` directly, which would duplicate the closures in app.py |
| Auth | extend the `require_login` middleware (`app.py:58`). `/mcp*` always needs a valid `mcp` key. On `/api/*` a valid `mcp` Bearer key is accepted, instead of the cookie, only when `(method, path)` matches `MCP_ROUTES`, a module-level regex allowlist. The key is stored on `request.state.key`. | one enforcement point, so even curl with an MCP key cannot apply a batch | per-tool checks |
| Keys | `Server._store_key` / `create_key` take `scope` (`core.py:342/360`). `POST /api/keys` accepts `scope`. `agent_key` (`app.py:111`) rejects scope `mcp`. `Keys.tsx` gets a select and a badge. | reuses the key store, hashing and revocation | new token system |
| Note-only on incidents | the PUT `/api/incidents/{id}` route: if the caller is an `mcp` key and `status` differs from the current one, return 403. The tool sends the current status itself. | the allowlist can't see the request body | separate route |
| Attribution | optional `by` param on `set_incident`, `update_draft`, `decide_all`, `merge_drafts` (`core.py:1284/768/794/805`). The routes pass `request.state.key["name"]` when present, and `save_query` stores `by`. The web UI shows "by X" next to incident notes and drafts. | small change, visible to the team | audit log collection |
| Output caps | tools default `limit=100` and clamp to 500. List-style results are cut to the cap and carry `truncated: true` plus a hint ("narrow filters or use log_histogram/log_facets"). | protects the AI's context | cursors |
| Tests | `tests/test_mcp.py` against the memory store with `TestClient(create_app(server))`, following the fixture pattern in `tests/test_server.py:54` | matches repo | — |

### Tool → route map (all on the `MCP_ROUTES` allowlist)
- **Logs**
  - `search_logs` → GET `/api/logs`
  - `log_histogram` → GET `/api/logs/histogram`
  - `log_facets` → GET `/api/logs/facets`
  - `query_schema` → GET `/api/query/schema`
  - `run_query` → POST `/api/query`. This is already read-only: `validate_sql` on SQL stores, and the read-only Mongo user when `MONGO_READONLY_URI` is set.
- **Incidents**
  - `list_incidents` → GET `/api/incidents`
  - `get_incident` → GET `/api/incidents/{id}`
  - `list_alerts` → GET `/api/alerts`
  - `add_incident_note` → GET the incident, then PUT with the same status
- **Templates**
  - `list_templates` → GET `/api/templates`
  - `list_review_batches` → GET `/api/review`
  - `get_review_batch` → GET `/api/review/{id}`
  - `check_draft_rule` → POST `/api/drafts/{id}/check`
  - `update_draft` → PUT `/api/drafts/{id}`
  - `merge_drafts` → POST `/api/drafts/{id}/merge`
  - `decide_drafts` → POST `/api/review/{id}/decide-all`
  - `generate_review_batch` → POST `/api/review/generate`
- **Health**
  - `overview` → GET `/api/overview`
  - `stats` → GET `/api/stats`
  - `system` → GET `/api/system` and GET `/api/system/alerts`
  - `agents` → GET `/api/agents`
- **Drift and warnings**
  - `drift` → GET `/api/drift`
  - `warnings` → GET `/api/warnings`
- **Saved queries**
  - `list_saved_queries` → GET `/api/queries`
  - `save_query` → POST `/api/queries`
- **Never allowlisted**: apply, discard, settings, keys, rules, password, agents remote-config, and query ask/build.

### Prompts
- `investigate_incident(id)`: call `get_incident`, then `system`/`agents` to rule out a pipeline fault, then `log_histogram` and `search_logs` around the window on related services. Answer with a timeline and a root-cause hypothesis that cites line timestamps and template IDs. Offer `add_incident_note`.
- `review_batch(id)`: call `get_review_batch`, then for each draft check its rule and naming, merge duplicates, and approve or reject with a reason. End by reminding the human to click Apply in the UI.

## Implementation order
Commit after each step once its tests pass. Branch: `worktree-feat-mcp`. No Claude attribution lines. Open the PR against Niang931.
0. Write `docs/plan/persona.md`, `features.md` and `tech-decisions.md` from this file.
1. `uv add mcp`. Add the key scope in core, the intake check and the keys route, with tests.
2. Middleware allowlist, `request.state.key` and the incident status guard, with tests: an mcp key can GET logs, gets 403 on apply, and gets 403 on a status change.
3. `mcp.py`: tools, caps, prompts, the mount and the lifespan, with a test that lists tools and calls `search_logs` plus one write through the MCP endpoint.
4. Attribution `by` in core and routes, plus the web display.
5. Keys.tsx scope select and badge.
6. README section.

## Verification
- `uv run pytest tests/test_mcp.py tests/test_server.py tests/test_auth.py` and `uv run ruff check`.
- Manual run: start `logpipe server` (memory store) and create an mcp key in the Keys page. Then run `claude mcp add --transport http logpipe http://localhost:8080/mcp --header "Authorization: Bearer lp_..."` and in Claude Code ask it to "investigate the latest incident" and to "review the open batch". Check the note and drafts show "by <key>" in the UI and that Apply is still a human click.
- Negative checks: curl `/api/review/x/apply` with the mcp key returns 403. `/intake/v1/ping` with the mcp key returns 401. `/mcp` with no key returns 401.

## Risks
- `run_query` on ClickHouse relies on `validate_sql`. Mongo is fully read-only only when `MONGO_READONLY_URI` is set, so the README tells deployers to set it.
- Prompt injection: log lines are attacker-controlled text that the client AI reads. Writes are limited to proposals, so the worst case is a bad note or a wrong draft decision, which a human still reviews before Apply.
