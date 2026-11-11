# AI log analysis v2: persona

## Phase 0: Repo findings

**Stack**: Python 3 / FastAPI server (`logpipe/server/app.py`, `core.py`), React+TS UI (`web/src`), pytest (`tests/`).
Stores: memory (SQLite-on-demand for queries), ClickHouse, MongoDB (`logpipe/storage/*`, reads in `logpipe/server/reads.py`, `reads_mongo.py`).

1. **Parsed logs / templates**
   - Rows: `iot_logs` (log_id, timestamp, device_id=host, service, source, template_id, log_status, payload JSON, raw_message, tags). Schema `logpipe/storage/schema.sql`.
   - Templates: versioned registry file (`logpipe/registry.py`, `server.registry.current` → `templates`, `canonical_id`, `key_vocab`). Per-template metadata docs in control store (`server.tmeta`, kind `template`). `/api/templates` exposes counts. No per-template first_seen/baseline-frequency table.
   - Level is computed from the line (`reads.LEVEL_SQL`, `reads.level()`), not stored.
2. **LogBERT scores**
   - Per **window** (`detector.sliding_windows`, `server.window`), per **stream** model (`server.scorers[stream]`, `LogBERTScorer`).
   - Only windows over threshold are persisted: `anomaly_alerts` (miss_ratio, novelty_ratio, template_ids, reasons). No table of all window scores.
   - Mispredicted keys: recomputed on demand by `LogBERTScorer.explain(window)` → per-event `expected | unexpected | unseen` + expected template IDs (used by `Server.incident()`, `core.py:1254`).
   - Incidents = alerts on one service grouped by `INCIDENT_GAP` (5 min) (`Server.incidents()`).
3. **Existing AI code**
   - `logpipe/enrich/llm.py`: `LLMAdapter` protocol, `OfflineAdapter` (deterministic fallbacks), `ChatAdapter` (langchain structured output via `deepparse` providers; anthropic/openai/web). `_Traced` wrapper in `core.py` records prompts into job detail.
   - **Ask in English** (`POST /api/query/ask`): one LLM call writes raw SQL (CH/SQLite) or a builder spec (Mongo). Guard = `startswith(select|with)`; execution guard = CH `readonly=2`, SQLite `PRAGMA query_only` on a throwaway in-memory DB, Mongo `querybuild_mongo.check()`. UI already shows SQL in the SQL tab before the user presses Run.
   - **Incident explain** (`POST /api/incidents/{id}/explain` → job `_job_explain`, `core.py:1325`): builds a text of ≤60 incident events (marked * / ?) + ≤20 warn/error lines from other services ±1 min, one LLM call → `{summary, likely_cause, next_steps}`. No citations, no evidence ranking, no baseline, no "insufficient evidence" path. Offline fallback points at first unusual line.
   - No embeddings/vector index for templates in server path (`logpipe/learning/similarity.py`, `enrich/similarity.py` exist — to check whether reusable).
4. **Missing for the feature**: per-window score history, template baseline frequencies / first_seen, heartbeat/silence detection, vector index, QuerySpec validation for the LLM path (raw SQL today), citations, eval harness with fault-injection scenarios.

## Answers
- Ponytail OFF for this feature: build the full design.
- Pain: explanations vague/unsupported; can't ask "why" questions. Audience: on-call SRE.
- Success: measured on fault scenarios; approval gate + read-only proven; one unified Ask box; RCA cites real line/template IDs.
- Gate covers **every** query incl. RCA tool calls; user sees spec + compiled SQL/pipeline, can edit spec, approve/reject per query.
- Enforcement: server-side pending plan with ID; only `approve` executes. Toggle = server default (Settings) + per-request override on Ask box.
- LLM emits QuerySpec only, never raw SQL. Human SQL tab: Run = approval. Block CH table functions (url/file/remote/...) in both SQL tab and AI.
- Read-only: dedicated read-only DB user (CH readonly profile, like Mongo's `MONGO_RO_USER`); existing per-query guards stay.
- Window scores: recompute on demand via `scorer.explain()` (pipeline untouched). Baseline: same-length windows over previous 24h, median.
- Embeddings: LogBERT template embeddings (user). hybrid with a local text model (MiniLM) for text→template, because LogBERT vectors can't embed free text.
- Vector store: user floated a dedicated vector DB. numpy in-process default + Qdrant adapter behind `LogStore.vector_search` (compose profile).

- Ask UI replaces the Query page's "Ask in English" tab, plus a global shortcut. Alerts "Explain" becomes RCA anchored on the incident.
- RCA = background job (existing jobs system) that pauses in `waiting_approval`. General tier: brief answer, labelled "not from your logs".
- Offline: deterministic fast path + evidence-only RCA (timeline + bundle, no prose).
- Silence anchors: agent heartbeat gap + per-service line rate ≈ 0. Shutdown set: seed phrases → text-embedding search, cached per registry version, editable on Templates page.
- Agent loop: each follow-up round pauses for approval (max 3).
- Eval: extend `ShopGenerator` with fault modes; deterministic correctness (top hypothesis cites ground-truth template, plus no-fault controls expect "insufficient evidence"); record/replay LLM replies.
- Override: the Ask box can only tighten (turn approval on); it can't skip approval when the server requires it.
- Stores: memory + ClickHouse + Mongo. Budget ≈6k tokens, 20–40 lines, 300 chars/line. Pending plans expire after 15 min; relative times are re-resolved at approval.
- Delivery: one PR from `feat/ai-rca` to Niang931/Log_parser, one commit per milestone, no Claude attribution.

## Persona
- **Who**: on-call SRE/ops engineer running logpipe for their own services (web shop, IoT fleet). Reads logs and SQL fluently, and is under time pressure during an outage.
- **Problem**: today's Explain gives a plausible cause with no proof, and Ask can only write a query, not answer "why did X go down?". The cost is slower MTTR, wrong fixes and distrust of the AI features.
- **Solution**: one Ask box. A router turns the question into a validated QuerySpec, the user approves each read-only query, a deterministic RCA engine picks the anchor and ranks evidence, and the LLM writes a cited timeline and hypotheses, or says "insufficient evidence".
- **Out of scope**: external vendors (Datadog), LLM reading raw logs in bulk, LLM-written SQL, auto-remediation, multi-user permissions (single admin), changing the parse/detect pipeline.
