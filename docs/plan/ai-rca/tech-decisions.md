# AI log analysis v2: technical decisions

## Technical decisions
| Decision | Choice | Why | Rejected |
|---|---|---|---|
| Package | new `logpipe/ask/` (spec.py, router.py, store.py, index.py, rca.py, synth.py, plans.py) | isolated from parse/detect pipeline | putting it in core.py (already 1.5k lines) |
| LLM output | pydantic QuerySpec via existing `ChatAdapter._invoke` structured path; add `route()` + `synthesize()` to `LLMAdapter` and `_Traced` | reuses providers, tracing, offline fallback | raw SQL (`write_sql`), kept only until the old route is removed |
| Compilation | extend `querybuild.build` / `querybuild_mongo.build` with keyword/template/time-absolute filters; QuerySpec → builder spec | builder is already the injection guard on all stores | new compiler |
| Read guard | queries go through `server.reads.query` (CH `readonly=2`, SQLite `query_only`, Mongo `check()` + RO user) + new CH RO user env `CLICKHOUSE_RO_USER` + table-function denylist | defence in depth; mirrors Mongo RO commit 7af9dc3 | sqlglot parser (new dep). Instead: a table-function denylist regex, and an RO user without `CREATE TEMPORARY TABLE`/URL/FILE/REMOTE grants as the real barrier |
| Approval | control doc kind `ask_plan` {id, queries[], status, expires_at}; `POST /api/ask` → plan; `POST /api/ask/{id}/approve {approved:[ids], edits}` → job; jobs gain `waiting_approval` state for follow-ups | server-enforced; reuses control store + jobs + SSE `events.notify()` | UI-only confirm |
| Toggle | `DEFAULT_SETTINGS["ask_require_approval"]=True`; request `require_approval` can only OR it on | user rule: tighten-only | — |
| Window scores | `scorer.explain()` on lines from `reads.between()` in the RCA window | pipeline untouched | window_scores table |
| Baseline | per-template counts in 48 windows of 30 min over prior 24h (aggregate query through LogStore), median + MAD; new = 0 in baseline | simple, explainable | same-time-yesterday |
| Embeddings | LogBERT `BERTEmbedding` token table per stream (template→template) + `sentence-transformers/all-MiniLM-L6-v2` (text) as optional extra `[ask]`; falls back to TF-IDF if not installed | text query needs a text model | LogBERT-only |
| Vector store | `VectorIndex` protocol; `NumpyIndex` persisted at `data_dir/ask/index-{registry_version}.npz`, rebuilt on registry publish (incremental add for new templates) | small data; swappable | dedicated vector DB (cut by user) |
| Ranking | `score = anomaly × rarity × proximity`, with anomaly = max(miss_ratio, unseen) from explain, rarity = 1/(1+baseline_median) with a spike boost, proximity = exp(−Δt/10m) before the anchor; ties → earliest; each line carries its factor breakdown | explainable | learned ranker |
| Citations | evidence lines get IDs `L1..Ln`, templates `T<id>`; synth schema requires `cites: list[str]`; post-check drops hypotheses citing unknown IDs and marks them; zero valid → "insufficient evidence" | no hallucination tolerance | trust prompt only |
| UI | `web/src/pages/Query.tsx` Ask tab → `AskPanel` (plan approval list with editable spec + compiled query, RCA report); Alerts Explain → same report; Settings switch; shortcut in `shortcuts.tsx` | follows existing pages | new page |
| Eval | `logpipe/eval/rca_eval.py` + `ShopGenerator.fault(mode)`; `tests/fixtures/rca/<scenario>/` (lines + truth + recorded LLM replies); `tests/test_rca_replay.py` | CI-deterministic | live LLM |

## Verification
- `uv run pytest` (new: test_ask_spec, test_ask_router, test_ask_readonly, test_ask_plans, test_rca, test_rca_replay; existing test_server/_mongo/_clickhouse updated for the new Ask routes).
- `uv run python -m logpipe.eval.rca_eval` → table of evidence recall / correctness per ablation.
- Run `logpipe showcase --auto`, then by hand: ask "why did checkout fail?" → approve the plan in the UI → cited RCA report; flip the Settings toggle → it runs without a pause; try an INSERT in the SQL tab → rejected.
- After plan approval: write `docs/plan/persona.md`, `features.md`, `tech-decisions.md` first.

## Decisions made during the unattended run
Made without asking, since the user was away overnight. Each one is easy to revisit.

| Decision | Choice | Why |
|---|---|---|
| Plan docs location | `docs/plan/ai-rca/` | `docs/plan/*.md` already held the agent project's plan; not overwritten |
| Relative times at approval | **not** re-resolved: the approved text runs exactly as shown | "the query you approved is the query that ran" is the stronger guarantee; the 15-min TTL bounds staleness |
| Spec edits | separate `POST /api/ask/{id}/edit` that re-plans and waits again (even with approval off) | approving in the same call would run queries the person never saw |
| Template-catalog lookups | semantic/event template search runs at planning time, not as an approved query | it reads template text from the registry, not log data; every log-store read is gated |
| Anchor rule 5 | first ERROR/WARN template absent from the 24 h before | the server's detectors run with novelty threshold `inf`, so unseen-template faults raise no LogBERT alerts; without this, 7/15 faults had no anchor |
| Aggregates | `QuerySpec.aggregate` (fn, field, group_by, over_time) on the keyword intent | keeps the old Ask's "errors per service" questions working without LLM-written SQL |
| Mongo level filter | `LEVEL_EXPR` ($switch of LEVEL_RULES) in the builder; Query page gains level on Mongo | RCA needs error/warn reads on every store |
| Memory store queries | SQLite copy cached until the next insert | an RCA runs ~10 queries; rebuilding per query made the eval 10x slower |
| MiniLM | used when `sentence-transformers` is installed (not added to `pyproject`/`uv.lock`) | changing the lock overnight risked CI; TF-IDF + synonyms is the default |
| Qdrant / CLI | cut by the user | `VectorIndex`-style `search`/`neighbours` stay the swap point |

## Results (`python -m logpipe.eval.rca_eval`, rules only, no LLM)

| variant | evidence recall | answer correct |
|---|---|---|
| naive (last 200 lines) | 0.714 | - |
| full | 1.0 | 1.0 |
| no LogBERT | 1.0 | 1.0 |
| no semantic | 1.0 | 1.0 |

17 scenarios (15 faults + 2 controls). "answer correct" means the top lead cites the root-cause line (for a hung
service, that the silence leads); on the controls it means "insufficient evidence" or a clarification request.

Caveats: the heuristics were tuned on these same scenarios, so this is a fit, not a held-out score. LogBERT and
semantic search change nothing on this set; the scenarios don't need them. The LLM path is wired and
record/replay-tested with a scripted model, but no real-model numbers exist yet
(`--llm llm:anthropic --record eval/rca-replies.json`).
