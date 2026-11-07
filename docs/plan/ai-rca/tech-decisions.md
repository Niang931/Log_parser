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
