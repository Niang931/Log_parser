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
| MiniLM | used when `sentence-transformers` is installed (not added to `pyproject`/`uv.lock`) | changing the lock overnight risked CI; TF-IDF + synonyms is the default. **Superseded** by the refactor below: MiniLM is a dependency, TF-IDF and the synonym table are gone |
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


## Refactor and groups (follow-up PR, branch `refactor/ask-groups`)

| Decision | Choice | Why |
|---|---|---|
| Configuration | `logpipe/settings.py`: one `pydantic-settings` class per environment prefix (`CLICKHOUSE_`, `MONGO_`, `KAFKA_`, `LOGPIPE_`, `LOGPIPE_ASK_`, `LOGPIPE_RCA_`, plus the unprefixed app/LLM ones), read once through `get_settings()`; the UI's settings are a validated `RuntimeSettings` model | env lookups and tuning constants were spread over ~12 modules; every number an operator may tune now has one home and an env override |
| What stays a constant | regex tables (level rules, shutdown pattern), prompts, column maps, allowed Mongo stages | they are part of the code's contract, not settings |
| Store reads and builders | `logpipe/server/stores/`: `base` (interfaces `StoreReads` and `Compiler`, shared level rules and JSON shaping), `sql` (one compiler for both SQL dialects, plus `guard_sql`), `memory`, `clickhouse`, `mongo` | replaces `reads.py`, `reads_mongo.py`, `querybuild.py`, `querybuild_mongo.py` and Ask's `if engine == ...` branches; a new backend is one module implementing two interfaces |
| Ask's data | `logpipe/ask/models.py`: pydantic models for rows read (`TemplateCount`, `Bucket`, `LogLine`, ...), the report (`Anchor`, `Candidate`, `Evidence`, `Answer`) and every result kind; field names unchanged, so stored plans and the UI keep working | `rca.py` passed dicts keyed by strings like `out["template_counts2"]`; reads are now looked up by named position (`Discover.TEMPLATES`) |
| Text index | MiniLM (`sentence-transformers`) only; identifiers are split into words before encoding | the synonym table hand-coded what the model is for; `LOGPIPE_ASK_MODEL` swaps the model |
| Groups | control doc `group`; a stream's `group` field (at most one group per stream); `/api/groups`; a Groups card and a group picker on Streams | e.g. mail = ldap + proxy streams; a question naming a group covers all its services |
| Groups in Ask | `QuerySpec.groups` (fast path matches group names; the router's catalog lists groups with their services); a root-cause search reads only the named groups' services; the final prompt lists the groups of the services in the bundle and says a failure in one service of a group often shows up first in another | the LLM can't know which services belong together unless told |
| LLM context | the final prompt gets the person's whole question (it got the router's paraphrase before) | the summary intent previously saw no question at all |
| Ask UI | the question read back as chips, a progress bar (read, approve, run, answer), a "your turn" bar per round with a countdown, queries opened on demand; the report as a verdict plus Answer / Evidence / Ranking tabs, cites that light up and jump to their line | the flow and the report were hard to follow |

### Results with a real LLM (Gemini through the browser provider, `llm:web:gemini`, Chromium)

`WEB_BROWSER=chromium python -m logpipe.eval.rca_eval --llm llm:web:gemini --record artifacts/eval/rca-gemini-replies.json --variants full`

| variant | evidence recall | answer correct |
|---|---|---|
| naive (last 200 lines) | 0.714 | - |
| full, rules only | 1.0 | 1.0 |
| full + Gemini | 1.0 | 1.0 |

All 16 synthesizer calls came back valid; every hypothesis cited lines in the bundle (none dropped). On the hung-API
scenario Gemini said the evidence was insufficient (it can't see a cause in lines that just stop); the scenario
counts as correct only through the rule-based "shop-api stopped logging" lead shown next to its answer. Every eval
question was simple enough for the rule-based router, so routing went through Gemini only in a separate check
(an average-per-host-over-time question, and "is email ok? nobody can sign in", which it mapped to the `mail` group).
The recorded replies can't be replayed in CI: prompts carry the times of the world the eval builds, which change
on every run.
