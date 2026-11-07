# AI log analysis v2: features

## Features (each → problem solved)
1. **QuerySpec + validator**: strict pydantic spec, clamped time, whitelisted services/hosts/templates. *Stops wrong/unsafe queries.*
2. **Router + query builder**: fast path (regex: time + keywords + level) and otherwise one structured LLM call with few-shots. Intents: general, keyword, event, semantic, rca, summary. *Lets the user ask "why" in one box.*
3. **Approval gate**: server-side pending plans; per-query approve/reject/edit; Settings default + Ask-box tighten-only switch; 15-min TTL. *Puts every query under user control (explicit requirement).*
4. **Read-only hardening**: CH read-only user/profile, table-function block (url/file/remote/s3/...) via parse check in SQL tab and AI, tests that attempt writes on all 3 stores. *Proves read-only.*
5. **LogStore interface**: `search`, `template_stats`, `anomaly_windows`, `vector_search` over memory/CH/Mongo, compiled from QuerySpec only. *Runs deterministic tools without the LLM.*
6. **Embedding index**: LogBERT template vectors (neighbours) + MiniLM text vectors (question→template, shutdown set); numpy default, Qdrant adapter. *Semantic search + shutdown anchors.*
7. **RCA engine** (no LLM): anchor priority (user time → heartbeat/rate silence → first shutdown template → first sustained LogBERT window → else clarification), window selection [−30m, +2m] widening, candidates, explainable ranking score, 20–40-line bundle. *Gives evidence-backed explanations.*
8. **Synthesizer**: cited timeline, ranked hypotheses with line/template IDs, "insufficient evidence" option, citation checker that drops uncited claims; bounded follow-up loop (≤3, each approved). *No unsupported claims.*
9. **Interface**: `/api/ask` (plan → approve → job), RCA report view (answer + evidence + QuerySpec + ranking reasons), Alerts Explain → RCA. *Unified Ask box; debuggable.*
10. **Eval harness**: ~15 ShopGenerator fault modes, evidence recall + answer correctness, ablations (no LogBERT, no semantic, naive last-200-lines), replay regression test. *Measured on fault scenarios.*

Cut by user: `logpipe ask` CLI, Qdrant adapter (numpy only; `VectorIndex` interface stays swappable). Eval harness moved early. Old `/api/query/ask` + `write_sql` removed in this PR. MiniLM = optional extra `[ask]`, TF-IDF fallback.

## Milestones (one commit each, tests green before commit)
1. **QuerySpec + router fast path + LLM route()** (M). Accept: 30 sample questions → expected intents/specs; invalid LLM JSON fails closed. Review: few-shots, intent list.
2. **Read-only hardening + LogStore over 3 stores** (M). Accept: write attempts fail on memory/CH/Mongo; table functions rejected in SQL tab. Review: CH RO user setup in compose.
3. **Eval harness skeleton + fault modes** (M). Accept: ~15 ShopGenerator fault modes + 2 controls saved as fixtures; naive last-200-lines baseline and evidence-recall metric run (recall 0 for RCA until M5). Every later milestone reports its numbers.
4. **Approval gate + plans API + Settings toggle; remove old /api/query/ask + write_sql** (M). Accept: no execution without approve when required; tighten-only override; TTL expiry; edits re-validated.
5. **Anchor + baseline + evidence ranker** (L). Accept: evidence recall on the fault set beats the naive baseline; no-anchor → clarification. Risk: anchor heuristics.
6. **Synthesizer + citation checker + Alerts Explain → RCA; record/replay LLM** (M). Accept: uncited claims dropped; "insufficient evidence" on controls; correctness metric live.
7. **Embedding index (numpy; LogBERT + MiniLM/TF-IDF) + shutdown set** (L). Accept: "out of memory" → OOM template top-3; shutdown set editable on Templates page; ablation with/without semantic reported.
8. **Agent follow-up loop with per-round approval** (S/M). Accept: ≤3 rounds, each pauses.
9. **UI: Ask panel, approval list, RCA report, shortcut** (M). Then final ablation table in the PR description.

Riskiest assumptions + cheap experiments: (a) anchor heuristics: check on the gateway-outage incident and 3 fault modes before building synth; (b) ranking formula: print factor breakdowns and eyeball recall@bundle across scenarios in M4; (c) baseline: compare median-24h vs immediately preceding on the 3 scenarios; (d) LogBERT neighbour vectors are meaningful: measure neighbour precision in M6, and drop them if they're noise.

Fault modes (ground-truth template): process killed (SIGTERM), OOM kill, segfault, disk full (ENOSPC), DB port blocked (conn refused), DB pool exhausted, payment gateway timeout (existing), DNS failure, TLS cert expired, config reload error, dependency 5xx cascade, memory pressure/GC thrash, slow query storm, clock skew/token expiry, agent dies (silence), plus 2 no-fault controls.
