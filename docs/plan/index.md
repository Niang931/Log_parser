# Design records

Each larger feature was planned before it was built: a **persona** (who it's for, the problem,
what's out of scope), a **feature list** in build order, and a table of **technical decisions**,
each with the choice, the reason and what was rejected. The records are kept as written, with
later changes appended, so the reasoning behind the current code can be traced.

| Feature | Persona | Features | Decisions |
|---|---|---|---|
| MCP endpoint for AI assistants | [persona](persona.md) | [features](features.md) | [decisions](tech-decisions.md) |
| AI log analysis v2 (Ask + root cause) | [persona](ai-rca/persona.md) | [features](ai-rca/features.md) | [decisions](ai-rca/tech-decisions.md) |

Decisions made since, summarised:

| Decision | Choice | Why |
|---|---|---|
| RCA benchmark | RCAEval's blind task view, scored by its own evaluator, run through `uvx` | held-out real testbeds; no new dependencies |
| MCP access to the RCA | an `ask` tool that approves its own read-only rounds | an agent can't click Approve; every query it runs is still a read |
| Learner cycles in evals | `settle()`: run cycles until one creates no template (cap 10) | a fixed count was a guess; "until empty" never ends for tiny clusters |
| RCA baseline slices | whole window-sized slices ending at the window | the newest slice, the only one with short history, was being dropped |
| Web chat cookie banners | always **Reject all** | the banner blocked the prompt box; consent shouldn't be given silently |
| LLM ranking in benchmarks | services the LLM's hypotheses cite lead the ranking | otherwise an LLM run scores the same as the rules |
| Bench progress output | one plain line per step with a block bar | in-place redrawing flickered with other output |
