# Maintenance

## Everyday workflow

```sh
git switch -c feat/<name>            # features go on a branch; PRs to Niang931/Log_parser
make format && make lint             # ruff (CI also runs ruff format --check)
make test-fast                       # unit tests without model training
make test                            # everything but integration
make test-integration                # ClickHouse + Kafka (starts them)
make typecheck                       # pyright
```

- **Commit each finished, tested piece**, with explicit `git add <paths>` (several people
  may work in one tree).
- **Comments go above the code**, not trailing, and explain *why*. A known limit gets
  `# shortcut: <the limit>, <when to upgrade>`.
- **Settings**: a new tunable number goes in `settings.py` under the right prefix, not as a
  module constant.
- **LLM calls**: every new kind of call gets a hostile line in
  `tests/fixtures/prompt_injection.txt` and a deterministic offline fallback.

## Recurring tasks

| When | Task | How |
|---|---|---|
| A benchmark improves on purpose | refresh the gate baseline | `make data-loghub`, run the three reduced evals, `logpipe eval gate --write-baseline`, commit `eval/baseline.json` ([CI](../ci.md#refreshing-the-baseline)) |
| A prompt changes | re-record LLM replies | `python -m logpipe.eval.rca_eval --llm … --record …`; replays fail loudly on unseen prompts |
| Dependabot PR | review and merge | weekly uv (root + `DeepParse/`) and Actions updates, grouped minor + patch |
| A web chat site breaks | fix selectors | add a selector at the front of the tuple in `ai_scraper/sites.py`; the capture in `DeepParse/.webchat/errors/` shows the page |
| camoufox upgrades | refetch its browser | `make web-setup` |
| The mask model drifts | retrain on a GPU | `make retrain-data HOLDOUT=…`, `notebooks/colab_retrain.ipynb`, then `make retrain-eval MASKS=…` |
| Server models go stale | retrain streams | PSI > 0.25 on **System → Model drift**, or automatic after `retrain_days` with unseen templates |
| Docs change | nothing manual | `docs.yml` builds and publishes the site on push to `main` |

## Debugging

| Symptom | Where to look |
|---|---|
| Lines stay `PENDING` | **Maintenance** (held new sources wait for review); queue lag and dead letters in `logpipe worker` output or **System**; `CycleReport` (`deferred` means the cluster was too small or over the per-cycle cap) |
| An alert storm after a rollout | fleet suppression (`fleet_min_devices`, `fleet_min_minutes`); retrain the stream so the new templates are known |
| Ask says "insufficient evidence" | the report's **Ranking** tab: anchor, window, and each candidate's anomaly × rarity × proximity. "no baseline" means the store has no history before the window |
| RCA misses a service that *called* the faulty one | known limit ([RCAEval findings](../benchmarks/rcaeval.md#findings)): use an LLM synthesizer |
| Web LLM errors | `DeepParse/.webchat/errors/<time>_<site>_<Error>/`: `page.html`, `screenshot.png`, `error.json` |
| Server errors | `--data-dir/server.log` (tracebacks); the console shows one line per warning |
| A job failed | **Jobs** page: step log, retry |
| An eval regression | `artifacts/eval/REPORT.md` vs `eval/baseline.json`; the nightly artifact `eval-report-<run>` |

## Releases and images

- `images.yml` builds and pushes the server image and the multi-arch agent image
  (amd64, arm64, arm/v7) to GHCR on pushes to `main` and on `v*` tags.
- The server image builds `web/` in a Node stage and copies it into `logpipe/server/static`.
- The agent is a static Go binary (~15 MB image); `agent/README.md` covers install and config.

## The docs site

```sh
make docs          # live preview on http://127.0.0.1:8000
make docs-build    # strict build into site/ (fails on broken links)
```

Material for MkDocs runs through `uvx`, so it isn't a project dependency. Pages are in
`docs/`, the navigation in `mkdocs.yml`; diagrams are Mermaid code blocks. GitHub Pages must be
set to **Source: GitHub Actions** once (repository settings) for `docs.yml` to publish.

## Known gaps

- A fast-path process doesn't reload a registry another process wrote; Kafka workers' new
  templates reach it on restart.
- Hand-written parsing rules are tried one by one before the tree; index them if there are
  ever thousands.
- The MCP bench reaches into server internals (`_job_loop`, `_stop`) to run jobs while an agent
  works; a public `start_jobs()` would be cleaner.
- RCA ranks services by their own lines, so a caller with loud errors can outrank the callee
  that failed.
