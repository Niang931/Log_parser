# CI / CD

Everything in CI runs with `LLM_BACKEND=offline`: deterministic, no paid LLM calls.

## On every pull request (`.github/workflows/ci.yml`)

| Job | What it runs | Budget |
|---|---|---|
| Lint | `ruff check logpipe tests`; `ruff format --check` is advisory until the codebase is formatted in one dedicated PR | 5 min |
| Unit tests | `pytest -m "not integration"` with coverage (`coverage.xml` artifact), then `DeepParse/tests -m "not integration"` | 10 min |
| Integration | ClickHouse 25.8 service container, then `pytest -m integration`. Credentials are the ones from `.env.example` (`logpipe` / `ci-password` / db `logpipe`). | 15 min |
| Docker build | builds `DeepParse/Dockerfile` with the GitHub Actions layer cache, never pushes | 15 min |

Same commands locally:

```bash
make lint
make test
make test-integration
docker build DeepParse
```

The integration marker is `@pytest.mark.integration`. `tests/test_clickhouse_integration.py` is the smoke test that
proves the service wiring; tests of the ClickHouse storage backend go under the same marker when it lands.

## Nightly / on `main` (`.github/workflows/eval.yml`)

Runs at 02:00 Asia/Ho_Chi_Minh, on every push to `main`, and by hand (Actions → "Nightly eval gate" → Run workflow).

1. `logpipe fetch-data loghub2k` (cached on the hash of `logpipe/eval/datasets.py`)
2. `logpipe eval parsing --no-tune` (GA/PA on 14 Loghub-2k systems + IoT_fleet, offline masks, Drain depth 5 / sim 0.4)
3. `logpipe eval alias` (28 labelled template pairs)
4. `logpipe eval iot --epochs 3 --devices 3 --test-hours 2` (reduced end-to-end fleet run, about 80 s)
5. `logpipe eval report`, then `logpipe eval gate`

The gate (`logpipe/eval/gate.py`) compares with `eval/baseline.json` and fails the run when:

| Check | Rule |
|---|---|
| parsing mean GA | may not drop more than `tolerance.ga` (0.02) |
| alias wrong merges | `rules_offline` must stay at `false_merges` (0), hard limit |
| IoT fleet F1 (`full` variant) | may not drop more than `tolerance.iot_f1` (0.05) |

Everything under `artifacts/eval` (REPORT.md, JSON, CSV) is uploaded as the `eval-report-<run>` artifact, pass or fail.

### Refreshing the baseline

After an intentional improvement, record the new numbers and commit them:

```bash
make data-loghub
uv run logpipe eval parsing --no-tune --out artifacts/eval
uv run logpipe eval alias --out artifacts/eval
uv run logpipe eval iot --out artifacts/eval --epochs 3 --devices 3 --test-hours 2
uv run logpipe eval gate --out artifacts/eval --write-baseline
git add eval/baseline.json
```

Or run the workflow by hand with "write_baseline" ticked and copy `baseline.json` out of the uploaded artifact.
Keep the IoT parameters identical to `eval_params` in the baseline; the F1 of a 3-epoch run is not comparable with a 30-epoch one.

## Repository hygiene

- **Branch protection on `main`** (needs admin on `Niang931/Log_parser`; the ruleset is not a file in the repo):

  ```bash
  gh api -X PUT repos/Niang931/Log_parser/branches/main/protection \
    --input - <<'JSON'
  {
    "required_status_checks": {"strict": true,
      "contexts": ["Lint (ruff)", "Unit tests", "Integration (ClickHouse)", "Docker build (no push)"]},
    "enforce_admins": false,
    "required_pull_request_reviews": {"required_approving_review_count": 1},
    "restrictions": null,
    "allow_force_pushes": false,
    "allow_deletions": false
  }
  JSON
  ```

  The context names are the job `name:` fields in `ci.yml`; rename both together.
- **Dependabot** (`.github/dependabot.yml`): weekly uv updates for `/` and `/DeepParse`, grouped minor+patch, plus GitHub Actions versions.
- **pre-commit** (`.pre-commit-config.yaml`): ruff check/format on `logpipe` and `tests`, `uv lock` consistency, YAML and large-file checks. Install once with `uv run --with pre-commit pre-commit install`.
