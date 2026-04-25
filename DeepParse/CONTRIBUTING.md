# Contributing to DeepParse

Thanks for your interest in DeepParse! This document explains how to set up the
development environment, our coding conventions, and the process for opening
pull requests.

## Quick start

```bash
git clone https://github.com/NightBaRron1412/DeepParse.git
cd DeepParse
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test,lint]"
pytest -q
```

The full suite runs in under a second on a laptop.

## Optional extras

| Extra | When to install | Command |
|---|---|---|
| `test` | Always (dev) — pytest + coverage + hypothesis | `pip install -e ".[test]"` |
| `lint` | Always (dev) — ruff + mypy | `pip install -e ".[lint]"` |
| `hf` | When using `synth_masks(mode="hf")` — torch + transformers + peft | `pip install -e ".[hf]"` |
| `train` | When fine-tuning your own adapter — adds `datasets` + `trl` | `pip install -e ".[train]"` |

## Project layout

See [README.md → Repository layout](README.md#repository-layout) for the full
tree. The hot paths to know:

- `deepparse/drain/` — Mask-First applier and Drain engine.
- `deepparse/synth/` — Mask synthesisers (offline stub + HF backend).
- `deepparse/training/` — LoRA fine-tuning entry point.
- `deepparse/tools/` — LogHub-2k fetcher and instruction-tuning dataset builder.
- `deepparse/utils/sampling.py` — Algorithm 1 (entropy-greedy sampling).
- `tests/` — 59 tests, < 1 s on a laptop.

## Local quality gates

These run in CI on every PR; please run them locally before pushing:

```bash
ruff check deepparse tests        # lint
mypy                              # type-check
pytest -q --cov                   # tests + coverage gate (>= 80%)
```

If you've fetched real LogHub-2k data, you can also run the integration tests:

```bash
python -m deepparse.tools.fetch_loghub --systems Apache HDFS
pytest -q -m integration
```

## Coding conventions

- **Python ≥ 3.10**. Use `from __future__ import annotations` so type hints are
  string-evaluated (lets us write `list[str]` and `str | None` everywhere).
- **Docstrings** explain *why*, not *what* — well-named identifiers handle the
  *what*. Reference the paper's section/listing/algorithm number when a
  function implements a paper claim.
- **No new files unless required** — prefer extending an existing module.
- **No dead branches** — if a behaviour can't happen, don't validate it.
- **Determinism is non-negotiable** for `synth/` (offline mode), `drain/`, and
  `utils/sampling.py`. Add a regression test when you touch those modules.

## Commit conventions

We use [Conventional Commits](https://www.conventionalcommits.org/) so the
changelog can be generated:

```
feat(core): add typed-placeholder Mask-First strategy
fix(drain): two-pass parse so first line returns converged template
docs: rewrite README with badges and mermaid diagram
chore(ci): bump actions/setup-python to v5
test: property-based check for entropy-greedy invariants
```

Allowed prefixes: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`,
`build`, `perf`.

## Pull request checklist

- [ ] `pytest -q` passes (and any new behaviour has a regression test).
- [ ] `ruff check deepparse tests` is clean.
- [ ] `mypy` is clean (heavy LLM modules are excluded; everything else is
  type-checked).
- [ ] Coverage is ≥ 80 % (gate enforced by `pyproject.toml`).
- [ ] You ran `make demo` end-to-end and got `GA = PA = 1.000`.
- [ ] If you changed a paper claim, update [README.md → Citation](README.md#citation)
  and the relevant test in `tests/test_*` so the assertion still matches.
- [ ] Commit messages follow Conventional Commits.

## Reproducibility expectations

DeepParse is a research artifact; reproducibility is the highest priority. When
you change anything in `drain/`, `metrics/`, or `utils/sampling.py`, rerun the
demo and check the metrics CSV. When you change `synth/`, rerun
`tests/test_synth_stub_determinism.py`.

## License

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
