# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] – 2026-04-17

The first public, paper-aligned release of DeepParse, the artifact for the
EASE 2026 paper *DeepParse: Hybrid Log Parsing with LLM-Synthesized Regex Masks*.

### Added

- **Real LoRA fine-tuning pipeline** targeting `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
  with the paper's hyperparameters (LoRA r=8 α=32 dropout=0.01, AdamW lr=2e-4,
  batch=8, 25 epochs). `--small` swaps to Qwen2.5-0.5B-Instruct for CPU smoke
  runs. Cross-vendor stable on NVIDIA + AMD ROCm.
- **Hugging Face inference backend** that loads a fine-tuned LoRA adapter on
  top of the base model, prompts per Listing 2, defensively parses the Python
  regex list, validates each pattern, and backfills the four core variable
  classes when the LLM omits any. Supports the paper's self-consistency retry
  loop.
- **LogHub-2k corrected fetcher** (`python -m deepparse.tools.fetch_loghub`)
  that pulls the corrected benchmark from `logpai/loghub-2.0` and converts each
  system to the directory layout consumed by the eval runner.
- **Listing-2 instruction-tuning dataset builder** with `--entropy-k 50` to
  match the paper's per-system 50-shot training recipe.
- **Algorithm 1 (entropy-greedy sampling)** in `deepparse/utils/sampling.py`
  with digit/hex normalisation, Shannon entropy, and Jaccard < 0.8 rejection.
- **Property-based test suite** (`tests/test_property_*.py`) using Hypothesis
  to verify sampling and Drain invariants over thousands of random inputs.
- **Real LogHub-2k integration test** (`tests/test_loghub_integration.py`)
  skipped on a fresh checkout, exercised once data is present.
- **Self-contained Colab notebook** for the full Tier C training run.
- **`scripts/synth_all_with_adapter.sh`** convenience wrapper for per-system
  synth via a fine-tuned adapter.
- **Production CI**: multi-Python matrix (3.10/3.11/3.12), ruff lint, mypy
  type-check, pytest with coverage gate (≥ 80 %), Codecov upload, integration
  job, and demo-pipeline smoke test.
- **`CONTRIBUTING.md`** with development workflow, conventions, and PR checklist.
- **`examples/`** directory with runnable workflows for each reproduction tier.

### Changed

- **Drain defaults** updated to match the paper's Drain3 baseline configuration:
  `depth=5`, `similarity_threshold=0.4`, `max_children=100`.
- **Mask-First strategy** rewritten to use typed placeholders
  (`<VAR:TIMESTAMP>`, `<VAR:IP>`, …) per Section *Integration with Drain*; the
  parse tree keys on typed placeholders, but final templates render the
  LogHub-canonical `<*>` wildcard so they compare directly against LogHub-2k
  ground truth.
- **Grouping Accuracy metric** rewritten to the standard LogHub definition
  (partition-set equality, invariant under cluster-ID renumbering) per paper
  Section *Background*.
- **Drain `parse()`** now does a two-pass walk so every returned template
  reflects the cluster's converged state, not the state at insertion time.
- **DemoTiny** corpus expanded to 14 lines exercising every variable class;
  ground-truth `templates.json` is generated from the canonical pipeline so a
  faithful implementation hits GA = PA = 1.0 on the demo.
- **Public API defaults** match Listing 1 of the paper verbatim:
  `synth_masks(sys_logs, sample_size=50, temperature=0, max_length=512)`.
- **`pyproject.toml`** lowered Python floor to 3.10 so the artifact runs on
  conductor MI300A boxes (Python 3.10) and on Colab. Heavy LLM stack split into
  `[hf]` (inference) and `[train]` (adds `datasets` + `trl`) extras so the core
  install stays light.
- **Docker / Makefile / CI** switched from `uv` to plain `pip` for portability.

### Fixed

- `MaskApplier` strips duplicate named groups (`(?P<x>…)`) so LLM-emitted
  bundles never raise `re.error: redefinition of group name`.
- LoRA fine-tune correctly enables `enable_input_require_grads()` after
  gradient checkpointing — the standard PEFT mitigation for the
  `loss=0 grad_norm=NaN` failure mode (cross-vendor: CUDA + ROCm).
- LoRA adapter parameters cast to fp32 even when the base model is bf16;
  `--bf16` opts back into the paper's mixed-precision recipe once the stack is
  verified stable.
- Listing-2 training set emits proper Python raw-string literals (`r"\d+"`)
  rather than JSON-double-escaped patterns (`r"\\d+"`).
- Eval runner now reads annotated `templates.json` ground truth when present
  and falls back to the canonical-regex oracle otherwise.
- Default `attn_implementation="eager"` to avoid SDPA / Flash-Attention NaN
  gradients on AMD ROCm and on torch wheels built against a different ROCm
  version than the host driver. `--attn-impl=sdpa` opts back into the faster
  kernel when the stack is verified stable.

### Verified

- 59 unit + property-based + integration tests pass in < 6 s; coverage 80.6 %.
- `ruff` and `mypy` both clean across all 28 source files.
- All five reproduction tier examples (`examples/01_…` through `05_…`) run
  end-to-end on a clean checkout.
- Tier C end-to-end on AMD MI300A (cr66-8): 25-epoch LoRA fine-tune of
  `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` converges from train_loss 2.38 →
  0.09 in 33 minutes; trained adapter then synthesises system-specific masks
  (e.g. `blk_-?\d+` for HDFS block IDs, `0x[0-9a-fA-F]+` for BGL hex literals).
- `python -m build` produces a clean sdist + wheel; the release workflow uses
  these artefacts.

[Unreleased]: https://github.com/NightBaRron1412/DeepParse/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/NightBaRron1412/DeepParse/releases/tag/v1.0.0
