# DeepParse examples

Runnable end-to-end workflows for each reproduction tier described in the
[main README](../README.md#reproducing-the-paper).

| File | Tier | What it does | Runtime |
|---|---|---|---|
| [`01_quickstart.py`](01_quickstart.py) | A | Three-line public API on a tiny in-memory log batch | < 1 s |
| [`02_demo_pipeline.sh`](02_demo_pipeline.sh) | A | Bundled DemoTiny end-to-end (synth → eval → table) | < 30 s |
| [`03_loghub_offline_eval.sh`](03_loghub_offline_eval.sh) | B | Real LogHub-2k eval with the offline stub | ~1–2 min |
| [`04_finetune_and_eval.sh`](04_finetune_and_eval.sh) | C | Full LoRA fine-tune of DeepSeek-R1-Distill-Llama-8B + eval | ~30–90 min on a GPU |
| [`05_use_trained_adapter.py`](05_use_trained_adapter.py) | C | Load a fine-tuned adapter and parse a log batch | seconds (after model load) |

Run any of them from the repository root. None of them depend on each other —
`02` doesn't require `01` to have run, etc.

## Prerequisites

```bash
# A + B
pip install -e ".[test,lint]"

# C (training)
pip install -e ".[train]"
```
