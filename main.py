"""
DeepParse v2 — Entry point
==========================
This file contains only the CLI argument parser and calls run().
All logic lives in the pipeline/ package:

  pipeline/config.py    — constants, thresholds, schemas
  pipeline/telemetry.py — Telemetry class + set_global_seed
  pipeline/security.py  — prompt-injection sanitisation
  pipeline/loaders.py   — XML / CSV / TSV / JSON / YAML / INI loaders
  pipeline/masks.py     — mask loading, validation, adaptive synthesis
  pipeline/parser.py    — Drain wrapping, variable extraction, validation
  pipeline/writers.py   — SQLite / CSV / JSON / cluster report / PostgreSQL
  pipeline/pipeline.py  — run() orchestrator
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from pipeline.config import ADAPTIVE_MAX_RETRIES, ADAPTIVE_PARSE_THRESHOLD
from pipeline.pipeline import run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="DeepParse v2 — Silicon-Fab Log Parsing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples
        --------
        # Parse a single file (Groq free tier)
        python main.py --input artifacts/data/fab.log --llm-provider groq

        # Parse a whole folder
        python main.py --input-dir artifacts/data --llm-provider groq

        # Offline / CI (no API key needed)
        python main.py --input fab.log --llm-provider mock

        # Evaluation mode
        python main.py --mode eval
        """),
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    p.add_argument("--mode",   choices=["test", "eval"], default="test")
    p.add_argument("--seed",   type=int, default=42)

    # ── Input ─────────────────────────────────────────────────────────────────
    p.add_argument("--input",       default=None,
                   help="Single input file")
    p.add_argument("--input-dir",   default=None,
                   help="Process all supported files in a folder")
    p.add_argument("--fmt",         default="auto",
                   choices=["auto","xml","structured","semistructured","unstructured"])
    p.add_argument("--message-col", default="message",
                   help="CSV/JSON column containing log text")

    # ── LLM / masks ───────────────────────────────────────────────────────────
    p.add_argument("--llm-provider",       default="groq",
                   help="anthropic | groq | gemini | openai | ollama | mock")
    p.add_argument("--static-masks",       default=None,
                   help="Path to masks_fab_universal.json (auto-detected if omitted)")
    p.add_argument("--no-mask-cache",      action="store_true")
    p.add_argument("--adaptive-threshold", type=float, default=ADAPTIVE_PARSE_THRESHOLD)
    p.add_argument("--adaptive-rounds",    type=int,   default=ADAPTIVE_MAX_RETRIES)

    # ── Drain ─────────────────────────────────────────────────────────────────
    p.add_argument("--drain-sim",   type=float, default=0.5,
                   help="Similarity threshold 0–1 (default 0.5; per-source override applied)")
    p.add_argument("--drain-depth", type=int,   default=4)
    p.add_argument("--no-drain-state", action="store_true")

    # ── Limits ────────────────────────────────────────────────────────────────
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-logs",   type=int, default=1000)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--config",     default=None)

    args = p.parse_args(argv)

    shared = dict(
        mode                = args.mode,
        root_dir            = Path(__file__).parent,
        seed                = args.seed,
        output_dir          = Path(args.output_dir) if args.output_dir else None,
        input_fmt           = args.fmt,
        message_col         = args.message_col,
        llm_provider        = args.llm_provider,
        use_mask_cache      = not args.no_mask_cache,
        max_length          = args.max_length,
        max_logs            = args.max_logs,
        config              = args.config,
        static_masks_path   = args.static_masks,
        adaptive_threshold  = args.adaptive_threshold,
        adaptive_rounds     = args.adaptive_rounds,
        drain_sim_threshold = args.drain_sim,
        drain_depth         = args.drain_depth,
        save_drain_state    = not args.no_drain_state,
    )

    # ── Batch directory mode ──────────────────────────────────────────────────
    if args.input_dir:
        input_dir  = Path(args.input_dir)
        extensions = {".log",".txt",".csv",".tsv",".json",
                      ".jsonl",".xml",".yaml",".yml",".ini"}
        files = sorted(f for f in input_dir.iterdir()
                       if f.suffix.lower() in extensions and f.is_file())
        if not files:
            print(f"No supported files found in {input_dir}")
            return 1
        print(f"\nFound {len(files)} file(s) in '{input_dir}'\n")
        for i, f in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {f.name}")
            run(input_path=str(f), **shared)
        return 0

    # ── Single file mode ──────────────────────────────────────────────────────
    run(input_path=args.input, **shared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())