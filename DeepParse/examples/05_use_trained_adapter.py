"""DeepParse Tier C — load a trained LoRA adapter and parse a log batch.

Run from the repo root after fine-tuning (or after downloading an adapter
from the GitHub release):

    python examples/05_use_trained_adapter.py \\
        --adapter artifacts/checkpoints/deepparse-r1-8b
"""
from __future__ import annotations

import argparse
from pathlib import Path

from deepparse import Drain
from deepparse.synth.hf_deepseek_r1 import synthesize_hf_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True,
                        help="Path to a fine-tuned LoRA adapter directory")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    sys_logs = [
        "PacketResponder 1 for block blk_38865049064139660 terminating",
        "PacketResponder 0 for block blk_-6952295868487656571 terminating",
        "BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010",
        "workerEnv.init() ok /etc/httpd/conf/workers2.properties",
        "mod_jk child workerEnv in error state 6",
    ]

    print(f"loading adapter {args.adapter} ...")
    masks = synthesize_hf_from_checkpoint(
        args.adapter, sys_logs,
        max_length=args.max_length,
        self_consistency_attempts=1,
    )
    print(f"\nsynthesised {len(masks)} masks:")
    for m in masks:
        print(f"  {m.label:10s} {m.pattern}")

    drain = Drain()
    drain.load_masks([m.to_dict() for m in masks])
    print("\nparsed templates:")
    for line, template in zip(sys_logs, drain.parse_all(sys_logs)):
        print(f"  {line}")
        print(f"  -> {template}\n")


if __name__ == "__main__":
    main()
