"""LoRA fine-tuning script for the DeepParse mask synthesiser.

Defaults match the paper (Section 'LLM Configuration and Fine-Tuning'):

* Base model: ``deepseek-ai/DeepSeek-R1-Distill-Llama-8B``
* LoRA: rank=8, alpha=32, dropout=0.01
* Optimiser: AdamW, lr=2e-4, batch size=8, gradient accumulation=4 steps
* Precision: bfloat16 (falls back to float32 on CPU)
* Epochs: 25
* Max sequence length: 512 tokens

A ``--small`` flag swaps the base model to ``Qwen/Qwen2.5-0.5B-Instruct``
which trains and infers on CPU in minutes — useful for smoke tests and
for environments without a GPU.

Training data must be a JSONL file (one example per line) with keys
``instruction``, ``input``, ``output`` — the format produced by
:mod:`deepparse.tools.build_training_set`.

Usage:
    python -m deepparse.training.finetune \\
        --train artifacts/training/train.jsonl \\
        --output-dir artifacts/checkpoints/deepparse-r1-8b
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class FineTuneConfig:
    train_file: Path
    output_dir: Path
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    epochs: int = 25
    learning_rate: float = 2e-4
    batch_size: int = 8
    grad_accum: int = 4
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.01
    max_length: int = 512
    # Default fp32 for cross-vendor stability (works on NVIDIA + AMD).
    # The paper uses bf16; pass --bf16 once your stack is verified stable.
    bf16: bool = False
    # Eager attention is the cross-vendor-safe default (see comment in
    # run_training).  Pass --attn-impl=sdpa or =flash_attention_2 to opt
    # back into the faster kernels when your stack supports them.
    attn_implementation: str = "eager"
    seed: int = 1337
    eval_split: float = 0.0
    save_steps: int = 200
    log_steps: int = 25
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    @classmethod
    def small(cls, train_file: Path, output_dir: Path) -> "FineTuneConfig":
        return cls(
            train_file=train_file,
            output_dir=output_dir,
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            epochs=3,
            batch_size=2,
            grad_accum=2,
            max_length=256,
            bf16=False,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )


PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Output:\n{output}"
)


def _format_example(example: dict) -> str:
    return PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        input=example["input"],
        output=example["output"],
    )


def _load_jsonl(path: Path) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def run_training(cfg: FineTuneConfig) -> None:  # pragma: no cover - heavy
    """Execute the LoRA fine-tuning loop.

    Imports the heavy ``transformers`` / ``peft`` / ``datasets`` deps lazily
    so that ``deepparse`` itself remains importable when they aren't
    installed.
    """
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Heavy training deps missing. Install with:\n"
            "    pip install -e \".[train]\"\n"
            f"(import failed: {exc})"
        ) from exc

    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    examples = _load_jsonl(cfg.train_file)
    if not examples:
        raise SystemExit(f"No examples found in {cfg.train_file}")

    print(f"[finetune] {len(examples)} training examples")
    print(f"[finetune] base model: {cfg.model_name}")
    print(f"[finetune] LoRA rank={cfg.lora_rank} alpha={cfg.lora_alpha} dropout={cfg.lora_dropout}")
    print(f"[finetune] optimiser: AdamW lr={cfg.learning_rate} batch={cfg.batch_size} epochs={cfg.epochs}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Default: float32. Pass --bf16 to enable bf16 mixed precision.
    # bf16+LoRA is the paper's recipe (Section 'LLM Configuration') and
    # works on both NVIDIA (Ampere/Hopper) and AMD (MI200/MI300) GPUs;
    # we leave it opt-in because peft + bf16 occasionally produces NaN
    # gradients on certain torch / driver combinations regardless of
    # vendor.  When --bf16 is passed we also force the LoRA adapter
    # parameters to fp32 below, which is the standard mitigation.
    dtype = torch.bfloat16 if cfg.bf16 and torch.cuda.is_available() else torch.float32
    # attn_implementation="eager" is the safe portable choice: SDPA and
    # Flash-Attention both occasionally produce NaN gradients on AMD
    # ROCm builds (and on some older NVIDIA driver / torch combinations).
    # Eager attention is slower but trains correctly across vendors.
    # Pass --attn-impl=sdpa to opt back into SDPA when your stack is
    # verified stable.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=cfg.attn_implementation,
    )
    # gradient_checkpointing + LoRA needs use_cache=False AND
    # enable_input_require_grads().  Without the latter, gradients are
    # not registered on the inputs entering each checkpointed block, so
    # nothing flows back to the LoRA adapter — the symptom is the
    # textbook "loss = 0, grad_norm = NaN" pattern.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=cfg.target_modules,
    )
    model = get_peft_model(model, lora_config)
    # Force the LoRA adapter weights to float32 even when the base model
    # is bf16 — peft otherwise initialises them in the base dtype, which
    # is the single most common cause of "loss=0 grad_norm=NaN".
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.print_trainable_parameters()

    def _tokenize(batch):
        text = [_format_example(ex) + tokenizer.eos_token for ex in batch["__example"]]
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=cfg.max_length,
            padding=False,
        )
        encoded["labels"] = [ids.copy() for ids in encoded["input_ids"]]
        return encoded

    ds = Dataset.from_list([{"__example": ex} for ex in examples])
    ds = ds.map(_tokenize, batched=True, remove_columns=["__example"])

    # DataCollatorForSeq2Seq pads input_ids and labels independently to
    # the longest item in the batch, which is what we want for causal LM
    # fine-tuning with variable-length examples.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    train_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=cfg.log_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        bf16=cfg.bf16 and torch.cuda.is_available(),
        fp16=False,
        seed=cfg.seed,
        report_to=[],
    )

    # Transformers 5.x renamed `tokenizer=` to `processing_class=`. Probe at
    # runtime so the script keeps working across the 4.x → 5.x boundary.
    import inspect
    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": ds,
        "data_collator": collator,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))
    # Persist the training config alongside the adapter for traceability.
    (cfg.output_dir / "deepparse_finetune_config.json").write_text(
        json.dumps(
            {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[finetune] adapter saved to {cfg.output_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Training JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--small", action="store_true",
                        help="Use Qwen2.5-0.5B-Instruct + reduced hparams (CPU-friendly)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override base model name")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bf16", action="store_true",
                        help="Use bf16 mixed precision (default: float32 for ROCm stability)")
    parser.add_argument("--attn-impl", type=str, default=None,
                        choices=["eager", "sdpa", "flash_attention_2"],
                        help="Attention kernel (default: eager — cross-vendor safe; SDPA "
                             "and Flash-Attention occasionally produce NaN grads on AMD ROCm "
                             "and on torch built against a different ROCm version than the host).")
    args = parser.parse_args(argv)

    if args.small:
        cfg = FineTuneConfig.small(args.train, args.output_dir)
    else:
        cfg = FineTuneConfig(train_file=args.train, output_dir=args.output_dir)

    if args.model:
        cfg.model_name = args.model
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.max_length is not None:
        cfg.max_length = args.max_length
    if args.seed is not None:
        cfg.seed = args.seed
    if args.bf16:
        cfg.bf16 = True
    if args.attn_impl:
        cfg.attn_implementation = args.attn_impl

    run_training(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
