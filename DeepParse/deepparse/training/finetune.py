"""LoRA fine-tuning script for the DeepParse mask synthesiser.

Recipe (paper Section 'LLM Configuration and Fine-Tuning', adapted to small GPUs):

* Base model: ``deepseek-ai/DeepSeek-R1-Distill-Llama-8B`` (4-bit QLoRA, fits an
  8 GB GPU) or ``Qwen/Qwen2.5-3B-Instruct`` (16-bit LoRA, Colab T4).
* LoRA rank 8 / alpha 32 / dropout 0.01 on every attention and MLP projection.
* AdamW (paged 8-bit for 4-bit models), lr 2e-4, cosine schedule, 5 % warm-up,
  weight decay 0.01, 5 epochs over the whole training set, checkpoint every
  ``save_steps``.  No held-out split and no early stopping: the adapter is
  judged afterwards on GA/PA, not on eval loss.
* Completion-only loss: the prompt (instruction + log line, ~70 % of the
  tokens) is labelled ``-100`` so every gradient step is spent on the regex
  list the model has to emit.  ``--no-completion-only`` restores full-sequence
  loss.
* Dynamic padding with ``group_by_length``: a batch is padded to its own
  longest example (lengths range 50-450 tokens).

``--small`` swaps the base model to ``Qwen/Qwen2.5-0.5B-Instruct`` for CPU
smoke tests; ``--max-steps 3`` runs a few optimiser steps on the real model to
verify tokenisation, collator and saving.

Training data is a JSONL file with keys ``instruction``, ``input``, ``output``
as produced by :mod:`deepparse.tools.build_training_set`.

Usage:
    python -m deepparse.training.finetune --config configs/train_iot.yaml
    python -m deepparse.training.finetune --config configs/train_iot_qwen3b.yaml \\
        --output-dir /content/drive/MyDrive/checkpoints/deepparse-iot-qwen-3b --resume
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FineTuneConfig:
    train_file: Path
    output_dir: Path
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    epochs: float = 5
    learning_rate: float = 2e-4
    batch_size: int = 8
    grad_accum: int = 2
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.01
    max_length: int = 512
    # bf16 is used only when the GPU supports it natively (Ampere/Ada/Hopper);
    # otherwise the trainer falls back to fp16 (T4) or fp32 (CPU).
    bf16: bool = True
    attn_implementation: str = "sdpa"
    seed: int = 1337
    save_steps: int = 25
    log_steps: int = 10
    load_in_4bit: bool = False
    # None = auto: paged_adamw_8bit for 4-bit models, fused AdamW otherwise.
    optim: str | None = None
    completion_only: bool = True
    group_by_length: bool = True
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    # > 0 overrides epochs; handy for smoke tests.
    max_steps: int = -1
    # True = resume from the latest checkpoint in output_dir; a path = that checkpoint.
    resume_from_checkpoint: bool | str | None = None
    # Hard cap on this process's share of GPU memory (0-1).  On Windows the
    # driver otherwise spills to system RAM when VRAM runs out, which stalls
    # the whole machine instead of raising a clean OOM.
    gpu_memory_fraction: float | None = None
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    @classmethod
    def small(cls, train_file: Path, output_dir: Path) -> FineTuneConfig:
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

    @classmethod
    def from_yaml(cls, path: Path, train_file: Path | None, output_dir: Path | None) -> FineTuneConfig:
        """Build a config from a YAML file; CLI ``train_file``/``output_dir`` win when given."""
        from deepparse.utils.yaml_loader import load_yaml

        raw = dict(load_yaml(path))
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            print(f"[finetune] warning: ignoring unknown config keys {unknown}", file=sys.stderr)
        kwargs = {k: raw[k] for k in raw if k in known}
        for key in ("train_file", "output_dir"):
            if key in kwargs:
                kwargs[key] = Path(kwargs[key])
        if train_file is not None:
            kwargs["train_file"] = train_file
        if output_dir is not None:
            kwargs["output_dir"] = output_dir
        missing = [k for k in ("train_file", "output_dir") if k not in kwargs]
        if missing:
            raise SystemExit(f"{missing} must be given in {path} or on the command line")
        return cls(**kwargs)

    def to_json(self) -> dict:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(self).items()}


PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Output:\n{output}"
)


def _format_prompt(example: dict) -> str:
    """The part the model sees at inference time (everything before the answer)."""
    return PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        input=example["input"],
        output="",
    )


def _format_example(example: dict) -> str:
    return _format_prompt(example) + example["output"]


def _load_jsonl(path: Path) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def encode_example(tokenizer, example: dict, max_length: int, completion_only: bool) -> dict:
    """Tokenise one example into ``input_ids`` / ``attention_mask`` / ``labels``.

    Prompt and completion are tokenised separately and concatenated.  That is
    exactly what happens at inference (the prompt is encoded alone, the model
    then generates token by token), so there is no train/serve mismatch at the
    boundary.  With ``completion_only`` the prompt labels are ``-100``.
    """
    prompt_ids = tokenizer(_format_prompt(example), add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(example["output"], add_special_tokens=False)["input_ids"]
    completion_ids = completion_ids + [tokenizer.eos_token_id]

    # Never truncate the completion silently: cut the prompt from the left
    # first (the instruction text is identical in every example anyway).
    overflow = len(prompt_ids) + len(completion_ids) - max_length
    truncated = False
    if overflow > 0:
        truncated = True
        keep_prompt = max(len(prompt_ids) - overflow, 8)
        prompt_ids = prompt_ids[-keep_prompt:]
        room = max_length - len(prompt_ids)
        if len(completion_ids) > room:
            completion_ids = completion_ids[:room]

    input_ids = prompt_ids + completion_ids
    labels = ([-100] * len(prompt_ids) if completion_only else list(prompt_ids)) + completion_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "length": len(input_ids),
        "truncated": truncated,
    }


def _pick_optimizer(cfg: FineTuneConfig, cuda: bool) -> str:
    if cfg.optim:
        return cfg.optim
    if cfg.load_in_4bit:
        return "paged_adamw_8bit"
    return "adamw_torch_fused" if cuda else "adamw_torch"


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

    import inspect

    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    examples = _load_jsonl(cfg.train_file)
    if not examples:
        raise SystemExit(f"No examples found in {cfg.train_file}")

    cuda = torch.cuda.is_available()
    has_native_bf16 = cuda and torch.cuda.is_bf16_supported()
    use_bf16 = cfg.bf16 and has_native_bf16
    use_fp16 = cuda and not use_bf16
    if cuda and torch.cuda.get_device_capability()[0] >= 8:
        # Ampere+ : tf32 matmuls are free accuracy-wise for training and ~2x faster.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if cuda and cfg.gpu_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(float(cfg.gpu_memory_fraction), 0)
        cap = cfg.gpu_memory_fraction * torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"[finetune] GPU memory capped at {cap:.1f} GiB (fraction {cfg.gpu_memory_fraction})")

    optim = _pick_optimizer(cfg, cuda)
    device_name = torch.cuda.get_device_name(0) if cuda else "cpu"
    print(f"[finetune] device: {device_name} | precision: "
          f"{'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'} | 4-bit: {cfg.load_in_4bit}")
    print(f"[finetune] {len(examples)} examples | effective batch {cfg.batch_size * cfg.grad_accum}")
    print(f"[finetune] base model: {cfg.model_name}")
    print(f"[finetune] LoRA rank={cfg.lora_rank} alpha={cfg.lora_alpha} dropout={cfg.lora_dropout} "
          f"targets={cfg.target_modules}")
    print(f"[finetune] optimiser: {optim} lr={cfg.learning_rate} epochs={cfg.epochs} "
          f"completion_only={cfg.completion_only}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": cfg.attn_implementation,
    }
    if cfg.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit("Install: pip install bitsandbytes (QLoRA)") from exc

        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["torch_dtype"] = compute_dtype
        model_kwargs["device_map"] = {"": 0} if cuda else None
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
        # NOT peft.prepare_model_for_kbit_training: it upcasts every
        # non-quantised parameter to fp32, i.e. the 128k x 4096 embedding and
        # lm_head of Llama-3 (2 GB each) - the single allocation that blew the
        # 8 GB budget.  Only the small norm weights need fp32 for stability;
        # input grads are required for gradient checkpointing with frozen
        # embeddings, and checkpointing itself is enabled via TrainingArguments.
        for name, param in model.named_parameters():
            if "norm" in name.lower():
                param.data = param.data.float()
    else:
        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if use_bf16 else torch.float16 if cuda else torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False  # incompatible with gradient checkpointing

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=cfg.target_modules,
    )
    model = get_peft_model(model, lora_config)
    # Adapter weights in fp32 regardless of the base dtype: the standard
    # mitigation for NaN losses under fp16/bf16 mixed precision.
    for _, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.print_trainable_parameters()

    encoded = [encode_example(tokenizer, ex, cfg.max_length, cfg.completion_only) for ex in examples]
    n_trunc = sum(1 for e in encoded if e["truncated"])
    lengths = sorted(e["length"] for e in encoded)
    print(f"[finetune] token length p50={lengths[len(lengths) // 2]} "
          f"p95={lengths[int(len(lengths) * 0.95)]} max={lengths[-1]} | truncated: {n_trunc}")
    for e in encoded:
        e.pop("truncated")
    dataset = Dataset.from_list(encoded)

    # Pads input_ids / attention_mask / labels (with -100) to the longest item
    # in the batch.  Combined with group_by_length this keeps padding small.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    t_args_params = inspect.signature(TrainingArguments.__init__).parameters
    train_args_kwargs = {
        "output_dir": str(cfg.output_dir),
        "num_train_epochs": cfg.epochs,
        "max_steps": cfg.max_steps,
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "optim": optim,
        "logging_steps": cfg.log_steps,
        "save_strategy": "steps",
        "save_steps": cfg.save_steps,
        "save_total_limit": 2,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "group_by_length": cfg.group_by_length,
        "dataloader_pin_memory": cuda,
        "remove_unused_columns": True,
        "seed": cfg.seed,
        "report_to": [],
    }
    train_args_kwargs = {k: v for k, v in train_args_kwargs.items() if k in t_args_params}
    train_args = TrainingArguments(**train_args_kwargs)

    # Transformers 5.x renamed `tokenizer=` to `processing_class=`.
    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": dataset,
        "data_collator": collator,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    resume = cfg.resume_from_checkpoint
    if resume is True and not any(cfg.output_dir.glob("checkpoint-*")):
        print("[finetune] no checkpoint to resume from; starting fresh")
        resume = None
    train_result = trainer.train(resume_from_checkpoint=resume or None)
    print(f"[finetune] train metrics: {train_result.metrics}")
    if cuda:
        peak = torch.cuda.max_memory_allocated() / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"[finetune] peak GPU memory: {peak:.2f} GiB of {total:.1f} GiB")

    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))
    # Persist the training config alongside the adapter for traceability; the
    # synth step reads model_name / load_in_4bit back from it.
    (cfg.output_dir / "deepparse_finetune_config.json").write_text(
        json.dumps(cfg.to_json(), indent=2), encoding="utf-8",
    )
    print(f"[finetune] adapter saved to {cfg.output_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="YAML config file path")
    parser.add_argument("--train", type=Path, default=None,
                        help="Training JSONL (overrides train_file in the config)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Adapter output directory (overrides output_dir in the config)")
    parser.add_argument("--small", action="store_true",
                        help="Use Qwen2.5-0.5B-Instruct + reduced hparams (CPU-friendly)")
    parser.add_argument("--model", type=str, default=None, help="Override base model name")
    parser.add_argument("--qwen3b", action="store_true",
                        help="Shortcut: use Qwen/Qwen2.5-3B-Instruct as base model")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop after N optimiser steps (smoke tests)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--optim", type=str, default=None,
                        help="Trainer optimiser name (default auto: paged_adamw_8bit for 4-bit)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bf16", action="store_true", help="Prefer bf16 when the GPU supports it")
    parser.add_argument("--no-bf16", action="store_true", help="Force fp16/fp32 even on bf16 GPUs")
    parser.add_argument("--attn-impl", type=str, default=None,
                        choices=["eager", "sdpa", "flash_attention_2"],
                        help="Attention kernel (default sdpa; eager is the cross-vendor-safe fallback)")
    parser.add_argument("--load-in-4bit", "--4bit", action="store_true", dest="load_in_4bit",
                        help="Load the base model in 4-bit NormalFloat (QLoRA) to fit 8-16 GB GPUs")
    parser.add_argument("--no-completion-only", action="store_true",
                        help="Also compute loss on the prompt tokens (full-sequence LM loss)")
    parser.add_argument("--save-steps", type=int, default=None, help="Checkpoint every N steps")
    parser.add_argument("--gpu-memory-fraction", type=float, default=None,
                        help="Cap this process at a fraction of VRAM (e.g. 0.85) so it fails fast instead "
                             "of spilling to system RAM")
    parser.add_argument("--resume", nargs="?", const=True, default=None,
                        help="Resume from the latest checkpoint in --output-dir, or from the given path")
    args = parser.parse_args(argv)

    if args.config:
        cfg = FineTuneConfig.from_yaml(args.config, args.train, args.output_dir)
        if args.small:
            cfg = FineTuneConfig.small(cfg.train_file, cfg.output_dir)
    else:
        if args.train is None or args.output_dir is None:
            parser.error("--train and --output-dir are required without --config")
        cfg = (FineTuneConfig.small if args.small else FineTuneConfig)(args.train, args.output_dir)

    if args.load_in_4bit:
        cfg.load_in_4bit = True
    if args.qwen3b:
        cfg.model_name = "Qwen/Qwen2.5-3B-Instruct"
    if args.model:
        cfg.model_name = args.model
    for cli_name, cfg_name in [
        ("epochs", "epochs"), ("max_steps", "max_steps"), ("lr", "learning_rate"),
        ("batch_size", "batch_size"), ("grad_accum", "grad_accum"), ("lora_rank", "lora_rank"),
        ("lora_alpha", "lora_alpha"), ("max_length", "max_length"), ("optim", "optim"),
        ("seed", "seed"), ("attn_impl", "attn_implementation"), ("save_steps", "save_steps"),
        ("gpu_memory_fraction", "gpu_memory_fraction"),
    ]:
        value = getattr(args, cli_name)
        if value is not None:
            setattr(cfg, cfg_name, value)
    if args.bf16:
        cfg.bf16 = True
    if args.no_bf16:
        cfg.bf16 = False
    if args.no_completion_only:
        cfg.completion_only = False
    if args.resume is not None:
        cfg.resume_from_checkpoint = args.resume

    run_training(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
