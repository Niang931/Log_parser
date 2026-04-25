"""Command line entrypoints for DeepParse artifact."""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
from typing import Iterable, Optional

import click

from .dataset_loader import load_dataset
from .io_paths import build_paths
from .logging_utils import configure_logging, get_logger
from .masks_types import Mask
from .seeds import resolve_seed, set_global_seed
from .synth import synthesize_masks
from .utils.yaml_loader import load_yaml

LOGGER = get_logger(__name__)


@click.group()
@click.option("--log-dir", default="artifacts/outputs/logs", type=click.Path())
@click.option("--log-name", default="deepparse", type=str)
@click.pass_context
def cli(ctx: click.Context, log_dir: str, log_name: str) -> None:
    configure_logging(log_dir, log_name)
    ctx.ensure_object(dict)


def _load_base_config(path: str) -> dict:
    return load_yaml(path)


@cli.command()
@click.option("--dataset", type=str, required=False)
@click.option("--config", type=click.Path(), required=False)
@click.option("--k", type=int, default=None)
@click.option("--out", type=click.Path(), required=False)
@click.option("--mode", type=click.Choice(["offline", "hf"]), default="offline")
@click.option("--model", "model_name", type=str, default=None,
              help="Override base model name (default: deepseek-ai/DeepSeek-R1-Distill-Llama-8B)")
@click.option("--adapter", type=click.Path(), default=None,
              help="Path to a fine-tuned LoRA adapter directory")
@click.option("--strict", is_flag=True, default=False)
@click.option("--seed", type=int, default=None)
@click.pass_context
def synth(ctx: click.Context, dataset: Optional[str], config: Optional[str], k: Optional[int],
          out: Optional[str], mode: str, model_name: Optional[str], adapter: Optional[str],
          strict: bool, seed: Optional[int]) -> None:
    base = _load_base_config("configs/default.yaml")

    # Config here means the config yaml file path
    if config:
        conf_data = load_yaml(config)
        if "base_config" in conf_data:
            base = _load_base_config(conf_data["base_config"])
        datasets = conf_data.get("datasets", [])
    else:
        datasets = [dataset or "DemoTiny"]

    # Do perform the 16 dataset evaluation
    if dataset == "ALL":
        conf_data = load_yaml(config or "configs/eval_16_datasets.yaml")
        datasets = conf_data.get("datasets", [])

    seed = resolve_seed(seed or base.get("seed"))
    set_global_seed(seed)

    # TODO: question now is that what happens when we do not want to regenerate everything while appendng to the template list
    # Set the path and log sample size before synthesizing the masks for all datasets
    paths = build_paths(base["dataset_dir"], base["mask_dir"], base["output_dir"], base["log_dir"])
    k = k or base.get("k", 50)
    for name in datasets:
        dataset_obj = load_dataset(name, paths)
        out_path = Path(out or paths.mask_dir / f"{name}.json")
        synthesize_masks(
            dataset_obj, k, out_path,
            mode=mode, strict=strict,
            model_name=model_name, adapter_path=adapter,
        )


@cli.command()
@click.option("--dataset", required=True, type=str)
@click.option("--output", type=click.Path(), required=False)
@click.option("--seed", type=int, default=None)
@click.pass_context
def parse(ctx: click.Context, dataset: str, output: Optional[str], seed: Optional[int]) -> None:
    base = _load_base_config("configs/default.yaml")
    paths = build_paths(base["dataset_dir"], base["mask_dir"], base["output_dir"], base["log_dir"])
    dataset_obj = load_dataset(dataset, paths)
    mask_path = paths.mask_dir / f"{dataset}.json"
    if not mask_path.exists():
        raise click.ClickException(f"Mask file missing at {mask_path}")
    masks_data = json.loads(mask_path.read_text(encoding="utf-8"))
    masks = [Mask(**entry) for entry in masks_data]
    from .drain.drain_engine import DrainEngine

    engine = DrainEngine(masks=masks)
    templates = engine.parse(dataset_obj.logs)
    output_path = Path(output or paths.output_dir / f"{dataset}_parsed.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        # Use the stdlib CSV writer so log/template strings containing
        # ", \, or newlines are properly escaped (manual f-string
        # quoting silently corrupted rows on logs with embedded quotes).
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
        writer.writerow(["log", "template"])
        for log, template in zip(dataset_obj.logs, templates):
            writer.writerow([log, template])
    click.echo(f"Wrote parsed templates to {output_path}")


@cli.command()
@click.option("--config", type=click.Path(), required=True)
@click.option("--deterministic", is_flag=True, default=False)
@click.option("--seed", type=int, default=None)
@click.pass_context
def eval(ctx: click.Context, config: str, deterministic: bool, seed: Optional[int]) -> None:
    from .evaluation.eval_runner import EvaluationRunner

    runner = EvaluationRunner(Path(config))
    if seed is not None:
        runner.seed = seed
    runner.run()
    click.echo(f"Wrote metrics CSV to {runner.config.output_csv}")


@cli.command()
@click.option("--inputs", multiple=True, type=click.Path())
@click.option("--out", type=click.Path(), required=True)
@click.pass_context
def table(ctx: click.Context, inputs: Iterable[str], out: str) -> None:
    from .evaluation.tables import build_tables

    paths = [Path(path) for path in (inputs or glob.glob("artifacts/outputs/*.csv"))]
    build_tables(paths, Path(out))


@cli.command()
@click.option("--dataset", required=True)
@click.option("--n", type=int, default=100)
@click.option("--config", type=click.Path(), default="configs/default.yaml")
@click.pass_context
def time(ctx: click.Context, dataset: str, n: int, config: str) -> None:
    from .evaluation.timing_bench import run_timing_benchmark

    base = _load_base_config(config)
    paths = build_paths(base["dataset_dir"], base["mask_dir"], base["output_dir"], base["log_dir"])
    mask_path = paths.mask_dir / f"{dataset}.json"
    if not mask_path.exists():
        raise click.ClickException(f"Mask file missing at {mask_path}")
    output_csv = Path(base.get("timing_csv", "artifacts/outputs/timing.csv"))
    run_timing_benchmark(dataset, paths, mask_path, n, output_csv)


if __name__ == "__main__":  # pragma: no cover
    cli()
