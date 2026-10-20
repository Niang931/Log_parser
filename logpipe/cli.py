"""``logpipe`` command line.

* ``calibrate``: build registry v1 from historical logs (+ evaluation report).
* ``replay``: parse files with the frozen registry only.
* ``run``: full pipeline (store, enrichment loop, LogBERT detector).
* ``worker``: learning worker consuming the unresolved queue (Kafka consumer group).
* ``train-logbert``: train the detector on per-device windows of parsed logs.
* ``gen-iot``: write the synthetic IoT fleet to disk.
* ``eval``: ``parsing`` | ``logbert`` | ``iot`` | ``alias`` benchmarks, ``report`` to summarise.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from deepparse.ai_scraper.session import BROWSERS

from .calibrate import GroundTruth, calibrate
from .config import Settings
from .enrich.llm import get_adapter
from .parsing.frozen_parser import FrozenParser, ParserHandle
from .registry import SchemaRegistry
from .sources.file_replay import replay


def _calibrate(args: argparse.Namespace) -> None:
    truth = None
    if args.dataset:
        lines, truth = GroundTruth.from_deepparse_dir(args.dataset)
    else:
        lines = [event.line for event in replay(args.logs)]
    result = calibrate(
        lines, get_adapter(args.llm, browser=args.browser), sample_size=args.sample, ground_truth=truth, tune=args.tune,
        depth=args.depth, similarity_threshold=args.similarity,
    )
    SchemaRegistry(result.snapshot).save(args.registry)
    report_path = args.registry.with_suffix(".report.json")
    report_path.write_text(json.dumps(result.report.to_json(), indent=2), encoding="utf-8")
    r = result.report
    print(f"registry v{result.snapshot.version}: {r.templates} templates, {len(r.masks)} masks "
          f"from {r.lines} lines -> {args.registry}")
    print(f"holdout coverage {r.holdout_coverage:.1%} (extraction {r.holdout_extraction_rate:.1%}) "
          f"on {r.holdout_lines} held-out lines; depth={r.depth} similarity={r.similarity_threshold}")
    if r.ga is not None:
        print(f"GA={r.ga} PA={r.pa} (DeepParse baseline GA={r.baseline_ga} PA={r.baseline_pa})")
    print(f"report -> {report_path}")


def _replay(args: argparse.Namespace) -> None:
    handle = ParserHandle(SchemaRegistry.load(args.registry).current)
    statuses: Counter[str] = Counter()
    for event in replay(args.logs):
        result = handle.parse(event.line)
        statuses[result.status] += 1
        if not args.quiet:
            print(json.dumps({
                "device_id": event.device_id,
                "template_id": result.template_id,
                "status": result.status,
                "payload": result.payload,
            }))
    total = sum(statuses.values()) or 1
    print(
        f"parsed {total} lines with registry v{handle.version}: "
        f"{statuses['FINAL']} FINAL, {statuses['PENDING']} PENDING "
        f"({statuses['PENDING'] / total:.1%} pending)"
    )


def _run(args: argparse.Namespace) -> None:
    from .anomaly.detector import DetectorConfig
    from .anomaly.scorer import LogBERTScorer
    from .pipeline import Pipeline
    from .storage import open_store
    from .unresolved import open_queue

    store = open_store(args.store)
    queue = open_queue(args.queue)
    has_model = args.model and (args.model / "config.json").exists()
    pipeline = Pipeline.assemble(
        SchemaRegistry.load(args.registry), store,
        adapter=None if args.no_enrich else get_adapter(args.llm, browser=args.browser),
        scorer=LogBERTScorer.load(args.model) if has_model else None,
        detector_config=DetectorConfig(window=args.window, stride=args.stride),
        registry_path=args.registry,
        enrich_every=args.enrich_every,
        queue=queue,
    )
    if pipeline.worker is not None and args.rehydrate:
        print(f"rehydrated {pipeline.worker.rehydrate()} pending rows from the store")
    stats = pipeline.run(replay(args.logs))
    queue.close()
    store.close()
    print(json.dumps(stats.summary(), indent=2, default=str))
    for alert in stats.alerts[: args.show_alerts]:
        print(f"ALERT {alert.device_id} {alert.window_start:%H:%M:%S}-{alert.window_end:%H:%M:%S} "
              f"reasons={','.join(alert.reasons)} miss={alert.miss_ratio:.2f} "
              f"novelty={alert.novelty_ratio:.2f}")


def _worker(args: argparse.Namespace) -> None:
    """Consume the unresolved queue until ``--max-cycles`` or Stop."""
    import time

    from .enrich.worker import EnrichmentWorker
    from .storage import open_store
    from .unresolved import open_queue

    registry = SchemaRegistry.load(args.registry)
    handle = ParserHandle(registry.current)
    store = open_store(args.store)
    queue = open_queue(args.queue)
    worker = EnrichmentWorker(
        registry, handle, store, get_adapter(args.llm, browser=args.browser),
        registry_path=args.registry, queue=queue, max_buffer=args.batch,
        max_attempts=args.max_attempts, consume_timeout=args.poll,
    )
    print(f"worker: registry v{handle.version}, queue {args.queue}, store {args.store}")

    cycles = 0
    try:
        while args.max_cycles is None or cycles < args.max_cycles:
            report = worker.run_cycle()
            cycles += 1
            if report.buffered:
                print(json.dumps({
                    "cycle": cycles, "buffered": report.buffered, "clusters": report.clusters,
                    "published": report.published, "aliased": report.aliased,
                    "rejected": len(report.rejected), "resolved": report.resolved,
                    "still_pending": report.still_pending, "dead_lettered": report.dead_lettered,
                    "registry_version": report.registry_version, "queue_lag": report.queue_lag,
                    "llm_calls": report.llm_calls, "seconds": round(report.seconds, 2),
                }))
            elif args.interval:
                time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("worker: stopping")
    finally:
        queue.close()
        store.close()
    print(json.dumps({"cycles": cycles, "queue": worker.queue.stats().to_json()}))


def _train_logbert(args: argparse.Namespace) -> None:
    from .anomaly.detector import sliding_windows
    from .anomaly.logbert import LogBERTConfig, TrainConfig, train_logbert
    from .anomaly.scorer import LogBERTScorer
    from .anomaly.vocab import TemplateVocab

    parser = FrozenParser(SchemaRegistry.load(args.registry).current)
    per_device: dict[str, list[int]] = defaultdict(list)
    for event in replay(args.logs):
        per_device[event.device_id].append(int(parser.parse(event.line).template_id))
    windows = [w for seq in per_device.values() for w in sliding_windows(seq, args.window, args.stride)]
    if not windows:
        raise SystemExit("no windows to train on")
    vocab = TemplateVocab.build(windows)
    encoded = [vocab.encode(w) for w in windows]
    random.Random(0).shuffle(encoded)
    n_valid = max(1, len(encoded) // 10)
    trained = train_logbert(
        encoded[n_valid:], encoded[:n_valid],
        LogBERTConfig(len(vocab), args.hidden, args.layers, args.heads, max_len=args.window + 1),
        TrainConfig(epochs=args.epochs),
    )
    scorer = LogBERTScorer.from_training(trained, vocab, version=args.out.name)
    scorer.save(args.out)
    print(f"trained on {len(windows)} windows from {len(per_device)} devices, "
          f"vocab {len(vocab)} -> {args.out}")
    print("thresholds are defaults; tune them with labelled data (see `logpipe eval iot`)")


def _gen_iot(args: argparse.Namespace) -> None:
    from .eval.synth_iot import generate

    fleet = generate(devices_per_kind=args.devices, history_hours=args.history_hours,
                     test_hours=args.test_hours, seed=args.seed)
    fleet.write(args.out)
    print(f"wrote history/validation/test logs and labels.tsv to {args.out}")


def _fetch_data(args: argparse.Namespace) -> None:
    from .eval.datasets import fetch_archive, fetch_loghub2k

    for name in args.datasets:
        path = fetch_loghub2k() if name == "loghub2k" else fetch_archive(name)
        print(f"{name} -> {path}")


def _eval(args: argparse.Namespace) -> None:
    out = args.out
    adapter = get_adapter(args.llm, browser=args.browser)
    if args.benchmark == "report":
        from .eval.report import write_report

        print(f"report -> {write_report(out)}")
    elif args.benchmark == "parsing":
        from .eval.parsing_eval import run_parsing_eval

        run_parsing_eval(adapter, tune=not args.no_tune, out_dir=out)
    elif args.benchmark == "alias":
        from .eval.alias_eval import run_alias_eval

        result = run_alias_eval()
        out.mkdir(parents=True, exist_ok=True)
        (out / "alias.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        for name, r in result["deciders"].items():
            print(f"[alias] {name}: precision={r['precision']} recall={r['recall']} "
                  f"(TP={r['TP']} FP={r['FP']} FN={r['FN']})")
    elif args.benchmark == "logbert":
        from .anomaly.logbert import TrainConfig
        from .eval.datasets import load_bgl, load_hdfs
        from .eval.logbert_eval import LogBERTEvalConfig, run_logbert_eval

        cfg = LogBERTEvalConfig(train=TrainConfig(epochs=args.epochs, threads=args.threads))
        if args.dataset == "HDFS":
            sessions = load_hdfs(n_normal=args.normal, n_abnormal=args.abnormal)
        else:
            sessions = load_bgl(max_lines=args.max_lines)
            cfg = replace(cfg, num_candidates=15)  # LogBERT's BGL setting
        run_logbert_eval(sessions, adapter, cfg, out_dir=out / args.dataset.lower())
    elif args.benchmark == "iot":
        from .anomaly.logbert import TrainConfig
        from .eval.iot_eval import IoTEvalConfig, run_iot_eval

        run_iot_eval(adapter, IoTEvalConfig(
            devices_per_kind=args.devices, test_hours=args.test_hours,
            train=TrainConfig(epochs=args.epochs, threads=args.threads, patience=6),
        ), out_dir=out / "iot")


def main(argv: list[str] | None = None) -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(prog="logpipe")
    parser.add_argument("--browser", choices=BROWSERS, default=settings.web_browser,
                        help="browser for LLM backend llm:web:<site>")
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calibrate", help="build registry v1 from historical logs")
    cal.add_argument("--logs", type=Path, default=settings.logs_path)
    cal.add_argument("--dataset", type=Path, help="DeepParse dataset dir (raw.log + templates.json)")
    cal.add_argument("--registry", type=Path, default=settings.registry_path)
    cal.add_argument("--llm", default=settings.llm_backend)
    cal.add_argument("--sample", type=int, default=200)
    cal.add_argument("--depth", type=int, default=5)
    cal.add_argument("--similarity", type=float, default=0.4)
    cal.add_argument("--tune", action="store_true", help="grid-search Drain params (needs --dataset)")
    cal.set_defaults(func=_calibrate)

    rep = sub.add_parser("replay", help="parse log files with the frozen registry")
    rep.add_argument("--logs", type=Path, default=settings.logs_path)
    rep.add_argument("--registry", type=Path, default=settings.registry_path)
    rep.add_argument("--quiet", action="store_true", help="print only the summary")
    rep.set_defaults(func=_replay)

    run = sub.add_parser("run", help="full pipeline: parse, store, enrich, detect")
    run.add_argument("--logs", type=Path, default=settings.logs_path)
    run.add_argument("--registry", type=Path, default=settings.registry_path)
    run.add_argument("--store", choices=["memory", "clickhouse"], default=settings.store)
    run.add_argument("--queue", choices=["memory", "kafka"], default=settings.queue, help="unresolved queue; kafka + --no-enrich is the publish-only fast path")
    run.add_argument("--llm", default=settings.llm_backend)
    run.add_argument("--model", type=Path, default=settings.model_path)
    run.add_argument("--enrich-every", type=int, default=5_000)
    run.add_argument("--no-enrich", action="store_true", help="no learning in this process (use `logpipe worker`)")
    run.add_argument("--rehydrate", action="store_true", help="reload PENDING rows from the store")
    run.add_argument("--window", type=int, default=20)
    run.add_argument("--stride", type=int, default=5)
    run.add_argument("--show-alerts", type=int, default=20)
    run.set_defaults(func=_run)

    wk = sub.add_parser("worker", help="learning worker: consume the unresolved queue")
    wk.add_argument("--registry", type=Path, default=settings.registry_path)
    wk.add_argument("--store", choices=["memory", "clickhouse"], default=settings.store)
    wk.add_argument("--queue", choices=["memory", "kafka"], default="kafka")
    wk.add_argument("--llm", default=settings.llm_backend)
    wk.add_argument("--batch", type=int, default=10_000, help="rows per cycle")
    wk.add_argument("--poll", type=float, default=1.0, help="seconds to wait for a batch")
    wk.add_argument("--interval", type=float, default=2.0, help="sleep after an empty cycle")
    wk.add_argument("--max-attempts", type=int, default=3, help="failed validations before a row goes to the dead-letter topic")
    wk.add_argument("--max-cycles", type=int, default=None, help="stop after N cycles (default: run forever)")
    wk.set_defaults(func=_worker)

    tl = sub.add_parser("train-logbert", help="train LogBERT on per-device windows")
    tl.add_argument("--logs", type=Path, default=settings.logs_path)
    tl.add_argument("--registry", type=Path, default=settings.registry_path)
    tl.add_argument("--out", type=Path, default=settings.model_path)
    tl.add_argument("--window", type=int, default=20)
    tl.add_argument("--stride", type=int, default=5)
    tl.add_argument("--hidden", type=int, default=64)
    tl.add_argument("--layers", type=int, default=2)
    tl.add_argument("--heads", type=int, default=4)
    tl.add_argument("--epochs", type=int, default=30)
    tl.set_defaults(func=_train_logbert)

    gen = sub.add_parser("gen-iot", help="write the synthetic IoT fleet")
    gen.add_argument("--out", type=Path, default=Path("artifacts/iot_fleet"))
    gen.add_argument("--devices", type=int, default=5, help="devices per kind")
    gen.add_argument("--history-hours", type=float, default=12)
    gen.add_argument("--test-hours", type=float, default=6)
    gen.add_argument("--seed", type=int, default=7)
    gen.set_defaults(func=_gen_iot)

    fd = sub.add_parser("fetch-data", help="download the evaluation datasets up front")
    fd.add_argument("datasets", nargs="*", choices=["loghub2k", "HDFS", "BGL"],
                    default=["loghub2k", "HDFS", "BGL"])
    fd.set_defaults(func=_fetch_data)

    ev = sub.add_parser("eval", help="run a benchmark")
    ev.add_argument("benchmark", choices=["parsing", "logbert", "iot", "alias", "report"])
    ev.add_argument("--out", type=Path, default=Path("artifacts/eval"))
    ev.add_argument("--llm", default=settings.llm_backend)
    ev.add_argument("--no-tune", action="store_true")
    ev.add_argument("--dataset", choices=["HDFS", "BGL"], default="HDFS")
    ev.add_argument("--normal", type=int, default=20_000)
    ev.add_argument("--abnormal", type=int, default=2_000)
    ev.add_argument("--max-lines", type=int, default=1_000_000)
    ev.add_argument("--devices", type=int, default=5)
    ev.add_argument("--test-hours", type=float, default=6)
    ev.add_argument("--epochs", type=int, default=30)
    ev.add_argument("--threads", type=int, default=None)
    ev.set_defaults(func=_eval)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
