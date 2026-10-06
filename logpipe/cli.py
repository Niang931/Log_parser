"""``logpipe`` command line: ``calibrate`` builds registry v1, ``replay`` parses files."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .calibrate import calibrate
from .config import Settings
from .enrich.llm import get_adapter
from .parsing.frozen_parser import ParserHandle
from .registry import SchemaRegistry
from .sources.file_replay import replay


def _calibrate(args: argparse.Namespace) -> None:
    lines = [event.line for event in replay(args.logs)]
    snapshot = calibrate(lines, get_adapter(args.llm), sample_size=args.sample)
    registry = SchemaRegistry(snapshot)
    registry.save(args.registry)
    print(
        f"registry v{snapshot.version}: {len(snapshot.templates)} templates, "
        f"{len(snapshot.masks)} masks from {len(lines)} lines -> {args.registry}"
    )


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


def main(argv: list[str] | None = None) -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(prog="logpipe")
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calibrate", help="build registry v1 from historical logs")
    cal.add_argument("--logs", type=Path, default=settings.logs_path)
    cal.add_argument("--registry", type=Path, default=settings.registry_path)
    cal.add_argument("--llm", default=settings.llm_backend)
    cal.add_argument("--sample", type=int, default=200)
    cal.set_defaults(func=_calibrate)

    rep = sub.add_parser("replay", help="parse log files with the frozen registry")
    rep.add_argument("--logs", type=Path, default=settings.logs_path)
    rep.add_argument("--registry", type=Path, default=settings.registry_path)
    rep.add_argument("--quiet", action="store_true", help="print only the summary")
    rep.set_defaults(func=_replay)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
