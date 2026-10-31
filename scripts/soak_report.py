#!/usr/bin/env python3
"""Turn the metrics JSONL of `logpipe load` into the numbers for docs/evaluation.md.

    scripts/soak_report.py artifacts/chaos/<run>/metrics.jsonl [--workers w1.log w2.log]

Checks (exit 1 if one fails):
  * RSS growth over the last 20/24 of the run is below 5 %
  * queue lag of the workers does not trend upwards (second half vs first half)
"""

import argparse
import json
import statistics
import sys
from pathlib import Path


def _median_at(values: list[float], start: int, size: int) -> float:
    return statistics.median(values[start : start + size])


def _lines(path: Path):
    for text in path.read_text().splitlines():
        try:
            yield json.loads(text)
        except json.JSONDecodeError:  # progress text or a half-written last line
            continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", type=Path)
    ap.add_argument("--workers", type=Path, nargs="*", default=[], help="worker stdout logs")
    ap.add_argument("--tail", type=float, default=20 / 24, help="share of the run to judge RSS on")
    ap.add_argument("--max-rss-growth", type=float, default=0.05)
    args = ap.parse_args()

    rows = list(_lines(args.metrics))
    if len(rows) < 10:
        sys.exit(f"{args.metrics}: {len(rows)} samples, need at least 10")
    hours = rows[-1]["t"] / 3600

    rss = [r["rss_mb"] for r in rows if r.get("rss_mb")]
    edge = max(1, len(rss) // 20)  # medians of the first and last 5 % hide single spikes
    begin = int(len(rss) * (1 - args.tail))
    growth = _median_at(rss, len(rss) - edge, edge) / _median_at(rss, begin, edge) - 1

    rate = [r["lines_per_s"] for r in rows if r["lines_per_s"] > 0]
    p99 = [r["parse_us"]["p99"] for r in rows if r.get("parse_us")]
    pending_age = [r["oldest_pending_s"] for r in rows]
    result = {
        "hours": round(hours, 2),
        "samples": len(rows),
        "generated": rows[-1]["generated"],
        "lines_per_s_median": round(statistics.median(rate), 1),
        "parse_p99_us_max": max(p99) if p99 else None,
        "rss_mb_start": rss[0], "rss_mb_end": rss[-1], "rss_mb_max": max(rss),
        f"rss_growth_last_{args.tail:.0%}": round(growth, 4),
        "oldest_pending_s_max": max(pending_age),
        "status_end": rows[-1]["status"],
    }
    ok = growth < args.max_rss_growth

    lags = [e["queue_lag"] for w in args.workers for e in _lines(w) if e.get("queue_lag") is not None]
    if lags:
        half = len(lags) // 2
        first, second = statistics.mean(lags[:half]), statistics.mean(lags[half:])
        result |= {"queue_lag_max": max(lags), "queue_lag_mean_first_half": round(first, 1),
                   "queue_lag_mean_second_half": round(second, 1)}
        # 25 % + 100 rows of slack so noise around an empty queue is not a trend
        ok = ok and second <= first * 1.25 + 100
    result["pass"] = ok
    print(json.dumps(result, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
