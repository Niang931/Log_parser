"""Regression gate for the nightly evaluation.

Reads the JSON written by ``logpipe eval parsing|alias|iot`` under ``--out``
and compares three numbers with a committed baseline:

* parsing: mean GA over the Loghub-2k systems must not drop by more than
  ``tolerance.ga`` (mean PA is reported, not gated: it swings with the
  template conventions of individual systems);
* alias: the rule-based decider must still make **zero** wrong merges
  (``false_merges`` is a hard limit, default 0);
* IoT fleet: F1 of the ``full`` variant must not drop by more than
  ``tolerance.iot_f1``.

``logpipe eval gate --out artifacts/eval --baseline eval/baseline.json``
exits 1 on any regression and prints one line per check.
``--write-baseline`` records the current results as the new baseline instead
(run it after an intentional improvement, then commit the file).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASELINE = Path("eval/baseline.json")
DEFAULT_TOLERANCE = {"ga": 0.02, "iot_f1": 0.05}
ALIAS_DECIDER = "rules_offline"


@dataclass
class Check:
    name: str
    baseline: float | None
    current: float | None
    ok: bool
    detail: str

    def line(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        base = "n/a" if self.baseline is None else f"{self.baseline:.4f}"
        cur = "n/a" if self.current is None else f"{self.current:.4f}"

        return f"[{mark}] {self.name}: baseline {base} -> current {cur}  ({self.detail})"


def _load(path: Path) -> dict | list | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def collect(out_dir: Path) -> dict:
    """Pull the gated numbers out of an ``artifacts/eval`` tree (missing parts stay absent)."""
    result: dict = {}

    parsing = _load(out_dir / "parsing.json")
    if parsing:
        rows = [r for r in parsing if r.get("GA") is not None]
        result["parsing"] = {
            "systems": len(rows),
            "mean_GA": round(statistics.mean(float(r["GA"]) for r in rows), 4),
            "mean_PA": round(statistics.mean(float(r["PA"]) for r in rows), 4),
        }

    alias = _load(out_dir / "alias.json")
    if alias:
        decider = alias["deciders"].get(ALIAS_DECIDER, {})
        result["alias"] = {
            "decider": ALIAS_DECIDER,
            "false_merges": int(decider.get("FP", 0)),
            "missed_merges": int(decider.get("FN", 0)),
            "pairs": alias.get("pairs"),
        }

    iot = _load(out_dir / "iot" / "results.json")
    if iot:
        full = iot["variants"]["full"]["test"]
        result["iot"] = {
            "F1": round(float(full["F1"]), 4),
            "P": round(float(full["P"]), 4),
            "R": round(float(full["R"]), 4),
        }
    
    return result


def compare(current: dict, baseline: dict) -> list[Check]:
    tol = {**DEFAULT_TOLERANCE, **baseline.get("tolerance", {})}
    checks: list[Check] = []

    base, cur = baseline.get("parsing"), current.get("parsing")
    if base and cur:
        drop = base["mean_GA"] - cur["mean_GA"]
        checks.append(
            Check(
                "parsing mean GA",
                base["mean_GA"],
                cur["mean_GA"],
                drop <= tol["ga"],
                f"allowed drop {tol['ga']}, PA {cur['mean_PA']:.4f}",
            )
        )
    elif base:
        checks.append(
            Check(
                "parsing mean GA",
                base["mean_GA"],
                None,
                False,
                "parsing.json missing",
            )
        )

    base, cur = baseline.get("alias"), current.get("alias")
    if base and cur:
        limit = int(base.get("false_merges", 0))
        checks.append(
            Check(
                "alias wrong merges",
                float(limit),
                float(cur["false_merges"]),
                cur["false_merges"] <= limit,
                f"{ALIAS_DECIDER}, hard limit {limit}",
            )
        )
    elif base:
        checks.append(
            Check(
                "alias wrong merges",
                float(base.get("false_merges", 0)),
                None,
                False,
                "alias.json missing",
            )
        )

    base, cur = baseline.get("iot"), current.get("iot")
    if base and cur:
        drop = base["F1"] - cur["F1"]
        checks.append(
            Check(
                "IoT fleet F1 (full)",
                base["F1"],
                cur["F1"],
                drop <= tol["iot_f1"],
                f"allowed drop {tol['iot_f1']}, P {cur['P']:.3f} R {cur['R']:.3f}",
            )
        )
    elif base:
        checks.append(
            Check(
                "IoT fleet F1 (full)",
                base["F1"],
                None,
                False,
                "iot/results.json missing",
            )
        )
    
    return checks


def write_baseline(current: dict, path: Path, params: dict | None = None) -> Path:
    payload = {"tolerance": DEFAULT_TOLERANCE, **current}
    if params:
        payload["eval_params"] = params
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return path


def run_gate(out_dir: Path, baseline_path: Path = DEFAULT_BASELINE) -> int:
    """Print one line per check; return the process exit code (0 = all pass)."""
    baseline = _load(baseline_path)
    if not baseline:
        print(f"[gate] no baseline at {baseline_path}; run with --write-baseline first")
        return 2
    
    checks = compare(collect(out_dir), baseline)
    if not checks:
        print(f"[gate] nothing to compare under {out_dir}")
        return 2
    
    for check in checks:
        print(check.line())
    
    failed = [c for c in checks if not c.ok]
    print(f"[gate] {len(checks) - len(failed)}/{len(checks)} checks passed")
    
    return 1 if failed else 0
