"""Collect ``artifacts/eval/*`` results into one markdown report."""
from __future__ import annotations

import json
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in rows]
    return "\n".join(out)


def _parsing(root: Path) -> str:
    tuned, untuned = _load(root / "parsing.json"), _load(root / "parsing_untuned" / "parsing.json")
    if not tuned:
        return ""
    plain = {r["system"]: r for r in untuned or []}
    rows = []
    for r in tuned:
        u = plain.get(r["system"], {})
        rows.append([r["system"], r.get("baseline_GA"), u.get("GA"), r["GA"],
                     r.get("baseline_PA"), u.get("PA"), r["PA"], r.get("holdout_coverage")])
    return "## Parsing: Loghub-2k (DeepParse GA/PA)\n\n" + _table(
        ["system", "GA DeepParse", "GA logpipe", "GA logpipe+tuned",
         "PA DeepParse", "PA logpipe", "PA logpipe+tuned", "holdout coverage"], rows)


def _alias(root: Path) -> str:
    data = _load(root / "alias.json")
    if not data:
        return ""
    rows = [[name, r["precision"], r["recall"], r["FP"], r["FN"]]
            for name, r in data["deciders"].items()]
    return (f"## Alias decisions ({data['pairs']} labelled pairs)\n\n"
            + _table(["decider", "precision", "recall", "false merges", "missed merges"], rows))


def _logbert(root: Path, name: str) -> str:
    data = _load(root / name / "results.json")
    if not data:
        return ""
    rows = []
    for r in data["results"]:
        t, o = r["test"], r["oracle_test"]
        rows.append([r["unknown_mode"], "on" if r["novelty"] else "off",
                     t["P"], t["R"], t["F1"], o["F1"]])
    seqs = data["sequences_after_min_len"]
    return (
        f"## LogBERT: {data['dataset']}\n\n"
        f"Sequences after min_len: {seqs}. Unknown-template share: {data['unknown_share']}. "
        f"Vocab {data['vocab_size']}, {data['train']['epochs']} epochs in {data['train']['seconds']} s. "
        f"Parse latency per unique line: {data['parse_latency_us']} µs.\n\n"
        + _table(["unknowns", "novelty", "P", "R", "F1 (val-tuned)", "F1 (oracle, LogBERT protocol)"], rows)
    )


def _iot(root: Path) -> str:
    data = _load(root / "iot" / "results.json")
    if not data:
        return ""
    rows = []
    for name, v in data["variants"].items():
        t, e = v["test"], v["enrichment"]
        rows.append([name, t["P"], t["R"], t["F1"], t["false_positive_reasons"],
                     e["pending_rows_at_end"], e["resolve_lag_s"].get("p50"),
                     v["pipeline"]["adapter_calls"]])
    full = data["variants"].get("full", {})
    extra = ""
    if full:
        extra = (f"\n\nRecall by anomaly kind (full): {full['test']['recall_by_kind']}\n\n"
                 f"Parse latency (full): {full['pipeline']['parse_latency_us']} µs, "
                 f"{full['pipeline']['lines_per_second']} lines/s end to end.")
    return ("## IoT fleet, end to end\n\n" + f"Events: {data['events']}.\n\n"
            + _table(["variant", "P", "R", "F1", "FP reasons", "PENDING at end",
                      "resolve lag p50 (s)", "adapter calls"], rows) + extra)


def write_report(root: Path = Path("artifacts/eval")) -> Path:
    sections = [_parsing(root), _alias(root), _iot(root),
                _logbert(root, "hdfs"), _logbert(root, "bgl")]
    out = root / "REPORT.md"
    out.write_text("# Evaluation report\n\n" + "\n\n".join(s for s in sections if s) + "\n",
                   encoding="utf-8")
    return out
