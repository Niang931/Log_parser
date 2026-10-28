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


INTRO = """\
**logpipe** is this repo's whole log pipeline: DeepParse regex masks + Drain
mine templates from history (calibration), a frozen parser matches new lines
against those templates, an LLM enrichment loop resolves lines no template
matched (PENDING), and LogBERT flags anomalous windows of template ids.
**DeepParse** is the upstream parser on its own, with its default settings.
All scores are 0-1; higher is better unless the caption says otherwise."""

PARSING_CAPTION = """\
Can we turn raw log lines into the right templates? 2,000 labelled lines per
Loghub system; the last row is the average over systems.

- **GA (grouping accuracy)**: share of lines whose predicted group holds
  exactly the same lines as a ground-truth group.
- **PA (parsing accuracy)**: share of lines whose template text is exactly
  right (static words kept, every variable replaced by a wildcard).
- **DeepParse**: same masks, plain DeepParse Drain (depth 5, similarity 0.4,
  no IoT `MEASURE` mask). The baseline.
- **logpipe**: our defaults (extra `MEASURE` mask, our bucket keys), no
  tuning. Empty unless `eval parsing --no-tune --out <EVAL_OUT>/parsing_untuned` ran.
- **logpipe+tuned**: Drain depth/similarity grid-searched per system on the
  same labels it is scored on, so read it as an upper bound.
- **holdout coverage**: no labels needed. Mine templates on the first 80% of
  lines, freeze, then the share of the last 20% that match a template.
  Lines that miss would go PENDING in production."""

ALIAS_CAPTION = """\
When enrichment mines a new template, should it merge into an existing one
(same event, different wording or generalisation) or stay separate (e.g.
`contact=open` vs `contact=closed`)? Scored on hand-labelled pairs.

- **rules_offline**: token similarity + slot rules, unsure cases → no merge.
- **rules_yes_llm**: same, but an LLM that says yes to every unsure case
  (worst-case LLM): how many bad merges the rules alone stop.
- **token_sim_only**: merge whenever token similarity is high, no rules.
- **precision**: of the merges made, share that were correct.
- **recall**: of the pairs that should merge, share that were merged.
- **false merges**: pairs of different events glued together (worse: hides data).
- **missed merges**: pairs of the same event left as two templates. Both are
  counts, lower is better."""

IOT_CAPTION = """\
The whole pipeline on a synthetic IoT fleet with injected anomalies and a
firmware rollout that adds new templates mid-test. Each device's stream is
cut into sliding windows; a window is anomalous if it contains an injected
anomaly. Each row turns one design choice off (ablation).

- **full**: everything on: unknown lines become an `[UNK]` token, novelty
  signal, fleet suppression, enrichment loop.
- **no_fleet_suppression**: templates the rollout brings to many devices
  still count as novel (expect more false alarms).
- **mask_unknowns**: unknown lines become `[MASK]` (original design), no novelty.
- **no_enrichment**: unknown lines stay PENDING forever.
- **P / R / F1**: precision (flags that were real), recall (anomalies
  caught), F1 (their harmonic mean), per window.
- **FP reasons**: which signal fired on false alarms: `sequence` (LogBERT
  could not predict masked templates), `hypersphere` (window embedding far
  from normal), `novelty` (too many never-seen templates).
- **PENDING at end**: lines still unresolved when the test ends (lower is better).
- **resolve lag p50**: median seconds from a line going PENDING to
  enrichment resolving it (lower is better).
- **adapter calls**: LLM calls made by enrichment (cost)."""

LOGBERT_CAPTION = """\
LogBERT anomaly detection on a public labelled dataset (HDFS: one session
per block id; BGL: fixed windows). The parser is calibrated on normal
training sessions only, so templates first seen at test time come out as
unknown, like PENDING lines in production.

- **unknowns**: how unknown templates reach the model: `unk` (own token),
  `mask` (hidden as `[MASK]`), `drop` (removed).
- **novelty**: whether a high share of unknown templates alone flags a window.
- **P / R / F1 (val-tuned)**: thresholds picked on a validation split, then
  scored on test. The honest number.
- **F1 (oracle)**: thresholds picked on the test split itself, as the
  original LogBERT paper does. Optimistic; only for comparing with papers."""


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
    return "## Parsing: Loghub-2k (DeepParse GA/PA)\n\n" + PARSING_CAPTION + "\n\n" + _table(
        ["system", "GA DeepParse", "GA logpipe", "GA logpipe+tuned",
         "PA DeepParse", "PA logpipe", "PA logpipe+tuned", "holdout coverage"], rows)


def _alias(root: Path) -> str:
    data = _load(root / "alias.json")
    if not data:
        return ""
    rows = [[name, r["precision"], r["recall"], r["FP"], r["FN"]]
            for name, r in data["deciders"].items()]
    return (f"## Alias decisions ({data['pairs']} labelled pairs)\n\n" + ALIAS_CAPTION + "\n\n"
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
        f"## LogBERT: {data['dataset']}\n\n{LOGBERT_CAPTION}\n\n"
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
    return ("## IoT fleet, end to end\n\n" + IOT_CAPTION + "\n\n" + f"Events: {data['events']}.\n\n"
            + _table(["variant", "P", "R", "F1", "FP reasons", "PENDING at end",
                      "resolve lag p50 (s)", "adapter calls"], rows) + extra)


def write_report(root: Path = Path("artifacts/eval")) -> Path:
    sections = [_parsing(root), _alias(root), _iot(root),
                _logbert(root, "hdfs"), _logbert(root, "bgl")]
    out = root / "REPORT.md"
    out.write_text("# Evaluation report\n\n" + INTRO + "\n\n" + "\n\n".join(s for s in sections if s) + "\n",
                   encoding="utf-8")
    return out
