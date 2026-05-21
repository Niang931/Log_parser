"""
check_accuracy.py — DeepParse v2 Accuracy Diagnostic
======================================================
Run after any pipeline run to get a full quality report.

Usage
-----
# Check the latest run
python check_accuracy.py

# Check a specific run
python check_accuracy.py --run-id run_demo_fab_20260521T025919_42

# Check a specific file after parsing it
python check_accuracy.py --file artifacts/data/vendor4_thermaldyn.csv

# Show unparsed lines only
python check_accuracy.py --unparsed-only

# Full verbose output
python check_accuracy.py --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

# ── ANSI colours (Windows-safe fallback) ─────────────────────────────────────
try:
    import colorama; colorama.init()
    GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED    = "\033[91m"
    CYAN   = "\033[96m"; BOLD   = "\033[1m";  RESET  = "\033[0m"
    DIM    = "\033[2m"
except ImportError:
    GREEN = YELLOW = RED = CYAN = BOLD = RESET = DIM = ""

DB_PATH  = Path("artifacts/output/parsed_logs.db")
OUT_PATH = Path("artifacts/output")

# ─────────────────────────────────────────────────────────────────────────────

def load_run(run_id: str | None) -> tuple[str, list[dict]]:
    """Load records for run_id (or latest run if None)."""
    if not DB_PATH.exists():
        print(f"{RED}No database found at {DB_PATH}{RESET}")
        print("Run the pipeline first:  python main.py --input <file> ...")
        sys.exit(1)

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    if run_id is None:
        row = con.execute(
            "SELECT run_id FROM run_meta ORDER BY created_utc DESC LIMIT 1"
        ).fetchone()
        if not row:
            print(f"{RED}No runs found in database.{RESET}"); sys.exit(1)
        run_id = row["run_id"]

    records_raw = con.execute(
        "SELECT * FROM parsed_logs WHERE run_id = ? ORDER BY id", (run_id,)
    ).fetchall()
    meta = con.execute(
        "SELECT * FROM run_meta WHERE run_id = ?", (run_id,)
    ).fetchone()
    con.close()

    if not records_raw:
        print(f"{RED}Run '{run_id}' not found.{RESET}"); sys.exit(1)

    records = []
    for r in records_raw:
        row = dict(r)
        try:
            row["variables"] = json.loads(row["variables"])
        except Exception:
            row["variables"] = {}
        records.append(row)

    return run_id, records, dict(meta) if meta else {}


def colour_parse_rate(rate: float) -> str:
    pct = f"{rate*100:.1f}%"
    if rate >= 0.85:   return f"{GREEN}{BOLD}{pct}{RESET}"
    if rate >= 0.65:   return f"{YELLOW}{BOLD}{pct}{RESET}"
    return f"{RED}{BOLD}{pct}{RESET}"


def colour_wc(wc: float) -> str:
    if wc <= 0.4:  return f"{GREEN}{wc:.3f}{RESET}"
    if wc <= 0.7:  return f"{YELLOW}{wc:.3f}{RESET}"
    return f"{RED}{wc:.3f}{RESET}"


# ─────────────────────────────────────────────────────────────────────────────

def report(
    run_id: str,
    records: list[dict],
    meta: dict,
    unparsed_only: bool = False,
    verbose: bool       = False,
) -> None:

    total   = len(records)
    parsed  = [r for r in records if r["parsed"] == 1]
    failed  = [r for r in records if r["parsed"] == 0]
    rate    = len(parsed) / max(total, 1)

    # Wildcard ratio
    _MASK_RE = re.compile(r"<[A-Z_*][A-Z0-9_*]*>")
    def wc_ratio(tmpl: str) -> float:
        toks = tmpl.split()
        if not toks: return 0.0
        return sum(1 for t in toks if _MASK_RE.search(t)) / len(toks)

    avg_wc = sum(wc_ratio(r["template"]) for r in parsed) / max(len(parsed), 1)

    # Var counts
    avg_vars = sum(r["var_count"] for r in parsed) / max(len(parsed), 1)
    max_vars = max((r["var_count"] for r in parsed), default=0)

    # Template stats
    tmpl_counter = Counter(r["template"] for r in records)
    unique_tmpls = len(tmpl_counter)

    # ── Header ───────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*64}{RESET}")
    print(f"{BOLD}  DeepParse v2 — Accuracy Report{RESET}")
    print(f"{'='*64}")
    print(f"  {DIM}Run ID  :{RESET} {run_id[-45:]}")
    print(f"  {DIM}Provider:{RESET} {meta.get('llm_provider','?')}  "
          f"  {DIM}Masks:{RESET} {meta.get('mask_count','?')}  "
          f"  {DIM}Seed:{RESET} {meta.get('seed','?')}")
    print()

    # ── Core metrics ─────────────────────────────────────────────────────────
    print(f"  {BOLD}Parse Rate       {RESET}{colour_parse_rate(rate)}"
          f"  ({len(parsed)}/{total} lines)")

    wc_label = "excellent" if avg_wc<=0.3 else "good" if avg_wc<=0.5 else "fair" if avg_wc<=0.7 else "poor"
    print(f"  {BOLD}Wildcard Ratio   {RESET}{colour_wc(avg_wc)}"
          f"  ({wc_label} — lower = more specific templates)")

    print(f"  {BOLD}Unique Templates {RESET}{CYAN}{unique_tmpls}{RESET}"
          f"  (avg {total/max(unique_tmpls,1):.1f} lines per template)")

    print(f"  {BOLD}Avg Variables    {RESET}{CYAN}{avg_vars:.1f}{RESET}"
          f"  extracted per parsed line  (max: {max_vars})")

    # Grade
    grade = (
        f"{GREEN}A — Excellent{RESET}" if rate>=0.90 and avg_wc<=0.5 else
        f"{GREEN}B — Good{RESET}"      if rate>=0.80 and avg_wc<=0.65 else
        f"{YELLOW}C — Fair{RESET}"    if rate>=0.65 else
        f"{RED}D — Needs work{RESET}"
    )
    print(f"  {BOLD}Overall Grade    {RESET}{grade}")
    print()

    # ── What's working (top templates) ───────────────────────────────────────
    if not unparsed_only:
        print(f"{BOLD}  Top Templates (most frequent){RESET}")
        print(f"  {'─'*60}")
        for tmpl, count in tmpl_counter.most_common(8):
            wc = wc_ratio(tmpl)
            wc_col = colour_wc(wc)
            # Colour mask tokens
            coloured = re.sub(
                r'(<[A-Z_*][A-Z0-9_*]*>)',
                f"{CYAN}\\1{RESET}", tmpl
            )
            print(f"  {GREEN}[{count:3d}x]{RESET} wc={wc_col}  {coloured[:72]}")
        print()

    # ── Variable extraction sample ────────────────────────────────────────────
    if not unparsed_only and verbose and parsed:
        print(f"{BOLD}  Variable Extraction Samples (parsed lines){RESET}")
        print(f"  {'─'*60}")
        samples = [r for r in parsed if r["var_count"] >= 2][:5]
        for r in samples:
            print(f"  {DIM}raw :{RESET} {r['raw'][:80]}")
            print(f"  {DIM}tmpl:{RESET} {r['template'][:80]}")
            vars_str = "  ".join(f"{CYAN}{k}{RESET}={v}" for k,v in r["variables"].items())
            print(f"  {DIM}vars:{RESET} {vars_str}")
            print()

    # ── What's failing (unparsed lines) ──────────────────────────────────────
    if failed:
        print(f"{BOLD}  Unparsed Lines ({len(failed)}) — candidates for new masks{RESET}")
        print(f"  {'─'*60}")

        # Group by token pattern to spot common issues
        def tokenise(s: str) -> str:
            s = re.sub(r'\d+\.\d+[eE][+-]?\d+', '<SCI>', s)
            s = re.sub(r'\d+\.\d+', '<F>', s)
            s = re.sub(r'\b\d+\b', '<N>', s)
            return s

        pattern_groups: dict[str, list[str]] = {}
        for r in failed:
            key = tokenise(r["raw"])[:60]
            pattern_groups.setdefault(key, []).append(r["raw"])

        shown = 0
        for pattern, examples in sorted(pattern_groups.items(),
                                         key=lambda x: -len(x[1])):
            if shown >= 15: break
            count = len(examples)
            print(f"  {RED}[{count}x]{RESET} {examples[0][:80]}")
            if verbose and count > 1:
                for ex in examples[1:3]:
                    print(f"         {DIM}{ex[:80]}{RESET}")
            shown += 1
        print()

        # ── Suggested masks for failing lines ────────────────────────────────
        print(f"{BOLD}  Suggested Fixes{RESET}")
        print(f"  {'─'*60}")
        suggestions = _suggest_masks(failed)
        for s in suggestions:
            print(f"  {YELLOW}•{RESET} {s}")
        print()

    # ── Per-format breakdown (if multiple file stems) ─────────────────────────
    if verbose:
        print(f"{BOLD}  Accuracy by Line Type{RESET}")
        print(f"  {'─'*60}")
        categories = {
            "ISO timestamps":     re.compile(r'\d{4}-\d{2}-\d{2}T'),
            "Error codes":        re.compile(r'\bER-|DW-|RH-'),
            "Equipment IDs":      re.compile(r'\bEQP_'),
            "Numeric readings":   re.compile(r'\d+\.\d+[eE][+-]?\d+'),
            "XML attributes":     re.compile(r'\.\w+ ='),
            "INI key-values":     re.compile(r'^\w+\.\w+ = '),
            "Process jobs":       re.compile(r'\bPRJOB_|CJOB_'),
        }
        for cat, pat in categories.items():
            matched = [r for r in records if pat.search(r["raw"])]
            if not matched: continue
            cat_parsed = sum(r["parsed"] for r in matched)
            cat_rate   = cat_parsed / len(matched)
            bar = "█" * int(cat_rate * 20) + "░" * (20 - int(cat_rate * 20))
            col = GREEN if cat_rate>=0.85 else YELLOW if cat_rate>=0.65 else RED
            print(f"  {cat:<20} {col}{bar}{RESET} {cat_rate*100:.0f}%  ({len(matched)} lines)")
        print()

    # ── Summary verdict ───────────────────────────────────────────────────────
    print(f"{'='*64}")
    if rate >= 0.85:
        print(f"  {GREEN}✓ Parse rate meets production threshold (≥85%){RESET}")
    else:
        gap = 0.85 - rate
        needed = int(gap * total)
        print(f"  {RED}✗ Need {needed} more lines parsed to reach 85% threshold{RESET}")
        print(f"    → Add Anthropic/Groq key for LLM mask synthesis")
        print(f"    → Run with --adaptive-threshold 0.85 --adaptive-rounds 3")
        print(f"    → Check --message-col if reading wrong CSV column")
    print(f"{'='*64}\n")


def _suggest_masks(failed: list[dict]) -> list[str]:
    """Heuristic mask suggestions based on unparsed line patterns."""
    suggestions = []
    raw_lines = [r["raw"] for r in failed]
    combined  = "\n".join(raw_lines)

    checks = [
        (r"\d{4}-\d{2}-\d{2}",        "Date tokens present — add --static-masks or check ISO_TS mask"),
        (r"[A-Z]{2,}_[A-Z]{2,}_\d+",  "Fab ID tokens (EQP/LOT/RCP style) not being masked — check masks_fab_universal.json"),
        (r"\bVER_",                     "Version tokens not masked — VER_ pattern may be missing"),
        (r"\bER-\d+",                   "Error codes present — ER- pattern may not be matching"),
        (r"^\[",                        "INI section headers [Section] — these are structure, not variables; acceptable"),
        (r"[A-Za-z]+ = [A-Za-z ]+$",   "Free-text values (e.g. Vendor = ThermalDyn) — add TEXT_VALUE mask"),
        (r"\d+\.\d+[eE][+-]?\d+",      "Scientific notation not masked — SCI_NUM mask may have wrong regex"),
        (r'"[^"]{4,}"',                 "Quoted string values — add mask for quoted text"),
        (r"[A-Za-z]+\.[A-Za-z]+ = ",   "Section.key = value lines — INI_SECTION_KV mask may need adjustment"),
        (r"(?i)(complete|running|idle|error|pass|fail)\b",
                                        "Status tokens — add STATUS_TOKEN mask to masks_fab_universal.json"),
    ]

    for pattern, msg in checks:
        if re.search(pattern, combined, re.MULTILINE):
            suggestions.append(msg)

    if not suggestions:
        suggestions.append("Patterns are novel — run with --llm-provider anthropic or groq for LLM synthesis")
    return suggestions[:6]


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="DeepParse v2 — Accuracy Diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run-id",       default=None, help="Run ID to check (default: latest)")
    ap.add_argument("--unparsed-only",action="store_true", help="Show only unparsed lines")
    ap.add_argument("--verbose",      action="store_true", help="Full detail including variable samples")
    args = ap.parse_args()

    run_id, records, meta = load_run(args.run_id)
    report(
        run_id,
        records,
        meta,
        unparsed_only=args.unparsed_only,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
