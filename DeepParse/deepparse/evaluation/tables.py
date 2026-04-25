"""Generate LaTeX tables from CSV outputs."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from ..logging_utils import get_logger

LOGGER = get_logger(__name__)


def _read_csv(path: Path) -> List[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def _format_value(value: str) -> str:
    try:
        return f"{float(value):.3f}"
    except ValueError:
        return value


def _latex_table(rows: List[dict[str, str]], caption: str, label: str) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    header_line = " & ".join(headers)
    body_lines = [
        " & ".join(_format_value(row[h]) for h in headers) + r" \\"
        for row in rows
    ]
    body = "\n".join(body_lines)
    latex = (
        "\\begin{tabular}{" + "c" * len(headers) + "}\n"
        + header_line + r" \\" + "\n"
        + "\\hline\n"
        + body + "\n"
        + "\\end{tabular}"
    )
    return (
        f"% {caption}\n"
        "\\begin{table}[ht]\n"
        "\\centering\n"
        f"{latex}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )


def build_tables(csv_paths: Iterable[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in csv_paths:
        rows = _read_csv(csv_path)
        if not rows:
            continue
        headers = rows[0].keys()
        if "GA" in headers and "PA" in headers:
            latex = _latex_table(rows, "Table I: Grouping and Parsing Accuracy", "tab:ga_pa")
            if latex:
                (output_dir / "table_I.tex").write_text(latex, encoding="utf-8")
                with (output_dir / "table_I_ga_pa.csv").open("w", encoding="utf-8", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(headers))
                    writer.writeheader()
                    writer.writerows(rows)
                LOGGER.info("Generated Table I from %s", csv_path)
        if "seconds" in headers:
            latex = _latex_table(rows, "Table II: Timing Benchmark", "tab:timing")
            if latex:
                (output_dir / "table_II.tex").write_text(latex, encoding="utf-8")
                with (output_dir / "table_II_timing.csv").open("w", encoding="utf-8", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(headers))
                    writer.writeheader()
                    writer.writerows(rows)
                LOGGER.info("Generated Table II from %s", csv_path)
