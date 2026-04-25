"""Build instruction-tuning data for the DeepParse mask synthesiser.

Training format follows Listing 2 of the paper exactly: every example is
``{"instruction": ..., "input": <log_line>, "output": <Python regex list>}``
where ``<output>`` is a string containing a literal Python list of raw
regex strings, one per variable slot detected in the line.

How regex strings are derived:

1. Read each system's ``templates.json`` (produced by
   :mod:`deepparse.tools.fetch_loghub`).
2. For every line, walk the EventTemplate left-to-right.  Every ``<*>``
   slot is paired with its corresponding substring in the raw line.
3. The substring is *generalised* to a canonical regex chosen by token
   shape (digits → ``\\d+``, IPv4 → standard regex, hex → ``0x[0-9a-fA-F]+``,
   timestamp/uuid/path → standard regexes, alphanumerics → ``[\\w.-]+``).
4. Resulting regex list is deduplicated *preserving order* — exactly
   matches the post-processing the paper describes.

Output: a single ``train.jsonl`` file (or one per system if
``--per-system`` is passed) under ``--out``.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

INSTRUCTION = (
    "Generate a Python list of regex patterns that capture the dynamic "
    "(variable) parts in the input log message while preserving the static "
    "structure."
)


# Regex shape detectors → canonical regex emitted into the training output.
# Order matters: more specific shapes first.
_SHAPE_DETECTORS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?$"),
     r"\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?"),
    (re.compile(r"^\d{6}$"),  # LogHub date 'YYMMDD' (e.g. 081109)
     r"\d{6}"),
    (re.compile(r"^\d{6}\s+\d{6}$"),
     r"\d{6}\s+\d{6}"),
    (re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
     r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    (re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?$"),
     r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"),
    (re.compile(r"^0x[0-9a-fA-F]+$"),
     r"\b0x[0-9a-fA-F]+\b"),
    (re.compile(r"^/[A-Za-z0-9_./\-]+$"),
     r"(?<!\S)/[A-Za-z0-9_./\-]+"),
    (re.compile(r"^(?:TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)$"),
     r"\b(?:TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b"),
    (re.compile(r"^-?\d+(?:\.\d+)?$"),
     r"-?\d+(?:\.\d+)?"),
    (re.compile(r"^blk_-?\d+$"),  # HDFS block IDs
     r"blk_-?\d+"),
    # Compound identifier with letters + digits + underscores.
    (re.compile(r"^[A-Za-z_][\w.\-:/]*\d[\w.\-:/]*$"),
     r"[\w.\-:/]+"),
]
_FALLBACK_REGEX = r"[^\s]+"


def _as_raw_literal(pattern: str) -> str:
    """Render ``pattern`` as a Python raw-string literal that, when parsed,
    evaluates back to the same characters as ``pattern``.

    A raw string ``r"..."`` cannot contain an unescaped ``"``; if the
    pattern does, fall back to a regular string with escaped backslashes.
    """
    if '"' not in pattern and not pattern.endswith("\\"):
        return f'r"{pattern}"'
    return '"' + pattern.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _classify_value(value: str) -> str:
    for detector, pattern in _SHAPE_DETECTORS:
        if detector.match(value):
            return pattern
    return _FALLBACK_REGEX


@dataclass
class TrainingExample:
    instruction: str
    input: str
    output: str

    def to_dict(self) -> dict:
        return {"instruction": self.instruction, "input": self.input, "output": self.output}


def _align_template(line: str, template: str) -> List[str] | None:
    """Return the list of variable-slot values in ``line`` matching ``template``.

    ``template`` is a LogHub-style string with ``<*>`` placeholders.
    Returns ``None`` if the regex built from the template fails to match
    (which happens occasionally for hand-corrected templates with
    whitespace ambiguity).
    """
    # Build a regex by escaping literal segments and replacing each <*>
    # with a non-greedy capture group that matches at least one char.
    parts = template.split("<*>")
    pattern = "(.+?)".join(re.escape(p) for p in parts)
    pattern = "^" + pattern + "$"
    match = re.match(pattern, line)
    if not match:
        # Try a more permissive match (leading/trailing whitespace).
        match = re.match(pattern.replace("$", r"\s*$").replace("^", r"^\s*"), line)
        if not match:
            return None
    return list(match.groups())


def build_examples(
    dataset_dir: Path,
    max_per_system: int | None = None,
    entropy_k: int | None = None,
) -> List[TrainingExample]:
    raw_path = dataset_dir / "raw.log"
    tpl_path = dataset_dir / "templates.json"
    if not raw_path.exists() or not tpl_path.exists():
        raise FileNotFoundError(f"{dataset_dir} missing raw.log or templates.json")

    lines = [line.rstrip("\n") for line in raw_path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    payload = json.loads(tpl_path.read_text(encoding="utf-8"))
    entries = payload["entries"] if isinstance(payload, dict) else payload
    if len(entries) != len(lines):
        raise ValueError(
            f"Length mismatch in {dataset_dir}: {len(lines)} logs vs {len(entries)} templates"
        )

    if entropy_k is not None and entropy_k < len(lines):
        # Paper, Section 'Implementation Details': training samples are
        # selected via the entropy-greedy algorithm.
        from ..utils.sampling import entropy_greedy_sample
        keep = sorted(set(entropy_greedy_sample(lines, entropy_k)))
        lines = [lines[i] for i in keep]
        entries = [entries[i] for i in keep]

    examples: List[TrainingExample] = []
    for line, entry in zip(lines, entries):
        template = entry["template"]
        values = _align_template(line, template)
        if values is None:
            continue  # skip lines we can't align cleanly
        regexes: List[str] = []
        seen: set[str] = set()
        for value in values:
            regex = _classify_value(value.strip())
            if regex not in seen:
                regexes.append(regex)
                seen.add(regex)
        # Format the output as a literal Python list of *raw strings*.  We
        # cannot use json.dumps here because that escapes for JSON
        # encoding and would double-escape the backslashes that make up
        # the regex (e.g. \\d+ instead of \d+).  Instead we emit each
        # pattern verbatim inside r"..." and only escape embedded double
        # quotes.
        output = "[\n    " + ",\n    ".join(_as_raw_literal(r) for r in regexes) + ",\n]"
        examples.append(TrainingExample(INSTRUCTION, line, output))
        if max_per_system is not None and len(examples) >= max_per_system:
            break
    return examples


def write_jsonl(examples: Iterable[TrainingExample], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict()) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/training/train.jsonl"))
    parser.add_argument("--systems", nargs="*", default=None,
                        help="Subset of system names; default = every directory under data-dir")
    parser.add_argument("--max-per-system", type=int, default=None)
    parser.add_argument("--entropy-k", type=int, default=None,
                        help="If set, select this many examples per system via entropy-greedy "
                             "sampling (paper Algorithm 1). Use 50 to match the paper.")
    parser.add_argument("--per-system", action="store_true",
                        help="Write one JSONL per system instead of a combined file")
    args = parser.parse_args(argv)

    if args.systems is None:
        systems = [p.name for p in sorted(args.data_dir.iterdir()) if (p / "raw.log").exists()]
    else:
        systems = args.systems

    if not systems:
        print("[build_training_set] no systems found", file=__import__("sys").stderr)
        return 1

    if args.per_system:
        for system in systems:
            examples = build_examples(args.data_dir / system, args.max_per_system, args.entropy_k)
            out = args.out.parent / f"{system}.jsonl"
            n = write_jsonl(examples, out)
            print(f"[build_training_set] {system}: {n} examples -> {out}")
    else:
        all_examples: List[TrainingExample] = []
        for system in systems:
            examples = build_examples(args.data_dir / system, args.max_per_system, args.entropy_k)
            print(f"[build_training_set] {system}: {len(examples)} examples")
            all_examples.extend(examples)
        n = write_jsonl(all_examples, args.out)
        print(f"[build_training_set] total: {n} examples -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
