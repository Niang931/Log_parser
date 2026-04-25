"""Coverage for deepparse.tools.build_training_set (Listing-2 builder)."""
from __future__ import annotations

import ast as _ast  # safe literal parser, not the dangerous builtin
import json
from pathlib import Path

from deepparse.tools.build_training_set import (
    INSTRUCTION,
    _align_template,
    _as_raw_literal,
    _classify_value,
    build_examples,
    write_jsonl,
)


def test_classify_value_recognises_canonical_shapes():
    assert _classify_value("2024-01-01 00:00:00") == r"\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?"
    assert "blk_" in _classify_value("blk_-1234567890")
    assert _classify_value("INFO") == r"\b(?:TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b"
    assert _classify_value("0xCAFE") == r"\b0x[0-9a-fA-F]+\b"
    assert _classify_value("550e8400-e29b-41d4-a716-446655440000").startswith(r"\b[0-9a-fA-F]{8}")


def test_align_template_recovers_variable_values():
    line = "PacketResponder 1 for block blk_38865049064139660 terminating"
    template = "PacketResponder <*> for block <*> terminating"
    values = _align_template(line, template)
    assert values == ["1", "blk_38865049064139660"]


def test_align_template_returns_none_on_mismatch():
    assert _align_template("foo bar baz", "completely <*> different") is None


def test_as_raw_literal_preserves_backslashes():
    out = _as_raw_literal(r"\d+")
    assert out == 'r"\\d+"'  # literal: r"\d+"


def test_as_raw_literal_escapes_double_quotes():
    out = _as_raw_literal(r'a"b')
    assert '"' in out
    # When the pattern contains ", we fall back to a regular string with
    # escaped backslashes.
    assert out.startswith('"') and out.endswith('"')


def test_build_examples_end_to_end(tmp_path: Path):
    dataset = tmp_path / "Tiny"
    dataset.mkdir()
    (dataset / "raw.log").write_text(
        "\n".join([
            "PacketResponder 1 for block blk_111 terminating",
            "PacketResponder 2 for block blk_222 terminating",
            "PacketResponder 3 for block blk_333 terminating",
        ]) + "\n",
        encoding="utf-8",
    )
    template = "PacketResponder <*> for block <*> terminating"
    (dataset / "templates.json").write_text(
        json.dumps({"entries": [{"cluster_id": 0, "template": template}] * 3}),
        encoding="utf-8",
    )

    examples = build_examples(dataset)
    assert len(examples) == 3
    for ex in examples:
        assert ex.instruction == INSTRUCTION
        assert "PacketResponder" in ex.input
        # Output must be a valid Python literal that round-trips through
        # ast.literal_eval (safe, parses literals only — not the builtin
        # eval()).
        regexes = _ast.literal_eval(ex.output)
        assert isinstance(regexes, list)
        assert all(isinstance(p, str) for p in regexes)
        assert len(regexes) >= 1


def test_build_examples_with_entropy_k_subsamples(tmp_path: Path):
    dataset = tmp_path / "Tiny"
    dataset.mkdir()
    lines = [f"event {i} value {i*7}" for i in range(20)]
    (dataset / "raw.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    template = "event <*> value <*>"
    (dataset / "templates.json").write_text(
        json.dumps({"entries": [{"cluster_id": 0, "template": template}] * 20}),
        encoding="utf-8",
    )

    sample = build_examples(dataset, entropy_k=5)
    assert len(sample) <= 5  # sampler may reject Jaccard-similar lines


def test_write_jsonl_round_trips(tmp_path: Path):
    from deepparse.tools.build_training_set import TrainingExample
    examples = [TrainingExample("instr", f"in {i}", "[]") for i in range(3)]
    out = tmp_path / "train.jsonl"
    n = write_jsonl(examples, out)
    assert n == 3
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["input"] for r in rows] == ["in 0", "in 1", "in 2"]
