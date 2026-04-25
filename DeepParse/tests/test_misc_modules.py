"""Coverage for small utility modules (tables / yaml / tokenize / logging)."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from deepparse.evaluation.tables import build_tables
from deepparse.logging_utils import configure_logging, get_logger
from deepparse.tokenize import mask_tokens, tokenize
from deepparse.utils.deterministic import timed_block
from deepparse.utils.yaml_loader import load_yaml, loads_yaml


# --- tokenize ----------------------------------------------------------------
def test_tokenize_strips_and_splits():
    assert tokenize("  a  b\tc\n") == ["a", "b", "c"]


def test_tokenize_returns_empty_for_blank():
    assert tokenize("   ") == []


def test_mask_tokens_substitutes_known_classes():
    out = mask_tokens(["INFO", "192.168.1.1", "hello"])
    assert out == ["<LOGLEVEL>", "<IPV4>", "hello"]


# --- yaml_loader -------------------------------------------------------------
def test_loads_yaml_scalars_and_lists():
    text = "\n".join([
        "k_int: 42",
        "k_float: 3.14",
        "k_true: true",
        "k_str: hello",
        "items:",
        "  - one",
        "  - two",
    ])
    out = loads_yaml(text)
    assert out["k_int"] == 42
    assert out["k_float"] == 3.14
    assert out["k_true"] is True
    assert out["k_str"] == "hello"
    assert out["items"] == ["one", "two"]


def test_load_yaml_from_file(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text("foo: bar\nbaz: 1\n", encoding="utf-8")
    assert load_yaml(path) == {"foo": "bar", "baz": 1}


# --- tables ------------------------------------------------------------------
def test_build_tables_writes_table_i_and_ii(tmp_path):
    table_i = tmp_path / "metrics.csv"
    with table_i.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", "method", "GA", "PA"])
        writer.writeheader()
        writer.writerow({"dataset": "Apache", "method": "DeepParse", "GA": 1.0, "PA": 1.0})

    table_ii = tmp_path / "timing.csv"
    with table_ii.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", "seconds", "n_logs"])
        writer.writeheader()
        writer.writerow({"dataset": "Apache", "seconds": 0.1, "n_logs": 100})

    out_dir = tmp_path / "out"
    build_tables([table_i, table_ii], out_dir)

    assert (out_dir / "table_I.tex").exists()
    assert (out_dir / "table_I_ga_pa.csv").exists()
    assert (out_dir / "table_II.tex").exists()
    assert (out_dir / "table_II_timing.csv").exists()
    assert "DeepParse" in (out_dir / "table_I.tex").read_text(encoding="utf-8")


# --- logging -----------------------------------------------------------------
def test_configure_logging_creates_log_file(tmp_path, monkeypatch):
    log_path = configure_logging(str(tmp_path), "deepparse-test")
    assert Path(log_path).exists()
    assert isinstance(get_logger("deepparse.tests"), logging.Logger)


# --- deterministic helpers ---------------------------------------------------
def test_timed_block_yields_label_and_start(capsys):
    with timed_block("hello") as (label, start):
        assert label == "hello"
        assert isinstance(start, float)
    captured = capsys.readouterr()
    assert "hello" in captured.out
