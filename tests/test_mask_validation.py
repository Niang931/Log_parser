from deepparse.masks_types import Mask

from logpipe.enrich.validate import (
    clean_keys,
    generalize_digits,
    safe_masks,
    template_problem,
)


def test_template_problem_rejects_over_general_and_misfitting_templates():
    assert "over-general" == template_problem(("<*>", "<*>", "<*>"), ["a b c"])
    assert "misses" == template_problem(("Door", "contact=<*>"), ["Window contact=open"])
    assert template_problem(("Door", "contact=<*>"), ["Door contact=open"]) is None


def test_clean_keys_falls_back_on_bad_llm_output():
    tokens = ("temp=<VAR:MEASURE>", "mode=<*>")
    assert clean_keys(["temperature", "mode"], tokens) == ["temperature", "mode"]
    assert clean_keys(["temperature"], tokens) == ["temp", "mode"]  # wrong count
    assert clean_keys(["Temp!", "mode"], tokens) == ["temp", "mode"]  # malformed
    assert clean_keys(["x", "x"], tokens) == ["temp", "mode"]  # duplicates
    assert clean_keys(None, tokens) == ["temp", "mode"]


def test_safe_masks_drops_invalid_and_backtracking_patterns():
    masks = [Mask("OK", r"\d+", ""), Mask("BAD", r"(", ""), Mask("NESTED", r"(a+)+b", "")]
    assert [m.label for m in safe_masks(masks)] == ["OK"]


def test_generalize_digits_only_touches_literal_tokens():
    assert generalize_digits(("fault", "code=E165", "in", "6548ms", "x=<VAR:MEASURE>")) == (
        "fault", "<*>", "in", "<*>", "x=<VAR:MEASURE>",
    )

