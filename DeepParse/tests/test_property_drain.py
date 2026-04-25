"""Property-based tests for the Drain engine + MaskApplier.

Hypothesis sweeps over many randomly-generated log corpora to ensure
the parser's structural invariants hold for *any* input that the
LogHub-2k pipeline might produce, not just hand-picked examples.
"""
from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from deepparse.drain.drain_engine import DrainEngine, WILDCARD
from deepparse.drain.masks_application import MaskApplier
from deepparse.masks_types import Mask
from deepparse.utils.regex_library import canonical_masks

# Tokens that look like real log tokens (alnum + a few separators).
_token = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters=".-_/:",
    ),
    min_size=1,
    max_size=12,
)
_log_line = st.lists(_token, min_size=1, max_size=10).map(" ".join)


@given(logs=st.lists(_log_line, min_size=1, max_size=50))
@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
def test_parse_returns_one_template_per_log(logs):
    engine = DrainEngine(masks=canonical_masks())
    out = engine.parse(logs)
    assert len(out) == len(logs)


@given(logs=st.lists(_log_line, min_size=1, max_size=50))
@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
def test_identical_lines_get_identical_template_ids(logs):
    """Paper claim, Section 'Integration with Drain'."""
    # Sprinkle each line in twice in a different position.
    interleaved = list(logs) + list(reversed(logs))
    engine = DrainEngine(masks=canonical_masks())
    pairs = engine.parse_with_ids(interleaved)
    by_line: dict[str, set[int]] = {}
    for line, (cid, _tmpl) in zip(interleaved, pairs):
        by_line.setdefault(line, set()).add(cid)
    # Every distinct line should resolve to exactly one cluster id.
    for line, ids in by_line.items():
        assert len(ids) == 1, f"line {line!r} produced cluster ids {ids}"


@given(logs=st.lists(_log_line, min_size=1, max_size=50))
@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
def test_parsing_is_pure(logs):
    """Two engine instances on identical input -> identical output."""
    a = DrainEngine(masks=canonical_masks()).parse(logs)
    b = DrainEngine(masks=canonical_masks()).parse(logs)
    assert a == b


@given(logs=st.lists(_log_line, min_size=1, max_size=30))
@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
def test_templates_have_no_typed_placeholders_after_normalisation(logs):
    """Final templates must use <*> (LogHub canonical), never <VAR:...>."""
    engine = DrainEngine(masks=canonical_masks())
    for template in engine.parse(logs):
        assert "<VAR:" not in template


@given(
    logs=st.lists(_log_line, min_size=1, max_size=20),
    masks=st.lists(
        st.builds(
            Mask,
            label=st.text("ABCDEFGH", min_size=1, max_size=8),
            pattern=st.sampled_from([r"\d+", r"[a-z]+", r"\S+"]),
            justification=st.just(""),
        ),
        min_size=0, max_size=4,
    ),
)
@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
def test_mask_applier_never_raises_on_valid_inputs(logs, masks):
    applier = MaskApplier(masks)
    for line in logs:
        out = applier.apply(line)
        assert isinstance(out, str)


def test_wildcard_constant_is_canonical_loghub_token():
    # Defensive: this constant ties the implementation to LogHub semantics.
    assert WILDCARD == "<*>"
