import copy

from deepparse.drain.drain_engine import WILDCARD, DrainEngine
from deepparse.utils.regex_library import canonical_masks
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_token = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters=".-_=:"),
    min_size=1,
    max_size=10,
)
_line = st.lists(_token, min_size=1, max_size=8).map(" ".join)


def _state(engine: DrainEngine):
    return [(c.cluster_id, list(c.template), c.size) for c in engine.clusters()], engine._next_id


@given(train=st.lists(_line, min_size=1, max_size=30), probe=st.lists(_line, max_size=30))
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_match_never_mutates(train, probe):
    engine = DrainEngine(masks=canonical_masks())
    for line in train:
        engine.add_log(line)
    before = copy.deepcopy(_state(engine))
    for line in probe + train:
        engine.match(line)
    assert _state(engine) == before


@given(train=st.lists(_line, min_size=1, max_size=30))
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_trained_lines_match_their_final_cluster(train):
    engine = DrainEngine(masks=canonical_masks())
    for line in train:
        engine.add_log(line)
    for line in train:
        assert engine.match(line) is not None


def test_unknown_line_returns_none():
    engine = DrainEngine()
    engine.add_log("Door sensor contact=open")
    assert engine.match("Door sensor contact=closed") is None
    assert engine.match("Totally different event") is None


def test_seeded_template_keeps_caller_id_and_matches_through_prefix_wildcard():
    engine = DrainEngine()
    engine.add_template(42, ["Heater", WILDCARD, "switched"])
    assert engine.match("Heater relay switched").cluster_id == 42
    assert engine.add_log("brand new line").cluster_id == 43
