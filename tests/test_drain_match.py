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
    return [(c.cluster_id, list(c.template), c.size) for c in engine.clusters], engine._next_id


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
    match = engine.match("Heater relay switched")
    assert match is not None and match.cluster_id == 42
    assert engine.add_log("brand new line").cluster_id == 43


def test_routing_keys_tokens_containing_digits_together():
    legacy = DrainEngine(depth=3, legacy_routing=True)
    drain3 = DrainEngine(depth=3)
    lines = [f"Receiving block blk_{i}{i} from node" for i in range(1, 6)]
    for line in lines:
        legacy.add_log(line)
        drain3.add_log(line)
    assert legacy.num_clusters == 5  # old DeepParse key: one bucket per block id
    assert drain3.num_clusters == 1
    assert drain3.clusters[0].template[2] == "blk_<*>"


def test_routing_wildcards_composite_placeholders():
    from deepparse.masks_types import Mask

    masks = [Mask(label="NUMBER", pattern=r"(?<=:)\d+", justification="")]
    engine = DrainEngine(masks=masks)
    for host in ("a.example.com", "b.example.org", "c.example.net"):
        engine.add_log(f"{host}:443 open through proxy")
    assert engine.num_clusters == 1  # host:<VAR:NUMBER> no longer splits per host
    legacy = DrainEngine(masks=masks, legacy_routing=True)
    for host in ("a.example.com", "b.example.org", "c.example.net"):
        legacy.add_log(f"{host}:443 open through proxy")
    assert legacy.num_clusters == 3


def test_match_fallback_only_visits_wildcard_buckets():
    engine = DrainEngine(depth=2)
    for i in range(500):
        # digit-free: stays literal
        engine.add_log(f"literal{'x' * (i % 50)}{'y' * (i // 50)} event happened")
    engine.add_template(9_999, ["Heater", WILDCARD, "switched"])
    assert sum(len(b) for b in engine._wild.values()) == 1
    match = engine.match("Heater relay switched")
    assert match is not None and match.cluster_id == 9_999
    assert engine.match("unknown thing here") is None


def test_merge_keeps_shared_static_text():
    from deepparse.drain.drain_engine import _merge_token

    assert _merge_token("uid=0", "uid=5") == "uid=<*>"
    assert _merge_token("ntp1,", "10.0.0.1,") == "<*>,"
    assert _merge_token("(uid=0)", "(uid=42)") == "(uid=<*>)"
    assert _merge_token("succeeded", "failed") == WILDCARD
    assert _merge_token("<VAR:IP>,", "<VAR:HEX>,") == "<*>,"
    assert (
        _merge_token("a.example.com:<VAR:NUMBER>", "b.example.org:<VAR:NUMBER>")
        == "<*>:<VAR:NUMBER>"
    )
    assert _merge_token("<*>,", "x,") == "<*>,"
    assert _merge_token("<VAR:PATH>", "<VAR:PATH>?a=1") == WILDCARD


def test_composite_template_token_matches_later_lines():
    engine = DrainEngine(depth=2)
    for i, host in enumerate(("alpha,", "beta,", "gamma,")):
        engine.add_log(f"synchronized to {host} stratum {i}")
    assert engine.num_clusters == 1
    assert engine.clusters[0].template_str() == "synchronized to <*>, stratum <*>"
    assert engine.match("synchronized to delta, stratum 3") is not None


def test_frozen_composite_template_matches_new_values():
    engine = DrainEngine()
    engine.add_template(7, ["Receiving", "block", "blk_<*>", "src:", "<*>"])
    match = engine.match("Receiving block blk_-9876 src: /10.0.0.1")
    assert match is not None and match.cluster_id == 7


def test_composite_wildcard_may_be_empty():
    engine = DrainEngine(depth=2)
    engine.add_log("session opened by LOGIN(uid=0)")
    engine.add_log("session opened by (uid=0)")
    assert engine.num_clusters == 1
    assert engine.match("session opened by (uid=0)") is not None
