import pytest

from logpipe.registry import FIRST_TEMPLATE_ID, NewTemplate, SchemaRegistry


def _t(*tokens, keys=None, alias_of=None):
    return NewTemplate(tuple(tokens), tuple(keys or ()), alias_of)


def test_ids_are_monotonic_never_reused_and_versions_increase():
    registry = SchemaRegistry()
    snap1, ids1 = registry.publish([_t("a"), _t("b")])
    snap2, ids2 = registry.publish([_t("c")])
    assert ids1 == [FIRST_TEMPLATE_ID, FIRST_TEMPLATE_ID + 1]
    assert ids2 == [FIRST_TEMPLATE_ID + 2]
    assert (snap1.version, snap2.version) == (1, 2)
    assert set(snap1.templates) < set(snap2.templates)


def test_snapshots_are_immutable():
    snap, _ = SchemaRegistry().publish([_t("a")])
    with pytest.raises(TypeError):
        snap.templates[99] = None


def test_round_trip_preserves_ids_and_next_id(tmp_path):
    registry = SchemaRegistry()
    registry.publish([_t("temp=<VAR:MEASURE>", keys=["temp"]), _t("x")])
    registry.save(tmp_path / "reg.json")
    loaded = SchemaRegistry.load(tmp_path / "reg.json")
    assert loaded.current == registry.current
    _, ids = loaded.publish([_t("y")])
    assert ids == [registry.current.next_id]


def test_alias_resolves_to_canonical_id():
    registry = SchemaRegistry()
    _, (base,) = registry.publish([_t("Connection", "opened")])
    snap, (alias,) = registry.publish([_t("Connection", "was", "opened", alias_of=base)])
    assert snap.canonical_id(alias) == base
    assert snap.canonical_id(base) == base


def test_publish_validates_alias_target_and_key_count():
    registry = SchemaRegistry()
    with pytest.raises(ValueError, match="alias target"):
        registry.publish([_t("a", alias_of=12345)])
    with pytest.raises(ValueError, match="slots"):
        registry.publish([_t("temp=<*>", keys=[])])
    assert registry.current.version == 0
