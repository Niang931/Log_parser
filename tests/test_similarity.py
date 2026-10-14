from types import MappingProxyType

from logpipe.enrich.llm import OfflineAdapter
from logpipe.enrich.similarity import (
    classify_difference,
    find_alias,
    opposite_states,
    template_similarity,
    tokens_compatible,
)
from logpipe.registry import RegistrySnapshot, TemplateEntry


class YesAdapter(OfflineAdapter):
    def __init__(self):
        self.calls = 0

    def confirm_alias(self, new_tokens, old_tokens, new_samples):
        self.calls += 1
        return True


def _snapshot(*templates: str) -> RegistrySnapshot:
    entries = {i + 2: TemplateEntry(i + 2, tuple(t.split()), ()) for i, t in enumerate(templates)}
    return RegistrySnapshot(templates=MappingProxyType(entries))


def t(text: str) -> tuple[str, ...]:
    return tuple(text.split())


def test_tokens_compatible_handles_composite_slots():
    assert tokens_compatible("user=<*>", "user=alice")
    assert tokens_compatible("<VAR:MEASURE>", "<*>")
    assert not tokens_compatible("user=<*>", "slot=2")
    assert not tokens_compatible("open", "closed")


def test_similarity_is_slot_aware_and_handles_length_changes():
    assert template_similarity(t("Relay state=<*>"), t("Relay state=on")) == 1.0
    assert 0.5 < template_similarity(t("Connection to broker was established"),
                                     t("Connection to broker established")) < 1.0


def test_opposite_states_cover_pairs_and_negating_prefixes():
    assert opposite_states(t("Door contact=open"), t("Door contact=closed"))
    assert opposite_states(t("Lock state=locked"), t("Lock state=unlocked"))
    assert opposite_states(t("Device connected"), t("Device disconnected"))
    assert opposite_states(t("Broker connection established"), t("Broker connection failed"))
    assert opposite_states(t("Upload ok"), t("Upload timeout"))
    assert not opposite_states(t("Connection was established"), t("Connection established"))


def test_classify_difference():
    assert classify_difference(t("a <*> c"), t("a b c")) == "slot_only"
    assert classify_difference(t("a was b"), t("a b")) == "wording"
    assert classify_difference(t("MQTT publish x"), t("MQTT subscribe x")) == "literal"
    assert classify_difference(t("Door open"), t("Door closed")) == "conflict"


def test_slot_only_difference_aliases_without_asking_the_llm():
    adapter = YesAdapter()
    snap = _snapshot("Lock <*> state=locked user=auto")
    decision = find_alias(t("Lock <*> state=locked user=<*>"), snap, adapter, [])
    assert decision.alias_of == 2 and decision.kind == "slot_only"
    assert adapter.calls == 0


def test_opposite_states_never_merge_even_if_the_llm_says_yes():
    adapter = YesAdapter()
    snap = _snapshot("Door sensor contact=closed")
    assert find_alias(t("Door sensor contact=open"), snap, adapter, []).alias_of is None
    assert adapter.calls == 0


def test_wording_changes_need_llm_confirmation():
    snap = _snapshot("Connection to broker established")
    new = t("Connection to broker was established")
    assert find_alias(new, snap, OfflineAdapter(), []).alias_of is None
    assert find_alias(new, snap, YesAdapter(), []).alias_of == 2


def test_aliases_point_at_canonical_templates_only():
    entries = {
        2: TemplateEntry(2, t("Relay state=on"), ()),
        3: TemplateEntry(3, t("Relay state=<*>"), (), alias_of=2),
    }
    snap = RegistrySnapshot(templates=MappingProxyType(entries))
    assert find_alias(t("Relay state=on"), snap, OfflineAdapter(), []).alias_of == 2
