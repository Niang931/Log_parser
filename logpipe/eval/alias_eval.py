"""Alias-decision evaluation on hand-labelled template pairs.

Each pair is (new template, existing template, same event?).  The pairs
cover the cases the review called out: the same event with a different
generalisation or wording (should merge), and opposite states or outcomes
that a 0.92-cosine embedding rule would merge (must not merge).

Three deciders are compared:

* ``rules_offline``: token similarity + slot-only rule; borderline → no.
* ``rules_yes_llm``: the same, but an LLM that confirms *every* borderline
  case.  This is the worst-case LLM, and it shows the opposite-state guard
  holding on its own.
* ``token_sim_only``: alias whenever token similarity ≥ threshold, with no
  rules.  This is a proxy for a pure similarity threshold.
"""
from __future__ import annotations

from types import MappingProxyType

from ..enrich.llm import OfflineAdapter
from ..enrich.similarity import find_alias, template_similarity
from ..registry import RegistrySnapshot, TemplateEntry

PAIRS: list[tuple[str, str, bool]] = [
    # Same event, different generalisation (slot vs literal).
    ("Lock <*> state=locked battery=<VAR:MEASURE> user=<*>", "Lock <*> state=locked battery=<VAR:MEASURE> user=auto", True),
    ("Thermostat SN:<*> temp=<VAR:MEASURE> mode=<*>", "Thermostat SN:TH-<VAR:NUMBER> temp=<VAR:MEASURE> mode=auto", True),
    ("Relay toggled state=on source=<*>", "Relay toggled state=on source=app", True),
    ("Connected to broker <VAR:IPV4> after <*> retries", "Connected to broker <VAR:IPV4> after 3 retries", True),
    ("OTA check version=<*> available=false", "OTA check version=1.4.2 available=false", True),
    ("Gateway <*> connected clients=<VAR:MEASURE> uptime=<VAR:MEASURE>", "Gateway GW-1000 connected clients=<VAR:MEASURE> uptime=<VAR:MEASURE>", True),
    # Same event, different wording (needs an LLM to confirm).
    ("Connection to broker was established", "Connection to broker established", True),
    ("Firmware update has completed version=<VAR:MEASURE>", "Firmware update completed version=<VAR:MEASURE>", True),
    ("Sensor reading temp=<VAR:MEASURE> unit=C", "Sensor reading temp=<VAR:MEASURE>", True),
    ("Thermostat SN:<*> temp=<VAR:MEASURE> mode=<*> fw=<VAR:MEASURE>", "Thermostat SN:<*> temp=<VAR:MEASURE> mode=<*>", True),
    # Opposite states / outcomes: must never merge.
    ("Door sensor contact=open", "Door sensor contact=closed", False),
    ("Lock <*> state=locked battery=<VAR:MEASURE>", "Lock <*> state=unlocked battery=<VAR:MEASURE>", False),
    ("Heater relay switched state=on duty=<VAR:MEASURE>", "Heater relay switched state=off duty=<VAR:MEASURE>", False),
    ("Keypad code accepted user=<*> slot=<VAR:MEASURE>", "Keypad code rejected user=<*> slot=<VAR:MEASURE>", False),
    ("Connection to broker established", "Connection to broker failed", False),
    ("Device <*> connected to gateway", "Device <*> disconnected from gateway", False),
    ("Firmware update succeeded version=<VAR:MEASURE>", "Firmware update failed version=<VAR:MEASURE>", False),
    ("Alarm armed mode=<*>", "Alarm disarmed mode=<*>", False),
    ("Motion detected zone=<VAR:MEASURE>", "Motion cleared zone=<VAR:MEASURE>", False),
    ("Valve opened flow=<VAR:MEASURE>", "Valve closed flow=<VAR:MEASURE>", False),
    ("TLS handshake ok peer=<*>", "TLS handshake error peer=<*>", False),
    ("Battery level high value=<VAR:MEASURE>", "Battery level low value=<VAR:MEASURE>", False),
    ("Access granted user=<*>", "Access denied user=<*>", False),
    ("Device <*> online", "Device <*> offline", False),
    ("Schedule enabled profile=<*>", "Schedule disabled profile=<*>", False),
    # Different events that merely share structure.
    ("MQTT publish topic=<*> qos=<VAR:MEASURE> bytes=<VAR:MEASURE>", "MQTT subscribe topic=<*> qos=<VAR:MEASURE> bytes=<VAR:MEASURE>", False),
    ("Watchdog reset reason=<*> uptime=<VAR:MEASURE>", "Scheduled reset reason=<*> uptime=<VAR:MEASURE>", False),
    ("Sensor fault code=<*> retrying in <VAR:MEASURE>", "Sensor calibration code=<*> retrying in <VAR:MEASURE>", False),
]


class _YesAdapter(OfflineAdapter):
    def confirm_alias(self, new_tokens, old_tokens, new_samples) -> bool:
        return True


def _decide(decider: str, new: tuple[str, ...], old: tuple[str, ...], threshold: float) -> bool:
    snapshot = RegistrySnapshot(templates=MappingProxyType({2: TemplateEntry(2, old, ())}))
    if decider == "token_sim_only":
        return template_similarity(new, old) >= threshold
    adapter = _YesAdapter() if decider == "rules_yes_llm" else OfflineAdapter()
    return find_alias(new, snapshot, adapter, [], threshold=threshold).alias_of == 2


def run_alias_eval(threshold: float = 0.75) -> dict:
    results = {}
    for decider in ("rules_offline", "rules_yes_llm", "token_sim_only"):
        tp = fp = tn = fn = 0
        wrong = []
        for new, old, same in PAIRS:
            merged = _decide(decider, tuple(new.split()), tuple(old.split()), threshold)
            tp += merged and same
            fp += merged and not same
            fn += (not merged) and same
            tn += (not merged) and not same
            if merged != same:
                wrong.append({"new": new, "old": old, "same_event": same, "merged": merged})
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        results[decider] = {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
                            "precision": round(precision, 4), "recall": round(recall, 4),
                            "wrong": wrong}
    return {"threshold": threshold, "pairs": len(PAIRS), "deciders": results}
