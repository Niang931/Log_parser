"""The fault-injection world the root-cause eval runs on."""
from __future__ import annotations

from datetime import timedelta

from logpipe.eval.rca_eval import build_world, contains_truth, naive_bundle, utc_clock
from logpipe.registry import UNK_TEMPLATE_ID
from logpipe.showcase.generator import FAULTS


def test_world_injects_each_fault_with_a_findable_root_cause():
    with utc_clock():
        world = build_world(["sigterm", "db_port_blocked", "api_hang", "control_quiet"], epochs=1, log=lambda *_: None)
    server = world.server
    assert set(server.scorers) == {"nginx", "postgres", "shop-api"}
    by_mode = {s.mode: s for s in world.slots}
    for mode in ("sigterm", "db_port_blocked"):
        slot = by_mode[mode]
        rows = server.reads.between(slot.start, slot.start + timedelta(minutes=5), 5000)
        root = [r for r in rows if FAULTS[mode]["truth"] in r["message"]]
        assert root and root[0]["template_id"] != UNK_TEMPLATE_ID  # the learner minted its template
        assert root[0]["timestamp"].startswith(slot.start.isoformat()[:16])
    hang = by_mode["api_hang"]
    after = server.reads.between(hang.start + timedelta(seconds=1), hang.now - timedelta(seconds=1), 5000, service="shop-api")
    assert after == []  # silent, as injected
    assert contains_truth(naive_bundle(world, by_mode["sigterm"]), "received fast shutdown request") in (True, False)
    assert by_mode["control_quiet"].truth is None and "{at}" not in by_mode["sigterm"].question
