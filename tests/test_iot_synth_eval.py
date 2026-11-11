import random

from logpipe.eval.synth_iot import ANOMALY_KINDS, LiveFleet, generate


def test_generation_is_deterministic_and_labelled():
    a = generate(
        devices_per_kind=1, history_hours=0.5, validation_hours=0.5, test_hours=1, anomaly_rate=0.2
    )
    b = generate(
        devices_per_kind=1, history_hours=0.5, validation_hours=0.5, test_hours=1, anomaly_rate=0.2
    )
    assert [e.line for e in a.test] == [e.line for e in b.test]
    assert not any(e.anomaly for e in a.history)
    kinds = {e.anomaly for e in a.validation + a.test if e.anomaly}
    assert kinds and kinds <= set(ANOMALY_KINDS)


def test_rollout_happens_only_in_test_and_changes_formats():
    fleet = generate(
        devices_per_kind=3,
        history_hours=0.5,
        validation_hours=0.5,
        test_hours=2,
        anomaly_rate=0.0,
        rollout_share=1.0,
    )
    assert all(e.firmware == 1 for e in fleet.history + fleet.validation)
    upgraded = [e for e in fleet.test if e.firmware == 2]
    assert upgraded and any("fw=2.1.0" in e.line for e in upgraded)
    assert not any("fw=2.1.0" in e.line for e in fleet.history)


def test_per_device_timestamps_strictly_increase():
    fleet = generate(devices_per_kind=1, history_hours=0.5, validation_hours=0.1, test_hours=0.1)
    last = {}
    for e in fleet.history:
        assert e.timestamp > last.get(
            e.device_id, e.timestamp.min.replace(tzinfo=e.timestamp.tzinfo)
        )
        last[e.device_id] = e.timestamp


def test_write_produces_files_and_labels(tmp_path):
    fleet = generate(
        devices_per_kind=1,
        history_hours=0.2,
        validation_hours=0.2,
        test_hours=0.2,
        anomaly_rate=0.3,
    )
    fleet.write(tmp_path)
    assert len(list((tmp_path / "history").glob("*.log"))) == 4
    assert (tmp_path / "labels.tsv").read_text().count("\n") > 1


def test_live_fleet_is_ordered_and_obeys_rollout_and_injection():
    fleet = generate(1, 0.2, 0.1, 0, anomaly_rate=0.0)
    live = LiveFleet(
        fleet.devices,
        fleet.validation[-1].timestamp,
        0.3,
        random.Random(1),
        anomaly_rate=0.0,
        rollout_at=None,
        rollout_share=1.0,
    )
    plug = next(d for d in live.devices if d.startswith("PL"))
    live.inject(plug, "stuck")
    events = []
    while (e := live.next()) is not None:
        events.append(e)
        if len(events) == 50:
            assert live.rollout() == sorted(live.devices)
    assert [e.timestamp for e in events] == sorted(e.timestamp for e in events)
    assert {e.anomaly for e in events if e.device_id == plug} == {None, "stuck"}
    assert all(e.anomaly is None for e in events if e.device_id != plug)
    # Firmware 2 only after the rollout; an episode already queued keeps its version.
    assert events[0].firmware == 1 and events[-1].firmware == 2
    try:
        live.inject(plug, "reorder")
    except ValueError:
        pass
    else:
        raise AssertionError("reorder needs a door lock")
