from logpipe.eval.synth_iot import ANOMALY_KINDS, generate


def test_generation_is_deterministic_and_labelled():
    a = generate(devices_per_kind=1, history_hours=0.5, validation_hours=0.5, test_hours=1, anomaly_rate=0.2)
    b = generate(devices_per_kind=1, history_hours=0.5, validation_hours=0.5, test_hours=1, anomaly_rate=0.2)
    assert [e.line for e in a.test] == [e.line for e in b.test]
    assert not any(e.anomaly for e in a.history)
    kinds = {e.anomaly for e in a.validation + a.test if e.anomaly}
    assert kinds and kinds <= set(ANOMALY_KINDS)


def test_rollout_happens_only_in_test_and_changes_formats():
    fleet = generate(devices_per_kind=3, history_hours=0.5, validation_hours=0.5, test_hours=2,
                     anomaly_rate=0.0, rollout_share=1.0)
    assert all(e.firmware == 1 for e in fleet.history + fleet.validation)
    upgraded = [e for e in fleet.test if e.firmware == 2]
    assert upgraded and any("fw=2.1.0" in e.line for e in upgraded)
    assert not any("fw=2.1.0" in e.line for e in fleet.history)


def test_per_device_timestamps_strictly_increase():
    fleet = generate(devices_per_kind=1, history_hours=0.5, validation_hours=0.1, test_hours=0.1)
    last = {}
    for e in fleet.history:
        assert e.timestamp > last.get(e.device_id, e.timestamp.min.replace(tzinfo=e.timestamp.tzinfo))
        last[e.device_id] = e.timestamp


def test_write_produces_files_and_labels(tmp_path):
    fleet = generate(devices_per_kind=1, history_hours=0.2, validation_hours=0.2, test_hours=0.2,
                     anomaly_rate=0.3)
    fleet.write(tmp_path)
    assert len(list((tmp_path / "history").glob("*.log"))) == 4
    assert (tmp_path / "labels.tsv").read_text().count("\n") > 1
