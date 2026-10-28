from deepparse.tools.build_training_set import build_examples
from deepparse.tools.synth_iot_fleet import generate_iot_fleet, verify_ground_truth, write_dataset


def test_fleet_ground_truth_is_clean_and_trainable(tmp_path):
    logs, entries = generate_iot_fleet(count=2000, seed=42)
    assert len(logs) == len(entries) == 2000
    assert verify_ground_truth(logs, entries) == []
    write_dataset(tmp_path / "IoT_fleet", logs, entries)
    assert build_examples(tmp_path / "IoT_fleet", per_template=2)
