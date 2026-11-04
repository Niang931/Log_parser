import csv
import io
import zipfile

from deepparse.tools.sample_loghub_full import sample_system


def test_reservoir_caps_per_template_and_excludes_eval_lines(tmp_path):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["LineId", "Content", "EventId", "EventTemplate"])
    writer.writeheader()
    for i in range(50):
        writer.writerow({"LineId": i, "Content": f"open file {i}", "EventId": "E1",
                         "EventTemplate": "open file <*>"})
    writer.writerow({"LineId": 99, "Content": "shutdown", "EventId": "E2",
                     "EventTemplate": "shutdown"})
    path = tmp_path / "X.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("X/X_full.log_structured.csv", buf.getvalue())

    logs, entries = sample_system(path, per_template=3, exclude={"open file 7"}, seed=1)
    opened = [line for line, e in zip(logs, entries) if e["template"] == "open file <*>"]
    assert len(opened) == 3 and "open file 7" not in opened
    assert "shutdown" in logs
    assert sample_system(path, 3, {"open file 7"}, seed=1) == (logs, entries)  # deterministic
