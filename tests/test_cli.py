from logpipe.cli import main
from tests.conftest import FIXTURES


def test_calibrate_then_replay(tmp_path, capsys):
    registry = tmp_path / "registry.json"
    main(["calibrate", "--logs", str(FIXTURES), "--registry", str(registry)])
    main(["replay", "--logs", str(FIXTURES), "--registry", str(registry), "--quiet"])
    out = capsys.readouterr().out
    assert "registry v1" in out
    assert "0 PENDING" in out
