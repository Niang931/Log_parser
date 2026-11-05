import json

import pytest

from logpipe import start as wizard


def _answers(monkeypatch, *answers):
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(it))


def test_saved_answers_become_the_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docker-compose.yaml").write_text("services: {}\n")
    monkeypatch.setattr(wizard.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(wizard, "_free_port", lambda _: None)
    ran, launched = [], []
    # demo, kafka, memory store, offline, calibrate, devices, hours, port, start
    _answers(monkeypatch, "1", "kafka", "", "", "", "3", "0.5", "", "y")
    wizard.start(run=lambda cmd, check: ran.append(cmd), launch=launched.append)
    assert ran == [["docker", "compose", "up", "-d", "--wait", "kafka"]]
    assert launched == [["--browser", "camoufox", "demo", "--llm", "offline", "--queue", "kafka",
                         "--store", "memory", "--devices", "3", "--test-hours", "0.5", "--port", "8000"]]
    assert json.loads(wizard.SAVED.read_text())["queue"] == "kafka"

    # Second run: Enter keeps every saved answer, but each question is still asked.
    ran.clear(), launched.clear()
    _answers(monkeypatch, *[""] * 8, "n")
    wizard.start(run=lambda cmd, check: ran.append(cmd), launch=launched.append)
    assert not ran and not launched

    # `logpipe demo` skips the first question.
    _answers(monkeypatch, *[""] * 7, "n")
    wizard.start(launch=launched.append, mode="demo")


def test_pipeline_plan_calibrates_and_trains_only_when_missing(tmp_path):
    a = {"mode": "pipeline", "queue": "memory", "llm_kind": "api", "provider": "anthropic",
         "logs": "logs", "registry": str(tmp_path / "reg.json"), "model": str(tmp_path / "m"),
         "store": "clickhouse"}
    services, commands = wizard.plan(a)
    assert services == ["clickhouse"]
    assert [c[2] for c in commands[:1]] == ["calibrate"] and commands[1][0] == "train-logbert"
    assert "llm:anthropic" in commands[-1]
    (tmp_path / "reg.json").write_text("{}")
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "config.json").write_text("{}")
    assert [c[2] for c in wizard.plan(a)[1]] == ["run"]


def test_choose_rejects_out_of_range(monkeypatch, capsys):
    _answers(monkeypatch, "9", "2")
    assert wizard.choose("q", [("a", ""), ("b", "")], None) == "b"
    assert "pick 1-2" in capsys.readouterr().out
    _answers(monkeypatch)
    with pytest.raises(StopIteration):
        wizard.choose("q", [("a", "")], None)


def test_port_in_use_is_refused():
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        assert "in use" in wizard._free_port(str(sock.getsockname()[1]))
    assert wizard._free_port("http") == "not a port number"


def test_gemini_needs_no_sign_in(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wizard.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(wizard, "_free_port", lambda _: None)
    ran, launched = [], []
    # demo, memory, memory, web, gemini, chromium (offered first), recalibrate, sizes, port,
    # start, sign in? (Enter = no)
    _answers(monkeypatch, "", "", "", "web", "gemini", "", "recalibrate", "", "", "", "y", "")
    wizard.start(run=lambda cmd, check: ran.append(cmd), launch=launched.append)
    assert not ran  # no login window, no camoufox download
    cmd = launched[0]
    assert cmd[:2] == ["--browser", "chrome"] and "llm:web:gemini" in cmd and "--recalibrate" in cmd
    assert json.loads(wizard.SAVED.read_text())["masks"] == "calibrate"


def test_server_plan(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wizard, "_free_port", lambda _: None)
    launched = []
    # (server skips the mode question) memory, clickhouse, offline, port, data dir, demo?, start
    _answers(monkeypatch, "", "clickhouse", "", "9090", "", "y", "n")
    wizard.start(launch=launched.append, mode="server")
    saved = json.loads(wizard.SAVED.read_text())
    assert wizard.plan(saved) == (["clickhouse"], [[
        "--browser", "camoufox", "server", "--llm", "offline", "--queue", "memory",
        "--store", "clickhouse", "--port", "9090", "--data-dir", "artifacts/server", "--demo"]])


def test_free_port_ignores_time_wait_and_names_the_listener():
    import socket

    from logpipe.start import _free_port

    with socket.socket() as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", 0))
        port = srv.getsockname()[1]
        srv.listen()
        cli = socket.create_connection(("127.0.0.1", port))
        conn, _ = srv.accept()
        conn.close()  # the server side closes first -> its end sits in TIME_WAIT
        cli.close()
        busy = _free_port(str(port))
    assert busy and "in use" in busy
    assert _free_port(str(port)) is None  # listener gone; TIME_WAIT alone doesn't block
