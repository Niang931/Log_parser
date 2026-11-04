"""``logpipe start`` (or plain ``logpipe``): ask what to run, start the services it needs, run it.

Every answer is saved to ``.logpipe/start.json`` and offered as the default next time; the
questions are still asked, so a saved setup is one Enter per question away.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from deepparse.ai_scraper.sites import SITES
from deepparse.synth.chat_provider import API_KEY_ENV

SAVED = Path(".logpipe/start.json")


def load_env(path: Path = Path(".env")) -> None:
    """Read ``.env`` like ``uv run --env-file`` would, without overriding the real environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key, value.strip().strip('"'))


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        raise SystemExit("\nstopped") from None


def choose(question: str, options: list[tuple[str, str]], default: str | None) -> str:
    """Numbered menu; Enter takes the default (the saved answer), a number or a key picks."""
    keys = [k for k, _ in options]
    default = default if default in keys else keys[0]
    print(f"\n{question}")
    width = max(len(k) for k in keys)
    for n, (key, text) in enumerate(options, 1):
        print(f"  {n}) {key:<{width}}  {text}")
    while True:
        answer = _ask(f"choice [{keys.index(default) + 1}]: ") or default
        if answer.isdigit() and 1 <= int(answer) <= len(keys):
            return keys[int(answer) - 1]
        if answer in keys:
            return answer
        print(f"  pick 1-{len(keys)}")


def text(question: str, default: str, check: Callable[[str], str | None] = lambda _: None) -> str:
    """Free text; ``check`` returns an error message to ask again."""
    while True:
        answer = _ask(f"\n{question} [{default}]: ") or default
        if (error := check(answer)) is None:
            return answer
        print(f"  {error}")


def number(question: str, default: float, cast: type = int) -> float:
    def bad(answer: str) -> str | None:
        try:
            return None if cast(answer) > 0 else "must be positive"
        except ValueError:
            return "not a number"
    return cast(text(question, str(default), bad))


def confirm(question: str, default: bool = True) -> bool:
    answer = _ask(f"\n{question} [{'Y/n' if default else 'y/N'}]: ").lower()
    return default if not answer else answer.startswith("y")


def _exists(kind: str) -> Callable[[str], str | None]:
    return lambda p: None if Path(p).exists() else f"no {kind} at {p}"


def ask_setup(saved: dict) -> dict:
    """The questions; returns the answers (saved as next time's defaults)."""
    a: dict = {}
    a["mode"] = choose("What do you want to run?", [
        ("demo", "browser demo: a live synthetic IoT fleet through every stage"),
        ("pipeline", "the pipeline on your own log files"),
    ], saved.get("mode"))
    a["queue"] = choose("Unresolved queue (lines no template matched wait here to be learned)?", [
        ("memory", "in-process, nothing to install (straightforward)"),
        ("kafka", "Kafka broker in Docker, started for you"),
    ], saved.get("queue"))
    a["llm_kind"] = choose("Who names slot keys and confirms template re-associations?", [
        ("offline", "rules, no model (never re-associates a reworded template)"),
        ("web", "a chat site in a browser you are signed in to, no API key"),
        ("api", "a hosted API, needs its API key in the environment or .env"),
    ], saved.get("llm_kind"))
    if a["llm_kind"] == "web":
        a["site"] = choose("Which chat site?", [(s, "") for s in sorted(SITES)], saved.get("site"))
        a["browser"] = choose("Which browser drives it?", [
            ("camoufox", "bundled Firefox build (fetched on first use)"),
            ("chromium", "the chromium on your PATH"),
        ], saved.get("browser"))
    elif a["llm_kind"] == "api":
        a["provider"] = choose("Which provider?", [
            (p, " or ".join(env)) for p, env in API_KEY_ENV.items()], saved.get("provider"))

    if a["mode"] == "demo":
        a["masks"] = choose("Where do the regex masks come from?", [
            ("calibrate", "calibrate now on the demo history (about a minute)"),
            ("file", "a registry saved on Colab with the fine-tuned model"),
        ], saved.get("masks"))
        if a["masks"] == "file":
            a["registry"] = text("Registry file", saved.get("registry", "demo_registry.json"),
                                 _exists("registry"))
        a["devices"] = number("Devices per kind", saved.get("devices", 2))
        a["test_hours"] = number("Hours of live stream (event time)", saved.get("test_hours", 1), float)
        a["port"] = number("Port for the page", saved.get("port", 8000))
    else:
        a["logs"] = text("Log directory", saved.get("logs", "logs"), _exists("directory"))
        a["registry"] = text("Registry file (created by calibrating if missing)",
                             saved.get("registry", "artifacts/registry.json"))
        a["model"] = text("LogBERT model directory (trained if missing)",
                          saved.get("model", "artifacts/logbert"))
        a["store"] = choose("Where do parsed rows go?", [
            ("memory", "in memory, gone when the run ends"),
            ("clickhouse", "ClickHouse in Docker, started for you"),
        ], saved.get("store"))
    return a


def plan(a: dict) -> tuple[list[str], list[list[str]]]:
    """Docker services to start and the ``logpipe`` argv lists to run, in order."""
    llm = {"offline": "offline", "web": f"llm:web:{a.get('site')}",
           "api": f"llm:{a.get('provider')}"}[a["llm_kind"]]
    top = ["--browser", "chrome" if a.get("browser") == "chromium" else a.get("browser", "camoufox")]
    services = ["kafka"] if a["queue"] == "kafka" else []
    if a["mode"] == "demo":
        cmd = [*top, "demo", "--llm", llm, "--queue", a["queue"], "--devices", str(a["devices"]),
               "--test-hours", str(a["test_hours"]), "--port", str(a["port"])]
        if a["masks"] == "file":
            cmd += ["--registry", a["registry"]]
        return services, [cmd]
    if a["store"] == "clickhouse":
        services.append("clickhouse")
    commands = []
    if not Path(a["registry"]).exists():
        commands.append([*top, "calibrate", "--logs", a["logs"], "--registry", a["registry"], "--llm", llm])
    if not (Path(a["model"]) / "config.json").exists():
        commands.append(["train-logbert", "--logs", a["logs"], "--registry", a["registry"],
                         "--out", a["model"]])
    commands.append([*top, "run", "--logs", a["logs"], "--registry", a["registry"], "--model", a["model"],
                     "--llm", llm, "--queue", a["queue"], "--store", a["store"]])
    return services, commands


def _docker_up(services: list[str], run: Callable) -> None:
    if not shutil.which("docker"):
        raise SystemExit("docker is not on PATH; install Docker or pick the in-memory options")
    if not Path("docker-compose.yaml").exists():
        raise SystemExit("run this from the repository root (docker-compose.yaml not found)")
    if not Path(".env").exists() and Path(".env.example").exists():
        shutil.copy(".env.example", ".env")  # compose reads the ClickHouse settings from it
        load_env()
        print("created .env from .env.example")
    print(f"\n$ docker compose up -d --wait {' '.join(services)}")
    run(["docker", "compose", "up", "-d", "--wait", *services], check=True)


def start(run: Callable = subprocess.run, launch: Callable[[list[str]], None] | None = None) -> None:
    load_env()
    saved = json.loads(SAVED.read_text()) if SAVED.exists() else {}
    if saved:
        print(f"defaults are your last answers ({SAVED}); press Enter to keep each one")
    a = ask_setup(saved)
    services, commands = plan(a)
    SAVED.parent.mkdir(exist_ok=True)
    SAVED.write_text(json.dumps(a | {"signed_in": saved.get("signed_in", [])}, indent=2))

    print("\nThis will run:")
    if services:
        print(f"  docker compose up -d --wait {' '.join(services)}")
    for cmd in commands:
        print("  logpipe " + " ".join(cmd))
    if a["llm_kind"] == "api" and not any(os.environ.get(k) for k in API_KEY_ENV[a["provider"]]):
        print(f"  warning: {' or '.join(API_KEY_ENV[a['provider']])} is not set (add it to .env)")
    if not confirm("Start now?"):
        print("saved; run `logpipe start` again when ready")
        return

    if services:
        _docker_up(services, run)
    if a["llm_kind"] == "web":
        signed_in = saved.get("signed_in", [])
        if confirm(f"Sign in to {a['site']} in a browser window now? (needed once per site)",
                   a["site"] not in signed_in):
            cmd = [sys.executable, "-m", "deepparse.ai_scraper", "login", a["site"], "--browser",
                   "chrome" if a["browser"] == "chromium" else "camoufox"]
            if a["browser"] == "camoufox":
                run([sys.executable, "-m", "camoufox", "fetch"], check=True)
            run(cmd, check=True)
            SAVED.write_text(json.dumps(a | {"signed_in": sorted({*signed_in, a["site"]})}, indent=2))
    if launch is None:
        from .cli import main as launch
    for cmd in commands:
        print(f"\n$ logpipe {' '.join(cmd)}")
        launch(cmd)
