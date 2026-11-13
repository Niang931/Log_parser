"""``logpipe start`` (or plain ``logpipe``): ask what to run, start the services it needs, run it.

Every answer is saved to ``.logpipe/start.json`` and offered as the default next time; the
questions are still asked, so a saved setup is one Enter per question away.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from deepparse.ai_scraper.sites import SITES
from deepparse.synth.chat_provider import API_KEY_ENV
from dotenv import load_dotenv

from .settings import get_settings

SAVED = Path(".logpipe/start.json")


def load_env(path: Path = Path(".env")) -> None:
    """Read ``.env`` like ``uv run --env-file`` would, without overriding the real environment."""
    load_dotenv(path, override=False)
    get_settings.cache_clear()  # settings read before now missed these


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


def _free_port(answer: str) -> str | None:
    if not answer.isdigit():
        return "not a port number"
    with socket.socket() as sock:
        # Same as uvicorn: a just-stopped server leaves TIME_WAIT connections for about a
        # minute, which block a plain bind but not a real server.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", int(answer)))
        except OSError:
            who = _listener(int(answer))
            return f"port {answer} is in use{f' by {who}' if who else ''}; stop it or pick another"
    return None


def _listener(port: int) -> str | None:
    """'logpipe (pid 123)' for whatever listens on ``port``, when the OS tells us."""
    try:
        out = subprocess.run(
            ["ss", "-ltnpH", f"sport = :{port}"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r'users:\(\("([^"]+)",pid=(\d+)', out)
    return f"{m.group(1)} (pid {m.group(2)})" if m else None


def ask_password() -> str:
    """The UI's admin password ('' keeps the stored one, or generates one on a first start)."""
    import getpass

    while True:
        try:
            pw = getpass.getpass(
                "Admin password for the UI (Enter keeps the current one; a new "
                "server generates and prints one): "
            )
        except EOFError:
            raise SystemExit("\nstopped") from None
        if not pw:
            return ""
        if len(pw) < 8:
            print("  at least 8 characters")
        elif getpass.getpass("Again: ") == pw:
            return pw
        else:
            print("  they don't match")


def ask_setup(saved: dict, mode: str | None = None) -> dict:
    """The questions; returns the answers (saved as next time's defaults)."""
    a: dict = {}
    a["mode"] = mode or choose(
        "What do you want to run?",
        [
            ("demo", "browser demo: a live synthetic IoT fleet through every stage"),
            ("pipeline", "the pipeline on your own log files"),
            ("server", "the server for logpipe-agent hosts, with the operator UI"),
        ],
        saved.get("mode"),
    )
    a["queue"] = choose(
        "Unresolved queue (lines no template matched wait here to be learned)?",
        [
            ("memory", "in-process, nothing to install (straightforward)"),
            ("kafka", "Kafka broker in Docker, started for you"),
        ],
        saved.get("queue"),
    )
    a["store"] = choose(
        "Where do parsed rows go?",
        [
            (
                "memory",
                "in memory, gone when the run ends"
                + (" (API keys and agents too)" if a["mode"] == "server" else ""),
            ),
            (
                "clickhouse",
                "ClickHouse in Docker, started for you"
                + (" (the demo's Query tab then runs on it)" if a["mode"] == "demo" else ""),
            ),
            ("mongodb", "MongoDB in Docker, started for you (everything except raw SQL)"),
        ],
        saved.get("store"),
    )
    a["llm_kind"] = choose(
        "Who names slot keys and confirms template re-associations?",
        [
            ("offline", "rules, no model (never re-associates a reworded template)"),
            ("web", "a chat site driven in a browser (web scrape), no API key"),
            ("api", "a hosted API, needs its API key in the environment or .env"),
        ],
        saved.get("llm_kind"),
    )
    if a["llm_kind"] == "web":
        a["site"] = choose(
            "Which chat site?",
            [
                (name, "works signed out" if site.anonymous else "needs a one-time sign-in")
                for name, site in sorted(SITES.items())
            ],
            saved.get("site"),
        )
        browsers = [
            ("camoufox", "bundled Firefox build (one-time download)"),
            ("chromium", "the chromium on your PATH"),
        ]
        if shutil.which("chromium"):  # nothing to download, so offer it first
            browsers.reverse()
        a["browser"] = choose("Which browser drives it?", browsers, saved.get("browser"))
    elif a["llm_kind"] == "api":
        a["provider"] = choose(
            "Which provider?",
            [(p, " or ".join(env)) for p, env in API_KEY_ENV.items()],
            saved.get("provider"),
        )

    if a["mode"] == "demo":
        a["masks"] = choose(
            "Where do the regex masks come from?",
            [
                (
                    "calibrate",
                    "calibrate on the demo history once, then reuse it (artifacts/demo/)",
                ),
                ("recalibrate", "calibrate again, replacing that saved calibration"),
                ("file", "a registry saved on Colab with the fine-tuned model"),
            ],
            saved.get("masks"),
        )
        if a["masks"] == "file":
            a["registry"] = text(
                "Registry file", saved.get("registry", "demo_registry.json"), _exists("registry")
            )
        a["devices"] = number("Devices per kind", saved.get("devices", 2))
        a["test_hours"] = number(
            "Hours of live stream (event time)", saved.get("test_hours", 1), float
        )
        a["port"] = int(text("Port for the page", str(saved.get("port", 8000)), _free_port))
    elif a["mode"] == "server":
        a["server_port"] = int(
            text("Port for agents and the UI", str(saved.get("server_port", 8080)), _free_port)
        )
        a["data_dir"] = text(
            "Directory for the registry and models", saved.get("data_dir", "artifacts/server")
        )
        a["with_demo"] = confirm(
            "Also serve the developer demo at /demo/?", saved.get("with_demo", False)
        )
    else:
        a["logs"] = text("Log directory", saved.get("logs", "logs"), _exists("directory"))
        a["registry"] = text(
            "Registry file (created by calibrating if missing)",
            saved.get("registry", "artifacts/registry.json"),
        )
        a["model"] = text(
            "LogBERT model directory (trained if missing)", saved.get("model", "artifacts/logbert")
        )
    return a


def plan(a: dict) -> tuple[list[str], list[list[str]]]:
    """Docker services to start and the ``logpipe`` argv lists to run, in order."""
    llm = {
        "offline": "offline",
        "web": f"llm:web:{a.get('site')}",
        "api": f"llm:{a.get('provider')}",
    }[a["llm_kind"]]
    top = [
        "--browser",
        "chrome" if a.get("browser") == "chromium" else a.get("browser", "camoufox"),
    ]
    services = [
        name
        for name, on in (
            ("kafka", a["queue"] == "kafka"),
            ("clickhouse", a["store"] == "clickhouse"),
            ("mongo", a["store"] == "mongodb"),
        )
        if on
    ]
    if a["mode"] == "demo":
        cmd = [
            *top,
            "demo",
            "--llm",
            llm,
            "--queue",
            a["queue"],
            "--store",
            a["store"],
            "--devices",
            str(a["devices"]),
            "--test-hours",
            str(a["test_hours"]),
            "--port",
            str(a["port"]),
        ]
        if a["masks"] == "file":
            cmd += ["--registry", a["registry"]]
        elif a["masks"] == "recalibrate":
            cmd.append("--recalibrate")
        return services, [cmd]
    if a["mode"] == "server":
        cmd = [
            *top,
            "server",
            "--llm",
            llm,
            "--queue",
            a["queue"],
            "--store",
            a["store"],
            "--port",
            str(a["server_port"]),
            "--data-dir",
            a["data_dir"],
        ]
        return services, [cmd + ["--demo"] if a["with_demo"] else cmd]
    commands = []
    if not Path(a["registry"]).exists():
        commands.append(
            [*top, "calibrate", "--logs", a["logs"], "--registry", a["registry"], "--llm", llm]
        )
    if not (Path(a["model"]) / "config.json").exists():
        commands.append(
            ["train-logbert", "--logs", a["logs"], "--registry", a["registry"], "--out", a["model"]]
        )
    commands.append(
        [
            *top,
            "run",
            "--logs",
            a["logs"],
            "--registry",
            a["registry"],
            "--model",
            a["model"],
            "--llm",
            llm,
            "--queue",
            a["queue"],
            "--store",
            a["store"],
        ]
    )
    return services, commands


def docker_up(services: list[str], run: Callable = subprocess.run) -> None:
    """``docker compose up -d --wait`` (a no-op for services already healthy)."""
    if not shutil.which("docker"):
        raise RuntimeError("docker is not on PATH; install Docker or pick the in-memory options")
    if not Path("docker-compose.yaml").exists():
        raise RuntimeError("run from the repository root (docker-compose.yaml not found)")
    if not Path(".env").exists() and Path(".env.example").exists():
        shutil.copy(".env.example", ".env")  # compose reads the ClickHouse settings from it
        print("created .env from .env.example")
    load_env()
    print(f"$ docker compose up -d --wait {' '.join(services)}", flush=True)
    try:
        run(["docker", "compose", "up", "-d", "--wait", *services], check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"docker compose up {' '.join(services)} failed ({error.returncode})"
        ) from error


def _save(saved: dict, a: dict, signed_in: list[str]) -> None:
    SAVED.parent.mkdir(exist_ok=True)
    # A one-off recalibration must not become the default.
    masks = {"masks": "calibrate"} if a.get("masks") == "recalibrate" else {}
    # Merge so answers to questions this run skipped (another mode, another LLM) survive.
    SAVED.write_text(json.dumps(saved | a | masks | {"signed_in": signed_in}, indent=2))


def start(
    run: Callable = subprocess.run,
    launch: Callable[[list[str]], None] | None = None,
    mode: str | None = None,
) -> None:
    """``mode`` skips the first question (``logpipe demo`` asks only the demo's questions)."""
    load_env()
    saved = json.loads(SAVED.read_text()) if SAVED.exists() else {}
    if saved:
        print(f"defaults are your last answers ({SAVED}); press Enter to keep each one")
    a = ask_setup(saved, mode)
    services, commands = plan(a)
    _save(saved, a, saved.get("signed_in", []))

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
        try:
            docker_up(services, run)
        except RuntimeError as error:
            raise SystemExit(str(error)) from None
    if a["llm_kind"] == "web":
        signed_in = saved.get("signed_in", [])
        if a["browser"] == "camoufox" and confirm(
            "Download or update the camoufox browser now? (needed once, and after camoufox "
            "upgrades)",
            "camoufox" not in signed_in,
        ):
            run([sys.executable, "-m", "camoufox", "fetch"], check=True)
            signed_in = sorted({*signed_in, "camoufox"})
        anonymous = SITES[a["site"]].anonymous
        question = (
            f"Sign in to {a['site']}? Optional, it answers signed out"
            if anonymous
            else f"Sign in to {a['site']} in a browser window now? (needed once)"
        )
        if confirm(question, not anonymous and a["site"] not in signed_in):
            run(
                [
                    sys.executable,
                    "-m",
                    "deepparse.ai_scraper",
                    "login",
                    a["site"],
                    "--browser",
                    "chrome" if a["browser"] == "chromium" else "camoufox",
                ],
                check=True,
            )
            signed_in = sorted({*signed_in, a["site"]})
        _save(saved, a, signed_in)
    if a["mode"] == "server" and not get_settings().server.admin_password:
        os.environ["LOGPIPE_ADMIN_PASSWORD"] = ask_password()  # this run only; never saved
        get_settings.cache_clear()
    if launch is None:
        from .cli import main as launch
    for cmd in commands:
        print(f"\n$ logpipe {' '.join(cmd)}")
        launch(cmd)
