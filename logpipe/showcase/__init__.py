"""``logpipe showcase``: a guided tour of the whole product on a generated web shop.

It starts the server, a log generator and a real ``logpipe-agent``, then walks act by act:
connect the agent, draft templates, review them (with a hand-written rule), train, explore,
inject an incident, add a new service. Each act says what to do in the UI and waits until it
has happened; pressing Enter lets the showcase do it instead. ``--auto`` runs every act itself.
"""
from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .generator import FILES, ShopGenerator

ROOT = Path("artifacts/showcase")
HOST = "shop-web-1"
PG_RULE = ("%{timestamp:ts} UTC [%{integer:pid}] LOG: duration: %{number:duration_ms} ms "
           "statement: %{data:statement}")
B, D, G, Y, R = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[0m"


class Api:
    def __init__(self, base: str):
        self.base = base

    def __call__(self, path: str, body: dict | None = None, method: str | None = None):
        req = urllib.request.Request(self.base + path, method=method or ("POST" if body is not None else "GET"),
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as res:
            return json.loads(res.read() or b"null")


def find_agent(out: Path) -> str:
    """A logpipe-agent binary: built from ./agent with Go, else the one on PATH."""
    if shutil.which("go") and Path("agent/go.mod").exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"{D}$ go build -o {out} ./agent{R}", flush=True)
        subprocess.run(["go", "build", "-o", str(out.resolve()), "."], cwd="agent", check=True)
        return str(out)
    if found := shutil.which("logpipe-agent"):
        return found
    raise SystemExit("logpipe-agent not found: install Go (to build ./agent) or put logpipe-agent on PATH")


class Tour:
    def __init__(self, api: Api, url: str, auto: bool):
        self.api, self.url, self.auto = api, url, auto

    def act(self, n: int, title: str, lines: list[str]) -> None:
        print(f"\n{B}── Act {n}: {title} {'─' * max(0, 60 - len(title))}{R}")
        for line in lines:
            print(f"  {line}")

    def wait(self, what: str, done: Callable[[], bool], do: Callable[[], None] | None = None,
             timeout: float = 900) -> None:
        """Until ``done()``; Enter (or --auto) runs ``do`` instead of waiting for the UI."""
        hint = " (or press Enter and I'll do it)" if do else ""
        print(f"  {Y}… waiting: {what}{hint}{R}", flush=True)
        did, start = False, time.monotonic()
        if self.auto and do:
            do()
            did = True
        while not done():
            if time.monotonic() - start > timeout:
                raise SystemExit(f"gave up waiting: {what}")
            if not did and do and _enter_pressed(1.0):
                do()
                did = True
            elif did or not do:
                time.sleep(1.0)
        print(f"  {G}✓ {what}{R}")

    def pause(self, text: str = "press Enter for the next act") -> None:
        if self.auto:
            time.sleep(1)
            return
        print(f"  {D}{text}{R}", flush=True)
        while not _enter_pressed(3600):
            pass


def _enter_pressed(seconds: float) -> bool:
    if not sys.stdin.isatty():
        time.sleep(seconds)
        return False
    ready, _, _ = select.select([sys.stdin], [], [], seconds)
    if ready:
        sys.stdin.readline()
        return True
    return False


def run(port: int = 8080, auto: bool = False, ask: bool = True) -> None:
    from ..start import SAVED, ask_setup, docker_up, load_env, plan

    load_env()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]  # progress shows up live, even in a file
    auto = auto or not sys.stdin.isatty()  # nobody to press Enter
    if ask and not auto:
        saved = json.loads(SAVED.read_text()) if SAVED.exists() else {}
        print("The showcase runs `logpipe server`; pick its store, queue and LLM (Enter keeps the default).")
        a = ask_setup(saved | {"server_port": port}, "server")
    else:
        a = {"mode": "server", "queue": "memory", "store": "memory", "llm_kind": "offline", "server_port": port}
    a |= {"data_dir": str(ROOT / "server"), "with_demo": False}
    services, [cmd] = plan(a)
    port = int(a["server_port"])
    if services:
        docker_up(services)
    env = os.environ | {"CLICKHOUSE_DB": "logpipe_showcase"}
    if a["store"] == "clickhouse":  # a fresh database every run
        import clickhouse_connect
        clickhouse_connect.get_client(
            host=env.get("CLICKHOUSE_HOST", "127.0.0.1"), port=int(env.get("CLICKHOUSE_PORT", "8123")),
            username=env.get("CLICKHOUSE_USER", "default"), password=env.get("CLICKHOUSE_PASSWORD", ""),
        ).command("DROP DATABASE IF EXISTS logpipe_showcase")
    shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    agent_bin = find_agent(ROOT / "bin" / "logpipe-agent")
    url = f"http://localhost:{port}"
    api = Api(url)
    procs: list[subprocess.Popen] = []
    gen = ShopGenerator(ROOT / "logs")
    try:
        print(f"{D}$ logpipe {' '.join(cmd)}   (output in {ROOT / 'server.out'}){R}")
        procs.append(subprocess.Popen([sys.executable, "-c", "from logpipe.cli import main; main()", *cmd],
                                      stdout=(ROOT / "server.out").open("w"), stderr=subprocess.STDOUT, env=env))
        for _ in range(120):
            try:
                api("/api/overview")
                break
            except (urllib.error.URLError, ConnectionError):
                if procs[0].poll() is not None:
                    raise SystemExit(f"the server stopped; see {ROOT / 'server.out'}") from None
                time.sleep(0.5)
        # Shorter thresholds so the tour's new service is drafted within a minute or two.
        api("/api/settings", {"draft_lines": 200, "draft_after_minutes": 2}, "PUT")
        _tour(Tour(api, url, auto), gen, agent_bin, procs)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        gen.stop()
        for p in reversed(procs):
            p.terminate()
        for p in procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()


def _tour(t: Tour, gen: ShopGenerator, agent_bin: str, procs: list[subprocess.Popen]) -> None:
    api, url = t.api, t.url
    conf = ROOT / "agent"

    def agent(*args: str) -> None:
        print(f"  {D}$ logpipe-agent -c {conf} {' '.join(args)}{R}")
        subprocess.run([agent_bin, "-c", str(conf), *args], check=True, stdout=subprocess.DEVNULL)

    t.act(0, "The server is up", [f"Open {B}{url}{R} in a browser and keep it next to this terminal.",
                                  "Every act says what to look at; do it in the UI, or press Enter and I'll do it."])
    t.pause()

    gen.start()
    t.act(1, "A web shop is writing logs", [
        f"A generator writes a shop's logs into {ROOT / 'logs'} (about 3 orders a second):",
        *(f"  {svc:<9} {ROOT / 'logs' / FILES[svc][0]}" for svc in ("nginx", "shop-api", "postgres")),
        "The Java API also writes multiline stack traces when something fails.",
        f"Watch one: {D}tail -f {ROOT / 'logs' / FILES['shop-api'][0]}{R}"])
    t.pause()

    t.act(2, "Connect a collector agent", [
        f"In the UI: {B}Agents → Add an agent{R} → create a key and copy it.",
        "Paste the key here and press Enter (or just press Enter and I'll create one).",
        "Then I install the agent as your user, no sudo, and track the three files."])
    token = "" if t.auto or not sys.stdin.isatty() else input("  API key: ").strip()
    if not token.startswith("lp_"):
        token = api("/api/keys", {"name": "showcase"})["token"]
        print("  created a key named 'showcase'")
    agent("init", "--endpoint", url, "--api-key", token, "--hostname", HOST)
    for svc in ("nginx", "shop-api", "postgres"):
        path, source, multi = FILES[svc]
        extra = ["--multiline", multi] if multi else []
        agent("track", str((ROOT / "logs" / path).resolve()), "--service", svc, "--source", source,
              "--start", "beginning", *extra)
    procs.append(subprocess.Popen([agent_bin, "-c", str(conf), "run"],
                                  stdout=(ROOT / "agent.log").open("w"), stderr=subprocess.STDOUT))

    def online() -> bool:
        agents = api("/api/agents")["agents"]
        return any(a["hostname"] == HOST and a["online"] and len(a["status"].get("sources") or []) >= 3 for a in agents)
    t.wait(f"agent {HOST} online with 3 sources (see it appear on the Agents page)", online)
    t.pause()

    t.act(3, "Draft log templates from a sample", [
        f"In the UI: {B}Setup guide{R} → step through to {B}Sample{R} → {B}Draft templates{R}.",
        "Each source gives its most varied recent lines (priority sampling); Drain mines templates from them.",
        "Nothing reaches the parser yet: they become drafts for you to review."])

    def has_lines() -> bool:
        return sum(s["count"] for s in api("/api/setup")["sources"]) >= 300
    t.wait("a few hundred lines collected", has_lines)
    t.wait("templates drafted", lambda: api("/api/setup")["review"] is not None,
           lambda: api("/api/review/generate", {"sample_cap": 2000}))
    t.pause()

    batch = api("/api/setup")["review"]
    drafts = api(f"/api/review/{batch['id']}")["drafts"]
    pg = next((d for d in drafts if "statement:" in " ".join(d["tokens"])), None)
    t.act(4, "Review the templates, then calibrate", [
        f"In the UI: {B}Setup guide → Review{R}. Click a template to see the lines it matches.",
        *([f"The Postgres one came out as {D}… statement: <*> <*> <*> …{R}, one slot per word. Click {B}Edit{R}",
           f"and replace the rule with:  {B}{PG_RULE}{R}",
           "The table under it shows what each field extracts from real lines, and lines it misses."] if pg else []),
        f"Then {B}Approve undecided{R} and {B}Apply … and train{R}: templates go live, stored lines are",
        "back-filled, and LogBERT trains on each service's sequence of templates."])

    def review_done() -> bool:
        streams = api("/api/streams")["streams"]
        return api("/api/settings")["calibrated"] and all(s["has_model"] or s.get("waiting") for s in streams)

    def do_review() -> None:
        if pg:
            api(f"/api/drafts/{pg['id']}", {"rule": PG_RULE}, "PUT")
            print(f"  set the Postgres rule: {PG_RULE}")
        api(f"/api/review/{batch['id']}/decide-all", {"status": "approved"})
        api(f"/api/review/{batch['id']}/apply", {})
    t.wait("templates applied and the model trained", review_done, do_review)
    t.pause()

    t.act(5, "Look around", [
        f"{B}Log Explorer{R}: click a line; its side panel shows the template and the extracted fields",
        "  (a Postgres line now has duration_ms and statement). Stack traces are one event, not ten lines.",
        f"{B}Templates{R}: the Postgres template is marked 'rule'; open it to edit or link templates.",
        f"{B}Query → Builder{R}: Show {B}Average of @duration_ms{R}, By {B}service{R}, tick {B}over time{R}, Run.",
        f"{B}Overview / Analytics{R}: volume by level, errors by service, parsing coverage."])
    t.pause()

    t.act(6, "An incident: the payment gateway goes down", [
        "I'll switch the shop to an outage for a minute: payments retry and time out, orders fail",
        "with Java stack traces, nginx returns 502, Postgres runs out of connections.",
        f"Watch {B}Anomaly alerts{R}: LogBERT flags the API's broken order of events (sequence alerts).",
        f"{B}Maintenance{R} also lists the new error templates as 'new to a model', not as anomalies."])
    before = len(api("/api/alerts")["alerts"])
    t.pause("press Enter to start the incident")
    gen.incident(60)
    t.wait("an anomaly alert", lambda: len(api("/api/alerts")["alerts"]) > before, timeout=240)
    t.pause()

    t.act(7, "A new service appears", [
        "The shop starts an inventory service with its own log file, and I track it on the agent.",
        f"In {B}Maintenance{R} it first shows as 'collecting lines'; at 200 lines its templates are",
        "drafted on their own and wait for your review there. Approve them and click Add."])
    t.pause("press Enter to start the inventory service")
    gen.inventory = True
    inv_path, inv_source, _ = FILES["inventory"]
    gen.path("inventory").parent.mkdir(parents=True, exist_ok=True)
    gen.path("inventory").touch()
    agent("track", str((ROOT / "logs" / inv_path).resolve()), "--service", "inventory", "--source", inv_source,
          "--start", "beginning")
    sid = f"{HOST}|inventory|{inv_source}"

    def inv_batch() -> dict | None:
        return next((b for b in api("/api/maintenance")["review"] if sid in b["sources"]), None)
    t.wait("the inventory service's templates drafted for review", lambda: inv_batch() is not None, timeout=300)

    def inv_done() -> bool:
        return any(s["id"] == sid and s["calibrated"] for s in api("/api/sources")["sources"])

    def do_inv() -> None:
        b = inv_batch()
        if b:
            api(f"/api/review/{b['id']}/decide-all", {"status": "approved"})
            api(f"/api/review/{b['id']}/apply", {})
    t.wait("inventory templates reviewed and added", inv_done, do_inv)

    t.act(8, "That's the tour", [
        "Everything keeps running: new lines, the agent, the models. Try more in the UI:",
        "  write a rule on another template, link two templates, save a query, push a source to the",
        "  agent from its page, or start another incident.",
        f"Files: {ROOT}. Server output: {ROOT / 'server.out'}. Press {B}Ctrl+C{R} to stop everything."])
    while True:
        time.sleep(3600)
