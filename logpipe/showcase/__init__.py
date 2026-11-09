"""``logpipe showcase``: a guided tour of the whole product on a generated web shop.

It starts the server, a log generator and a real ``logpipe-agent``, then walks act by act:
connect the agent, draft templates, review them (with a hand-written rule), train, explore,
inject an incident, add a new service. Each act says what to do in the UI and waits until it
has happened; pressing Enter lets the showcase do it instead. ``--auto`` runs every act itself.
"""
from __future__ import annotations

import json
import os
import secrets
import select
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from ..settings import get_settings
from .generator import FILES, ShopGenerator

ROOT = Path("artifacts/showcase")
HOST = "shop-web-1"
PG_RULE = ("%{timestamp:ts} UTC [%{integer:pid}] LOG: duration: %{number:duration_ms} ms "
           "statement: %{data:statement}")
B, D, G, Y, R = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[0m"


class Api:
    def __init__(self, base: str):
        self.base = base
        self.cookie = ""  # the admin session, after login()

    def __call__(self, path: str, body: dict | None = None, method: str | None = None):
        req = urllib.request.Request(self.base + path, method=method or ("POST" if body is not None else "GET"),
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", "Cookie": self.cookie})
        with urllib.request.urlopen(req, timeout=30) as res:
            if set_cookie := res.headers.get("Set-Cookie"):
                self.cookie = set_cookie.split(";", 1)[0]
            return json.loads(res.read() or b"null")

    def login(self, password: str) -> None:
        self("/api/auth/login", {"password": password})


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


def _fit(text: str, indent: int) -> str:
    """One terminal line: a long status that wrapped would break redrawing it in place."""
    width = shutil.get_terminal_size((100, 24)).columns - indent - 1
    return text if len(text) <= width else text[:max(width - 1, 10)] + "…"


class Tour:
    def __init__(self, api: Api, url: str, auto: bool):
        self.api, self.url, self.auto = api, url, auto
        self._live = ""  # the counter line currently drawn under the cursor (terminal only)
        self._live_at = 0.0

    def act(self, n: int, title: str, lines: list[str]) -> None:
        print(f"\n{B}── Act {n}: {title} {'─' * max(0, 60 - len(title))}{R}")
        for line in lines:
            print(f"  {line}")

    def say(self, text: str, style: str = D, indent: int = 4, wrap: bool = False) -> None:
        """A line of output under the current act (cut to the terminal, or ``wrap``ped);
        clears a counter line first."""
        if self._live:
            print("\r\033[K", end="")
            self._live = ""
        width = shutil.get_terminal_size((100, 24)).columns - indent - 1
        for line in textwrap.wrap(text, width) if wrap else [_fit(text, indent)]:
            print(f"{' ' * indent}{style}{line}{R}", flush=True)

    def live(self, text: str) -> None:
        """A counter redrawn in place in a terminal; in a file, a line now and then."""
        if not text or text == self._live:
            return
        if sys.stdout.isatty():
            print(f"\r\033[K    {D}{_fit(text, 4)}{R}", end="", flush=True)
            self._live = text
        elif time.monotonic() - self._live_at >= 10:
            print(f"    {D}{text}{R}", flush=True)
            self._live_at = time.monotonic()

    def wait(self, what: str, done: Callable[[], bool], do: Callable[[], None] | None = None,
             timeout: float = 900, live: Callable[[], str] | None = None,
             steps: Callable[[], list[str]] | None = None, give_up: str = "") -> None:
        """Until ``done()``; Enter (or --auto) runs ``do`` instead of waiting for the UI.
        ``live`` is a counter (lines collected, your review); ``steps`` gives the background
        job's new steps (LLM prompts, answers, epochs), each printed as it happens. With
        ``give_up``, a timeout prints it and the tour goes on instead of stopping."""
        hint = " (or press Enter and I'll do it)" if do else ""
        self.say(f"… waiting: {what}{hint}", Y, 2)
        _drop_typed_ahead()  # an Enter pressed earlier must not do this act for you
        did, start = False, time.monotonic()
        if self.auto and do:
            do()
            did = True
        while True:
            for line in _ask(steps, []) if steps else []:
                self.say(line)
            if _ask(done, False):
                break
            if time.monotonic() - start > timeout:
                if not give_up:
                    raise SystemExit(f"gave up waiting: {what}")
                self.say(give_up, Y, 2, wrap=True)
                return
            if live:
                self.live(_ask(live, "server busy, still waiting"))
            if not did and do and _enter_pressed(1.0):
                self.say("(doing it for you)")
                do()
                did = True
            elif did or not do:
                time.sleep(1.0)
        self.say(f"✓ {what}", G, 2)

    def pause(self, text: str = "press Enter for the next act") -> None:
        if self.auto:
            time.sleep(1)
            return
        self.say(text, D, 2)
        while not _enter_pressed(3600):
            pass


def _ask(fn: Callable, fallback):
    """``fn()``, or ``fallback`` while the server is too busy to answer (it catches up)."""
    try:
        return fn()
    except (TimeoutError, urllib.error.URLError, ConnectionError):
        return fallback


def _drop_typed_ahead() -> None:
    if sys.stdin.isatty():
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)


def _job_steps(api: Api, *kinds: str) -> Callable[[], list[str]]:
    """New steps of the ``kinds`` jobs started from now on, as the Jobs page logs them."""
    old = {j["id"] for j in api("/api/jobs")["jobs"]}
    printed: dict[str, int] = {}

    def new() -> list[str]:
        out = []
        for j in sorted(api("/api/jobs")["jobs"], key=lambda j: j["created_at"]):
            if j["kind"] not in kinds or j["id"] in old:
                continue
            log = [m for _, m in j.get("log") or []]
            if j["id"] not in printed:
                out.append(f"{j['kind']} job started")
            out += log[printed.get(j["id"], 0):]
            if j["status"] == "failed":
                out.append(f"{j['kind']} job failed: {j.get('error')}")
                old.add(j["id"])
            printed[j["id"]] = len(log)
        return out
    return new


def _enter_pressed(seconds: float) -> bool:
    if not sys.stdin.isatty():
        time.sleep(seconds)
        return False
    ready, _, _ = select.select([sys.stdin], [], [], seconds)
    if ready:
        sys.stdin.readline()
        return True
    return False


def _interrupt(*_) -> None:
    raise KeyboardInterrupt


def run(port: int = 8080, auto: bool = False, ask: bool = True) -> None:
    from ..start import SAVED, ask_setup, docker_up, load_env, plan

    load_env()
    # `kill` (SIGTERM) stops the server and agent too, like Ctrl+C.
    signal.signal(signal.SIGTERM, _interrupt)
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
    settings = get_settings()
    password = settings.server.admin_password or secrets.token_urlsafe(9)
    # The server runs as a child process: its settings travel as environment variables.
    env = os.environ | {"CLICKHOUSE_DB": "logpipe_showcase", "LOGPIPE_ADMIN_PASSWORD": password}
    if a["store"] == "clickhouse":  # a fresh database every run
        import clickhouse_connect
        ch = settings.clickhouse
        clickhouse_connect.get_client(host=ch.host, port=ch.port, username=ch.user, password=ch.password,
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
                api("/api/auth/status")
                break
            except (urllib.error.URLError, ConnectionError):
                if procs[0].poll() is not None:
                    raise SystemExit(f"the server stopped; see {ROOT / 'server.out'}") from None
                time.sleep(0.5)
        api.login(password)
        # Shorter thresholds so the tour's new service is drafted within a minute or two.
        api("/api/settings", {"draft_lines": 200, "draft_after_minutes": 2}, "PUT")
        _tour(Tour(api, url, auto), gen, agent_bin, procs, password)
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


def _tour(t: Tour, gen: ShopGenerator, agent_bin: str, procs: list[subprocess.Popen], password: str) -> None:
    api, url = t.api, t.url
    conf = ROOT / "agent"

    def agent(*args: str) -> None:
        print(f"  {D}$ logpipe-agent -c {conf} {' '.join(args)}{R}")
        subprocess.run([agent_bin, "-c", str(conf), *args], check=True, stdout=subprocess.DEVNULL)

    t.act(0, "The server is up", [f"Open {B}{url}{R} in a browser and keep it next to this terminal.",
                                  f"Log in with the admin password {B}{password}{R}.",
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

    def lines() -> int:
        return sum(s["count"] for s in api("/api/setup")["sources"])
    t.wait("a few hundred lines collected", lambda: lines() >= 300, live=lambda: f"{lines()} / 300 lines")

    def batch() -> dict | None:
        """The open first review; you may draft again in the UI, so always the current one."""
        return api("/api/setup")["review"]

    def drafted() -> bool:  # and no newer draft still running
        return batch() is not None and not any(j["kind"] == "draft" and j["status"] in ("queued", "running")
                                               for j in api("/api/jobs")["jobs"])
    t.wait("templates drafted", drafted, lambda: api("/api/review/generate", {"sample_cap": 2000}),
           steps=_job_steps(api, "draft"))
    t.pause()

    def postgres(b: dict | None) -> dict | None:
        drafts = api(f"/api/review/{b['id']}")["drafts"] if b else []
        return next((d for d in drafts if "statement:" in " ".join(d["tokens"])), None)
    pg = postgres(batch())
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

    def reviewed() -> str:
        b = batch()
        if b is None:
            return "applied; training (steps below)"
        return f"your review: {b['approved']} approved, {b['drafts'] - b['approved'] - b['pending']} rejected, {b['pending']} undecided"

    def do_review() -> None:
        b = batch()
        if b is None:  # you already applied it in the UI
            return
        mine = postgres(b)
        if mine and not mine.get("rule_edited"):  # a rule you wrote yourself stays
            api(f"/api/drafts/{mine['id']}", {"rule": PG_RULE}, "PUT")
            t.say(f"set the Postgres rule: {PG_RULE}")
        api(f"/api/review/{b['id']}/decide-all", {"status": "approved"})
        api(f"/api/review/{b['id']}/apply", {})
    t.wait("templates applied and the model trained", review_done, do_review, live=reviewed,
           steps=_job_steps(api, "apply", "retrain"))
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
    t.wait("an anomaly alert", lambda: len(api("/api/alerts")["alerts"]) > before, timeout=180,
           give_up=("no alert this time: the model trained on only a few minutes of logs, so its threshold "
                    "may sit at 100% missed. Let it collect more, Retrain now on Streams, and try another "
                    "incident."),
           live=lambda: f"outage: {max(0, round(gen.incident_until - time.monotonic()))}s left")
    t.pause()

    t.act(7, "A firmware update: the same events, slightly different lines", [
        "shop-api ships v2.1. Checkout lines now carry the customer's currency (EUR, USD or GBP) instead",
        "of always EUR, and 'payment … authorized' now reads 'payment … approved'.",
        "The parser can't match the changed lines, so the online learner (every 30 s) mines templates for",
        "them and checks each against the events it already knows: a changed variable is linked by itself,",
        "a rewording only when the LLM confirms it is the same event.",
        f"Watch {B}Templates{R}: the new ones show up under the old, marked 'linked automatically' or 'linked by LLM'."])
    def templates() -> list[dict]:
        return api("/api/templates")["templates"]
    known = {x["template_id"] for x in templates()}
    t.pause("press Enter to roll out v2.1")
    gen.firmware = True

    def new(word: str) -> dict | None:
        return next((x for x in templates() if x["template_id"] not in known and word in x["text"]), None)
    t.wait("the learner handled the changed lines", lambda: bool(new("approved") and new("checkout started")), timeout=300,
           live=lambda: f"{api('/api/overview')['status'].get('PENDING', 0)} lines waiting for the learner (runs every 30 s)")
    currency, reworded = new("checkout started"), new("approved")
    assert currency is not None and reworded is not None
    old = next(x for x in templates() if "authorized" in x["text"] and x["alias_of"] is None)
    if currency["alias_of"] is not None:
        t.say(f"✓ the checkout line with a currency is linked to #{currency['alias_of']} by the pipeline: only a variable differs", G)
    else:
        t.say(f"the checkout line with a currency became #{currency['template_id']}, not linked", Y)
    if reworded["alias_of"] is not None:
        t.say(f"✓ 'payment … approved' is linked to #{reworded['alias_of']}: the LLM confirmed it is the same event", G)
    else:
        t.say("'payment … approved' stays a new template: a rewording is only linked when an LLM confirms it, and "
              "this run has none. Link it yourself: Maintenance → the template new to the shop-api model → "
              f"Same event as… → #{old['template_id']} (payment … authorized).", Y, 2, wrap=True)

        def link() -> None:
            try:  # also closes its "new to a model" item
                api(f"/api/warnings/novel_template:{reworded['template_id']}/same-as", {"target": old["template_id"]})
            except urllib.error.HTTPError:  # no such item yet
                api(f"/api/templates/{reworded['template_id']}/link", {"target": old["template_id"]})
        t.wait("the reworded line linked to the event it replaces",
               lambda: next(x for x in templates() if x["template_id"] == reworded["template_id"])["alias_of"] is not None, link)
    t.say("Linked lines count as the old event: the shop-api model needs no retrain and raises no alert for the change.", D, 2, wrap=True)
    t.pause()

    t.act(8, "A new service appears", [
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
    def inv_collected() -> str:
        c = next((c for c in api("/api/maintenance")["collecting"] if c["id"] == sid), None)
        return f"inventory: {c['have']} / {c['need']} lines" if c else ""
    t.wait("the inventory service's templates drafted for review", lambda: inv_batch() is not None, timeout=300,
           live=inv_collected, steps=_job_steps(api, "draft"))

    def inv_done() -> bool:
        return any(s["id"] == sid and s["calibrated"] for s in api("/api/sources")["sources"])

    def do_inv() -> None:
        b = inv_batch()
        if b:
            api(f"/api/review/{b['id']}/decide-all", {"status": "approved"})
            api(f"/api/review/{b['id']}/apply", {})
    def inv_reviewed() -> str:
        b = inv_batch()
        return f"your review: {b['approved']} approved, {b['pending']} undecided" if b else "added; retraining"
    t.wait("inventory templates reviewed and added", inv_done, do_inv, live=inv_reviewed,
           steps=_job_steps(api, "apply", "retrain"))

    t.act(9, "That's the tour", [
        "Everything keeps running: new lines, the agent, the models. Try more in the UI:",
        "  write a rule on another template, link two templates, save a query, push a source to the",
        "  agent from its page, or start another incident.",
        f"Files: {ROOT}. Server output: {ROOT / 'server.out'}. Press {B}Ctrl+C{R} to stop everything."])
    while True:
        time.sleep(3600)
