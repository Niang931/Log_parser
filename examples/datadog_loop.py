"""The Datadog loop, hands-free: shop logs -> Datadog Agent -> Datadog -> logpipe (--datadog-pull)
-> templates + LogBERT -> anomaly + RCA events and a grok pipeline back in Datadog.

Usage: python examples/datadog_loop.py [--port 8080] [--normal SECONDS] (`make demo-datadog-loop`)
Needs DD_API_KEY and DD_APP_KEY in .env, and Docker for the Agent.
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

from logpipe.settings import get_settings
from logpipe.showcase import Api
from logpipe.showcase.generator import ShopGenerator
from logpipe.start import load_env

ROOT = Path("artifacts/datadog-loop")
QUERY = "service:(nginx OR shop-api OR postgres)"


def wait(what: str, done, timeout: float = 600, every: float = 5):
    print(f"... {what}", flush=True)
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if result := done():
            return result
        time.sleep(every)
    raise SystemExit(f"gave up waiting for: {what}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--normal", type=float, default=300, help="seconds of normal traffic to learn")
    args = p.parse_args()
    load_env()
    dd = get_settings().datadog
    if not (dd.api_key and dd.app_key):
        raise SystemExit("set DD_API_KEY and DD_APP_KEY in .env")

    shutil.rmtree(ROOT, ignore_errors=True)
    (ROOT / "logs").mkdir(parents=True)
    Path(".logpipe").mkdir(exist_ok=True)
    gen = ShopGenerator(ROOT / "logs")
    gen.start()
    # recreated: the old container's mount points at the directory just deleted
    subprocess.run(
        [
            "docker",
            "compose",
            "--profile",
            "datadog",
            "up",
            "-d",
            "--force-recreate",
            "datadog-agent",
        ],
        check=True,
    )

    password = secrets.token_urlsafe(9)
    env = os.environ | {"LOGPIPE_ADMIN_PASSWORD": password}
    cmd = ["server", "--port", str(args.port), "--data-dir", str(ROOT / "server"), "--datadog"]
    server = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from logpipe.cli import main; main()",
            *cmd,
            "--datadog-pull",
            QUERY,
        ],
        stdout=(ROOT / "server.out").open("w"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    api = Api(f"http://localhost:{args.port}")
    try:

        def up() -> bool:
            try:
                return api("/api/auth/status") is not None
            except (urllib.error.URLError, ConnectionError):
                if server.poll() is not None:
                    raise SystemExit(f"the server stopped; see {ROOT / 'server.out'}") from None
                return False

        wait("server up", up, every=0.5)
        api.login(password)
        print(f"UI: http://localhost:{args.port}  password: {password}")

        def lines() -> int:
            return sum(s["count"] for s in api("/api/setup")["sources"])

        wait(
            "shop logs back from Datadog (Agent -> Datadog -> Logs Search)", lambda: lines() >= 300
        )
        print(f"learning normal traffic for {args.normal:.0f}s", flush=True)
        time.sleep(args.normal)

        api("/api/review/generate", {"sample_cap": 2000})
        batch = wait("templates drafted", lambda: api("/api/setup")["review"])
        api(f"/api/review/{batch['id']}/decide-all", {"status": "approved"})
        api(f"/api/review/{batch['id']}/apply", {})
        wait(
            "templates applied and LogBERT trained",
            lambda: (
                api("/api/settings")["calibrated"]
                and all(s["has_model"] or s.get("waiting") for s in api("/api/streams")["streams"])
            ),
        )
        out = api("/api/datadog/pipeline", {})
        print(f"grok pipeline 'logpipe (auto)' in Datadog: {out['rules']} rules")
        print(f"  {dd.app_url}/logs/pipelines")

        before = len(api("/api/incidents")["incidents"])
        print("payment gateway outage for 60s", flush=True)
        gen.incident(60)
        inc = wait(
            "an anomaly incident",
            lambda: (
                len(api("/api/incidents")["incidents"]) > before
                and api("/api/incidents")["incidents"][0]
            ),
        )
        print(f"incident {inc['id']} on {inc['key']}; posting to Datadog with its RCA")
        wait(
            "the RCA",
            lambda: api(f"/api/incidents/{inc['id']}").get("explanation"),
        )
        print(f"events: {dd.app_url}/event/explorer?query=source%3Alogpipe")
        print("running; Ctrl+C to stop")
        server.wait()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        gen.stop()
        server.terminate()
        server.wait(10)


if __name__ == "__main__":
    main()
