"""``logpipe showcase --datadog``: the Datadog Agent ships the shop's files to Datadog next to
logpipe-agent, and the tour shows what logpipe writes back. Everything it creates in Datadog
(dashboard, monitors, pipeline) is deleted on exit; events can't be."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

import httpx

from ..datadog_api import DatadogAPI
from ..datadog_grok import PIPELINE_NAME
from ..settings import get_settings

B, D, R = "\033[1m", "\033[2m", "\033[0m"


class ShowcaseDatadog:
    def __init__(self, host: str):
        settings = get_settings().datadog
        if not (settings.api_key and settings.app_key):
            raise SystemExit("showcase --datadog needs DD_API_KEY and DD_APP_KEY in .env")
        try:
            import datadog  # noqa: F401
            import ddtrace  # noqa: F401
        except ImportError:
            raise SystemExit(
                "showcase --datadog needs the extra: uv sync --extra datadog"
            ) from None
        self.settings, self.host = settings, host
        self.api = DatadogAPI(settings)
        self.created: list[str] = []  # API paths to DELETE on exit
        self.links: dict[str, str] = {}
        self.started = int(time.time())
        self.had_pipeline = True

    def env(self, url: str) -> dict[str, str]:
        """For the server process: links in events point at this showcase, and a fast tick."""
        return {"DD_PUBLIC_URL": url, "DD_PULL_EVERY_S": "5", "DD_APM": "1", "DD_ENV": "showcase"}

    def start(self) -> None:
        """The Agent (recreated: the showcase just replaced the directory it reads), the
        dashboard and the monitors."""
        Path(".logpipe").mkdir(exist_ok=True)
        subprocess.run(
            ["docker", "compose", "--profile", "datadog", "up", "-d", "--force-recreate"]
            + ["datadog-agent"],
            check=True,
            env=os.environ | {"DD_HOSTNAME": self.host},
        )
        docs = Path("docs/datadog")
        dash = json.loads((docs / "dashboard.json").read_text())
        found = next(
            (
                d
                for d in self.api.send("GET", "/api/v1/dashboard")["dashboards"]
                if d["title"] == dash["title"]
            ),
            None,
        )
        if found is None:
            found = self.api.send("POST", "/api/v1/dashboard", json=dash)
            self.created.append(f"/api/v1/dashboard/{found['id']}")
        self.links["dashboard"] = f"{self.settings.app_url}{found['url']}"
        for path in sorted(docs.glob("monitor-*.json")):
            mon = json.loads(path.read_text())
            if not self.api.send("GET", "/api/v1/monitor", params={"name": mon["name"]}):
                made = self.api.send("POST", "/api/v1/monitor", json=mon)
                self.created.append(f"/api/v1/monitor/{made['id']}")
        self.links["monitors"] = f"{self.settings.app_url}/monitors/manage?q=service%3Alogpipe"
        self.links["logs"] = (
            f"{self.settings.app_url}/logs?query=host%3A{self.host}%20-service%3Alogpipe"
        )
        self.links["events"] = f"{self.settings.app_url}/event/explorer?query=source%3Alogpipe"
        self.links["pipelines"] = f"{self.settings.app_url}/logs/pipelines"
        self.links["map"] = f"{self.settings.app_url}/apm/map?env=showcase"
        self.links["llm"] = (
            f"{self.settings.app_url}/llm/traces?query=%40ml_app%3A{self.settings.llmobs_ml_app}"
        )
        self.had_pipeline = self.pipeline()

    def event(self, title: str) -> bool:
        """An event of this showcase whose title starts with ``title`` is in Datadog."""
        data = self.api.send(
            "GET",
            "/api/v2/events",
            params={
                "filter[from]": str(self.started * 1000),
                "sort": "-timestamp",
                "page[limit]": 100,
            },
        ).get("data", [])
        titles = (
            d["attributes"].get("title") or d["attributes"].get("attributes", {}).get("title", "")
            for d in data
        )
        return any(t.startswith(title) for t in titles)

    def pipeline(self) -> bool:
        return any(
            p["name"] == PIPELINE_NAME
            for p in self.api.send("GET", "/api/v1/logs/config/pipelines")
        )

    def webhook(self, url: str, token: str, service: str) -> str:
        """What a Datadog monitor's webhook sends; returns the Ask plan id it started."""
        body = {
            "title": f"[Triggered] {service} error rate is high",
            "date": str(int(time.time() * 1000)),
            "tags": f"service:{service},env:showcase",
            "host": self.host,
            "id": f"showcase-{uuid.uuid4().hex[:8]}",
        }
        req = urllib.request.Request(
            f"{url}/intake/v1/datadog/webhook",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "X-API-Key": token},
        )
        with urllib.request.urlopen(req, timeout=30) as res:
            return json.loads(res.read())["id"]

    def cleanup(self) -> None:
        if not self.had_pipeline:  # exported during the tour, from the UI or by the tour
            self.created += [
                f"/api/v1/logs/config/pipelines/{p['id']}"
                for p in self.api.send("GET", "/api/v1/logs/config/pipelines")
                if p["name"] == PIPELINE_NAME
            ]
        for path in reversed(self.created):
            try:
                self.api.send("DELETE", path)
                print(f"  {D}deleted in Datadog: {path}{R}")
            except httpx.HTTPError as error:
                print(f"  could not delete {path} in Datadog: {error}")
