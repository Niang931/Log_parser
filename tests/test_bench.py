"""``logpipe bench``: a real server under rising load, run from the UI as a job."""
from __future__ import annotations

from logpipe.bench import table
from tests.test_server import env, run_jobs  # noqa: F401 - env is a fixture


def test_bench_job_records_a_run(env):  # noqa: F811
    server, client, _ = env
    assert client.post("/api/bench", json={"store": "clickhouse"}).status_code == 400  # memory server
    assert client.post("/api/bench", json={"store": "mongodb"}).status_code == 400
    job = client.post("/api/bench", json={"max_rate": 500, "step_seconds": 2}).json()
    assert job["kind"] == "bench"
    run_jobs(server)
    runs = client.get("/api/bench").json()["runs"]
    assert len(runs) == 1
    run = runs[0]
    assert run["store"] == "memory" and run["host"]["cpus"] >= 1
    paced = [s for s in run["steps"] if s["target"]]
    assert paced[0]["target"] == 500 and paced[0]["kept_up"] and paced[0]["processed_per_s"] > 400
    assert paced[0]["parse_us"]["p99"] > 0 and paced[0]["cpu_pct"] > 0
    assert run["max_sustained_per_s"] >= 400 and run["flood_per_s"] > 0
    assert "max sustained" in table(run)
    assert client.delete(f"/api/bench/{run['id']}").json() == {"ok": True}
    assert client.get("/api/bench").json()["runs"] == []
