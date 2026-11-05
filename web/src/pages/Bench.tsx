// Benchmark: how many lines/s this machine's server keeps up with, at what CPU and memory.
// Each run is `logpipe bench` in its own process; runs are kept so they can be compared.
import { useState } from "react";
import { api, useAction, usePoll, type Job } from "../api";
import { Card, Empty, JobRow, Loading, PageHead, TimeAgo, fmt } from "../ui";

type Step = {
  target: number | null; offered_per_s: number; processed_per_s: number; rejected: number; backlog: number;
  cpu_pct: number; rss_mb: number; parse_us: { p50?: number; p95?: number; p99?: number };
  inserts: number; insert_ms: { p50?: number; p99?: number }; kept_up: boolean;
};
type Run = {
  id: string; started_at: string; finished_at: string; store: string; step_seconds: number;
  host: { cpus: number; machine: string; system: string; python: string; commit: string; hostname: string };
  max_sustained_per_s: number; falls_behind_at: number | null; flood_per_s: number | null; peak_rss_mb: number; steps: Step[];
};

function Steps({ run }: { run: Run }) {
  return (
    <div className="table-wrap">
      <table>
        <caption className="sr-only">Benchmark steps</caption>
        <thead><tr>
          <th className="right">Target/s</th><th className="right">Offered/s</th><th className="right">Processed/s</th>
          <th className="right">CPU %</th><th className="right">RSS MB</th><th className="right">Parse p50 µs</th>
          <th className="right">p99 µs</th><th className="right">Insert p99 ms</th><th>Kept up</th>
        </tr></thead>
        <tbody>
          {run.steps.map((s, i) => (
            <tr key={i}>
              <td className="right num">{s.target === null ? "flood" : fmt(s.target)}</td>
              <td className="right num">{fmt(s.offered_per_s)}</td>
              <td className="right num"><strong>{fmt(s.processed_per_s)}</strong></td>
              <td className="right num">{s.cpu_pct}</td>
              <td className="right num">{s.rss_mb}</td>
              <td className="right num">{s.parse_us.p50 ?? "–"}</td>
              <td className="right num">{s.parse_us.p99 ?? "–"}</td>
              <td className="right num">{s.insert_ms.p99 ?? "–"}</td>
              <td>{s.target === null ? <span className="muted">ceiling</span>
                : s.kept_up ? <span className="badge good">yes</span> : <span className="badge bad">fell behind</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Bench() {
  const { data, error } = usePoll<{ runs: Run[]; cpus: number }>("/api/bench", 5000);
  const jobs = usePoll<{ jobs: Job[] }>("/api/jobs", 2000).data?.jobs.filter((j) => j.kind === "bench").slice(0, 3) ?? [];
  const [store, setStore] = useState("memory");
  const [maxRate, setMaxRate] = useState("");
  const [stepSeconds, setStepSeconds] = useState("10");
  const [open, setOpen] = useState<string | null>(null);
  const start = useAction(() => api("/api/bench", { method: "POST", json: { store, max_rate: Number(maxRate) || 0, step_seconds: Number(stepSeconds) || 10 } }));
  const remove = useAction((id: string) => api(`/api/bench/${id}`, { method: "DELETE" }));
  const latest = data?.runs[0];
  const shown = data?.runs.find((r) => r.id === open) ?? latest;

  return (
    <>
      <PageHead title="Benchmark" sub="Feeds a fresh server the synthetic IoT fleet at rising rates until it can't keep up. It runs in its own process, so this server keeps working; the load generator's CPU is included." />
      <div className="stack">
        <Card title="Run a benchmark" hint="About a minute at 10 s per step. ClickHouse runs use a throwaway database.">
          <form className="row" onSubmit={(e) => { e.preventDefault(); start.mutate(); }}>
            <label className="row small">Store
              <select value={store} onChange={(e) => setStore(e.target.value)}>
                <option value="memory">memory (parser and pipeline only)</option>
                <option value="clickhouse">ClickHouse (includes writes)</option>
              </select>
            </label>
            <label className="row small">Highest rate
              <input type="number" min={100} step={100} placeholder="100000" value={maxRate} onChange={(e) => setMaxRate(e.target.value)} style={{ width: 110 }} />
            </label>
            <label className="row small">Seconds per step
              <input type="number" min={2} max={120} value={stepSeconds} onChange={(e) => setStepSeconds(e.target.value)} style={{ width: 70 }} />
            </label>
            <button className="btn primary" disabled={start.isPending}>Start</button>
          </form>
          {start.error && <p className="error-text" role="alert">{start.error.message}</p>}
          {jobs.length > 0 && <div style={{ marginTop: 12 }}>{jobs.map((j) => <JobRow key={j.id} job={j} />)}</div>}
        </Card>
        {!data ? <Loading error={error} /> : !shown ? <Empty title="No runs yet">Start one above; results are kept for comparison.</Empty> : (
          <>
            <div className="tiles">
              <div className="tile"><div className="label">Max sustained</div><div className="value num">{fmt(shown.max_sustained_per_s)}<span className="small muted"> lines/s</span></div>
                <div className="foot">{shown.falls_behind_at ? `falls behind at ${fmt(shown.falls_behind_at)}/s` : "kept up with every step"}</div></div>
              <div className="tile"><div className="label">Ceiling (flood)</div><div className="value num">{shown.flood_per_s ? fmt(shown.flood_per_s) : "–"}<span className="small muted"> lines/s</span></div>
                <div className="foot">offered as fast as the intake takes it</div></div>
              <div className="tile"><div className="label">CPU at max sustained</div>
                <div className="value num">{shown.steps.filter((s) => s.kept_up).at(-1)?.cpu_pct ?? "–"}<span className="small muted"> %</span></div>
                <div className="foot">100% = one core; {shown.host.cpus} cores on {shown.host.hostname}</div></div>
              <div className="tile"><div className="label">Peak memory</div><div className="value num">{fmt(Math.round(shown.peak_rss_mb))}<span className="small muted"> MB</span></div>
                <div className="foot">RSS of the benchmark process</div></div>
            </div>
            <Card title={`Steps, ${shown.store} store`} hint={<>Run <TimeAgo iso={shown.started_at} />{shown.host.commit && <> at commit <span className="mono">{shown.host.commit}</span></>}, {shown.step_seconds} s per step</>}>
              <Steps run={shown} />
            </Card>
            <Card title="All runs" hint="Pick one to see its steps">
              <div className="table-wrap">
                <table>
                  <thead><tr><th>When</th><th>Store</th><th className="right">Max sustained/s</th><th className="right">Ceiling/s</th><th className="right">Peak MB</th><th>Host</th><th>Commit</th><th /></tr></thead>
                  <tbody>
                    {data.runs.map((r) => (
                      <tr key={r.id} className={r.id === shown.id ? "selected" : ""}>
                        <td><button className="btn small ghost" onClick={() => setOpen(r.id)} aria-pressed={r.id === shown.id}><TimeAgo iso={r.started_at} /></button></td>
                        <td>{r.store}</td>
                        <td className="right num">{fmt(r.max_sustained_per_s)}</td>
                        <td className="right num">{r.flood_per_s ? fmt(r.flood_per_s) : "–"}</td>
                        <td className="right num">{Math.round(r.peak_rss_mb)}</td>
                        <td className="small">{r.host.hostname} · {r.host.cpus} cores · Python {r.host.python}</td>
                        <td className="mono small">{r.host.commit || "–"}</td>
                        <td className="right"><button className="btn small ghost danger" onClick={() => confirm("Delete this run?") && remove.mutate(r.id)}>Delete</button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </>
        )}
      </div>
    </>
  );
}
