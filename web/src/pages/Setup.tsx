import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { api, useAction, usePoll, type Job, type Settings, type Source, type Stream } from "../api";
import { CopyBlock, JobRow, Loading, PageHead, TimeAgo, fmt, sourceLabel } from "../ui";

interface SetupState {
  keys: number;
  agents: { hostname: string; online: boolean; sources: number }[];
  sources: Source[];
  settings: Settings;
  streams: Stream[];
  jobs: Job[];
}

function Step({ n, title, done, current, children }: {
  n: number; title: string; done: boolean; current: boolean; children: ReactNode;
}) {
  return (
    <li className={`step ${done ? "done" : current ? "current" : ""}`}>
      <span className="step-num" aria-hidden="true">{done ? "✓" : n}</span>
      <div className="step-body card">
        <div className="spread">
          <h2>{title}</h2>
          {done && <span className="badge good">Done</span>}
        </div>
        <div style={{ marginTop: 8 }}>{children}</div>
      </div>
    </li>
  );
}

export function installCommands(token: string) {
  const endpoint = window.location.origin;
  return {
    docker: [
      "docker run -d --name logpipe-agent --restart unless-stopped \\",
      `  -e LOGPIPE_ENDPOINT=${endpoint} \\`,
      `  -e LOGPIPE_API_KEY=${token} \\`,
      "  -e LOGPIPE_HOSTNAME=$(hostname) \\",
      "  -v /var/log:/var/log:ro \\",
      "  -v /var/run/docker.sock:/var/run/docker.sock:ro \\",
      "  -v logpipe-agent:/var/lib/logpipe-agent \\",
      "  ghcr.io/minhthien2103/logpipe-agent:latest",
    ].join("\n"),
    binary: [
      `sudo logpipe-agent init --endpoint ${endpoint} --api-key ${token}`,
      "sudo logpipe-agent track '/var/log/myapp/*.log' --service myapp --auto-multiline",
      "sudo logpipe-agent check && sudo logpipe-agent run",
    ].join("\n"),
  };
}

export default function Setup() {
  const { data, error } = usePoll<SetupState>("/api/setup");
  const [token, setToken] = useState<string | null>(null);
  const [keyName, setKeyName] = useState("first agents");
  const [how, setHow] = useState<"docker" | "binary">("docker");
  const [cap, setCap] = useState<number | null>(null);
  const createKey = useAction((name: string) => api<{ token: string }>("/api/keys", { method: "POST", json: { name } }));
  const calibrate = useAction((sample_cap: number) => api("/api/setup/calibrate", { method: "POST", json: { sample_cap } }));

  if (!data) return <><PageHead title="Setup guide" /><Loading error={error} /></>;
  const calibrated = data.settings.calibrated;
  const job = [...data.jobs].reverse().find((j) => j.kind === "calibrate");
  const running = job && (job.status === "queued" || job.status === "running");
  const done = [data.keys > 0, data.agents.length > 0, data.sources.length > 0, true, calibrated];
  const current = done.findIndex((d) => !d);
  const sampleCap = cap ?? data.settings.sample_cap;
  const cmds = installCommands(token ?? "<your API key>");

  return (
    <>
      <PageHead title="Setup guide"
        sub="Connect every collector and track every log file first. Calibration then learns the log templates from a capped sample of each source and trains an anomaly model per stream." />
      {calibrated && (
        <div className="callout info" style={{ marginBottom: 16 }}>
          <div className="grow">Calibrated <TimeAgo iso={data.settings.calibrated_at} />. Sources added from now on show up under
            {" "}<Link to="/warnings">Warnings</Link>, where you can recalibrate to include them.</div>
          <Link className="btn small" to="/">Go to overview</Link>
        </div>
      )}
      <ol className="steps" style={{ listStyle: "none", padding: 0, margin: 0 }}>
        <Step n={1} title="Create an API key for your agents" done={done[0]} current={current === 0}>
          <p className="ink-2">Agents send it with every request. You can create one per environment and revoke it later under <Link to="/keys">API keys</Link>.</p>
          {token ? (
            <>
              <p><strong>Copy it now</strong>: the key is only shown once.</p>
              <CopyBlock text={token} />
            </>
          ) : (
            <form className="row" onSubmit={(e) => {
              e.preventDefault();
              createKey.mutate(keyName, { onSuccess: (r) => setToken(r.token) });
            }}>
              <input aria-label="Key name" value={keyName} onChange={(e) => setKeyName(e.target.value)} />
              <button className="btn primary" disabled={createKey.isPending}>Create key</button>
              {data.keys > 0 && <span className="muted small">{data.keys} active key{data.keys > 1 ? "s" : ""} already</span>}
            </form>
          )}
          {createKey.error && <div className="error-text">{createKey.error.message}</div>}
        </Step>

        <Step n={2} title="Install the agent on each host" done={done[1]} current={current === 1}>
          <div className="tabs" role="tablist">
            <button role="tab" aria-selected={how === "docker"} className={how === "docker" ? "active" : ""} onClick={() => setHow("docker")}>Docker</button>
            <button role="tab" aria-selected={how === "binary"} className={how === "binary" ? "active" : ""} onClick={() => setHow("binary")}>Binary</button>
          </div>
          <CopyBlock text={cmds[how]} />
          <div style={{ marginTop: 12 }}>
            <h3>Connected agents</h3>
            {data.agents.length === 0 ? <p className="muted">Waiting for the first agent to connect… (it shows up here the moment it starts)</p> : (
              <div className="row">
                {data.agents.map((a) => (
                  <Link key={a.hostname} to={`/agents/${encodeURIComponent(a.hostname)}`} className="badge">
                    <span className={`dot ${a.online ? "good" : "off"}`} />{a.hostname} · {a.sources} sources
                  </Link>
                ))}
              </div>
            )}
          </div>
        </Step>

        <Step n={3} title="Track the log files on every host" done={done[2]} current={current === 2}>
          <p className="ink-2">
            On a host: <code>logpipe-agent track '/var/log/app/*.log' --service app --multiline '^\d{"{4}"}-'</code>.
            Or open an agent under <Link to="/agents">Agents</Link> and push sources to it from here.
            Let each source collect a few hundred lines before calibrating.
          </p>
          {data.sources.length > 0 && (
            <div className="table-wrap">
              <table>
                <thead><tr><th>Source</th><th className="right">Lines so far</th><th>First seen</th></tr></thead>
                <tbody>
                  {data.sources.map((s) => (
                    <tr key={s.id}>
                      <td>{sourceLabel(s.id)} <span className="muted small">{s.source}</span></td>
                      <td className="right num">{fmt(s.count)}</td>
                      <td><TimeAgo iso={s.first_seen} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Step>

        <Step n={4} title="Group sources into streams (optional)" done={done[3] && data.streams.length > 1} current={false}>
          <p className="ink-2">
            Each stream gets its own LogBERT model, trained on the sequences of the sources it selects. The
            default <strong>All logs</strong> stream covers everything; add streams for sources that behave
            alike (e.g. <code>host: gw-*</code> for IoT gateways).
          </p>
          <div className="row">
            {data.streams.map((s) => <span key={s.id} className="badge">{s.name}</span>)}
            <Link to="/streams" className="btn small">Manage streams</Link>
          </div>
        </Step>

        <Step n={5} title="Calibrate" done={done[4]} current={current === 4 || current === -1}>
          <p className="ink-2">
            Takes up to <strong>{fmt(sampleCap)}</strong> lines per source, picked by entropy priority sampling,
            builds the template registry, back-fills every stored line, then trains one model per stream.
          </p>
          {!calibrated && !running && (
            <form className="row" onSubmit={(e) => { e.preventDefault(); calibrate.mutate(sampleCap); }}>
              <label className="field">
                Lines per source
                <input type="number" min={100} max={200000} step={100} value={sampleCap}
                  onChange={(e) => setCap(Number(e.target.value))} style={{ width: 140 }} />
              </label>
              <button className="btn primary" style={{ alignSelf: "flex-end" }}
                disabled={!data.sources.length || calibrate.isPending}>
                Start calibration
              </button>
              {!data.sources.length && <span className="muted small" style={{ alignSelf: "flex-end" }}>Needs at least one source.</span>}
            </form>
          )}
          {calibrate.error && <div className="error-text">{calibrate.error.message}</div>}
          {job && <div style={{ marginTop: 12 }}><JobRow job={job} /></div>}
        </Step>
      </ol>
    </>
  );
}
