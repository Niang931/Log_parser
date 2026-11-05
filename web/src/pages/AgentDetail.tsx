import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, useAction, usePoll, type Agent } from "../api";
import { agentProblem } from "./Agents";
import { Card, Empty, Loading, PageHead, TimeAgo, bytes, fmt } from "../ui";

const SNIPPETS: Record<string, string> = {
  "File": "  - type: file\n    path: /var/log/myapp/*.log\n    service: myapp\n    source: java\n    log_processing_rules:\n      - type: multi_line\n        name: new_event\n        pattern: '^\\d{4}-\\d{2}-\\d{2}'\n",
  "journald": "  - type: journald\n    unit: nginx.service\n",
  "Docker": "  - type: docker\n    container_labels: [logs=true]\n",
  "Syslog (UDP)": "  - type: udp\n    port: 5514\n    service: network\n",
  "Mask secrets": "    # add under a source's log_processing_rules:\n      - type: mask_sequences\n        pattern: 'password=\\S+'\n        replace_placeholder: 'password=[REDACTED]'\n",
};

function RemoteConfig({ agent }: { agent: Agent }) {
  const saved = agent.remote.yaml ?? "";
  const [text, setText] = useState(saved);
  const [dirty, setDirty] = useState(false);
  useEffect(() => { if (!dirty) setText(saved); }, [saved, dirty]);
  const save = useAction((yaml: string) => api(`/api/agents/${encodeURIComponent(agent.hostname)}/remote-config`, { method: "PUT", json: { yaml } }));
  const add = (snippet: string) => {
    const base = text.trim() ? text.replace(/\s*$/, "\n") : "logs:\n";
    setText(base + snippet);
    setDirty(true);
  };
  return (
    <Card title="Push sources to this agent"
      hint={<>Written on the host as <code>conf.d/remote.yaml</code> within a second while the agent is connected; files written by hand on the host are never touched.</>}
      actions={agent.remote.hash ? (agent.remote_applied ? <span className="badge good">applied by the agent</span> : <span className="badge warn">waiting for the agent</span>) : null}>
      <div className="row" style={{ marginBottom: 8 }}>
        <span className="muted small">Insert:</span>
        {Object.entries(SNIPPETS).map(([k, v]) => <button key={k} className="btn small" type="button" onClick={() => add(v)}>{k}</button>)}
      </div>
      <form className="stack" style={{ gap: 8 }} onSubmit={(e) => { e.preventDefault(); save.mutate(text, { onSuccess: () => setDirty(false) }); }}>
        <textarea rows={12} value={text} spellCheck={false} aria-label="Remote config YAML"
          placeholder={"logs:\n  - type: file\n    path: /var/log/app/*.log\n    service: app"}
          onChange={(e) => { setText(e.target.value); setDirty(true); }} />
        <div className="row">
          <button className="btn primary" disabled={!dirty || save.isPending}>Save &amp; push</button>
          {dirty && <button type="button" className="btn ghost" onClick={() => { setText(saved); setDirty(false); }}>Discard</button>}
          {saved && !dirty && <button type="button" className="btn ghost danger" onClick={() => save.mutate("")}>Remove all pushed sources</button>}
          {save.error && <span className="error-text">{save.error.message}</span>}
        </div>
      </form>
    </Card>
  );
}

export default function AgentDetail() {
  const { host = "" } = useParams();
  const nav = useNavigate();
  const { data, error } = usePoll<Agent>(`/api/agents/${encodeURIComponent(host)}`, 3000);
  const forget = useAction(() => api(`/api/agents/${encodeURIComponent(host)}`, { method: "DELETE" }));
  if (!data) return <><PageHead title={host} /><Loading error={error} /></>;
  const st = data.status;
  const problem = agentProblem(data);
  const configs = new Map((st.config ?? []).map((c) => [c.type + ":" + (c.path ?? "") + (c.unit ?? "") + (c.port ? `:${c.port}` : ""), c]));

  return (
    <>
      <PageHead title={data.hostname}
        sub={<span className="row"><span className={`dot ${data.online ? "good" : "off"}`} />{data.online ? "Online" : "Offline"} · last heartbeat <TimeAgo iso={data.last_seen} /> · agent {st.version} on {st.os}/{st.arch}</span>}
        actions={<>
          <Link className="btn" to={`/logs?host=${encodeURIComponent(data.hostname)}`}>View its logs</Link>
          <button className="btn ghost danger" onClick={() => confirm("Forget this agent? It reappears on its next heartbeat.") && forget.mutate(undefined, { onSuccess: () => nav("/agents") })}>Forget</button>
        </>} />
      <div className="stack">
        {problem && <div className="callout bad"><div><strong>Problem reported:</strong> {problem}</div></div>}
        <Card title="What it collects" hint="Every source from its conf.d, with live counters">
          {!st.sources?.length ? <Empty title="Nothing tracked yet">Run <code>logpipe-agent track PATH</code> on the host, or push a source below.</Empty> : (
            <div className="table-wrap">
              <table>
                <thead><tr><th>Type</th><th>Target</th><th>Service</th><th>Rules</th><th>From</th><th className="right">Lines</th><th className="right">Bytes</th><th>Last line</th></tr></thead>
                <tbody>
                  {st.sources.map((s) => {
                    const c = configs.get(s.id);
                    return (
                      <tr key={s.id}>
                        <td><span className="badge">{s.type}</span></td>
                        <td className="mono small">{s.target}
                          {s.files && s.files.length > 0 && s.type === "file" && <div className="muted">{s.files.length} file{s.files.length > 1 ? "s" : ""}: {s.files.slice(0, 3).join(", ")}{s.files.length > 3 ? "…" : ""}</div>}
                          {s.files && s.type === "docker" && <div className="muted">{s.files.join(", ")}</div>}
                          {s.error && <div className="error-text">{s.error}</div>}
                        </td>
                        <td className="small">{c?.service || s.service || <span className="muted">auto</span>}</td>
                        <td className="small">{c?.log_processing_rules?.map((r) => <div key={r.type + r.pattern} className="mono" title={r.pattern}>{r.type}</div>) ?? <span className="muted">–</span>}</td>
                        <td className="small mono">{c?.file ?? "–"}</td>
                        <td className="right num">{fmt(s.lines)}</td>
                        <td className="right num">{bytes(s.bytes)}</td>
                        <td className="small"><TimeAgo iso={s.last_line_at} /></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Card>
        <div className="grid cols-2">
          <Card title="Shipping">
            <dl className="kv">
              <dt>Shipped</dt><dd className="num">{st.shipper.sent.toLocaleString()} events</dd>
              <dt>Last send</dt><dd><TimeAgo iso={st.shipper.last_send_at} /></dd>
              <dt>Disk buffer</dt><dd>{bytes(st.shipper.buffer_bytes)}{st.shipper.buffer_bytes > 0 && " (server unreachable or slow)"}</dd>
              <dt>Dropped</dt><dd>{st.shipper.dropped}</dd>
              {st.shipper.last_error && <><dt>Last error</dt><dd className="error-text">{st.shipper.last_error}</dd></>}
            </dl>
          </Card>
          <Card title="Agent">
            <dl className="kv">
              <dt>Endpoint</dt><dd className="mono small">{st.endpoint}</dd>
              <dt>Tags</dt><dd>{st.tags?.length ? st.tags.map((t) => <span key={t} className="badge" style={{ marginRight: 4 }}>{t}</span>) : "–"}</dd>
              <dt>Running since</dt><dd><TimeAgo iso={st.started_at} /></dd>
              <dt>First seen</dt><dd><TimeAgo iso={data.first_seen} /></dd>
              <dt>API key</dt><dd>{data.key_name}</dd>
            </dl>
          </Card>
        </div>
        <RemoteConfig agent={data} />
      </div>
    </>
  );
}
