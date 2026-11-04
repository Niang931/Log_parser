import { Link, useNavigate } from "react-router-dom";
import { usePoll, type Agent } from "../api";
import { Card, Empty, Loading, PageHead, TimeAgo, bytes, fmt } from "../ui";

export function agentProblem(a: Agent): string | null {
  const st = a.status;
  return st.config_error ?? st.shipper?.last_error ?? st.sources?.find((s) => s.error)?.error ?? null;
}

export default function Agents() {
  const { data, error } = usePoll<{ agents: Agent[] }>("/api/agents", 3000);
  const nav = useNavigate();
  return (
    <>
      <PageHead title="Agents" sub="Every host running logpipe-agent, what it collects and whether it is healthy. Agents report every 15 seconds."
        actions={<Link to="/setup" className="btn">Add an agent</Link>} />
      <Card>
        {!data ? <Loading error={error} /> : data.agents.length === 0 ? (
          <Empty title="No agents yet">Install one with the command in the <Link to="/setup">setup guide</Link>.</Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>Host</th><th>Version</th><th className="right">Sources</th><th className="right">Shipped</th><th className="right">Buffered</th><th>Remote config</th><th>Last heartbeat</th><th>Health</th></tr></thead>
              <tbody>
                {data.agents.map((a) => {
                  const problem = agentProblem(a);
                  const go = () => nav(`/agents/${encodeURIComponent(a.hostname)}`);
                  return (
                    <tr key={a.hostname} className="clickable" onClick={go} tabIndex={0} onKeyDown={(e) => e.key === "Enter" && go()}>
                      <td><span className="row" style={{ flexWrap: "nowrap" }}><span className={`dot ${a.online ? "good" : "off"}`} aria-label={a.online ? "online" : "offline"} /><strong>{a.hostname}</strong></span></td>
                      <td className="small">{a.status.version} <span className="muted">{a.status.os}/{a.status.arch}</span></td>
                      <td className="right num">{a.status.sources?.length ?? 0}</td>
                      <td className="right num">{fmt(a.status.shipper?.sent)}</td>
                      <td className="right num">{a.status.shipper?.buffer_bytes ? bytes(a.status.shipper.buffer_bytes) : "–"}</td>
                      <td>{!a.remote.hash ? <span className="muted small">none</span> : a.remote_applied ? <span className="badge good">applied</span> : <span className="badge warn">pending</span>}</td>
                      <td className="small"><TimeAgo iso={a.last_seen} /></td>
                      <td>{!a.online ? <span className="badge">offline</span> : problem ? <span className="badge bad" title={problem}>error</span> : <span className="badge good">ok</span>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  );
}
