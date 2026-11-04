import { Link } from "react-router-dom";
import { usePoll, type Overview as O, type Stats } from "../api";
import { MinuteChart } from "./Stats";
import { Card, Empty, JobRow, Loading, PageHead, fmt, time } from "../ui";

function Tile({ label, value, foot, to }: { label: string; value: string; foot?: string; to?: string }) {
  const body = <><div className="label">{label}</div><div className="value num">{value}</div>{foot && <div className="foot">{foot}</div>}</>;
  return to ? <Link className="tile" to={to}>{body}</Link> : <div className="tile">{body}</div>;
}

export default function Overview() {
  const { data, error } = usePoll<O>("/api/overview");
  const stats = usePoll<Stats>("/api/stats?minutes=60", 5000);
  if (!data) return <><PageHead title="Overview" /><Loading error={error} /></>;
  const final = data.status.FINAL ?? 0;
  const pending = data.status.PENDING ?? 0;
  const parsed = final + pending ? Math.round((final / (final + pending)) * 1000) / 10 : 0;
  const openWarnings = Object.values(data.warnings).reduce((a, b) => a + b, 0);

  return (
    <>
      <PageHead title="Overview" sub="What is arriving, how much of it is understood, and what needs a look." />
      {!data.settings.calibrated && (
        <div className="callout info" style={{ marginBottom: 16 }}>
          <div className="grow"><strong>Not calibrated yet.</strong> Logs are stored but not parsed into templates until you finish the setup guide.</div>
          <Link to="/setup" className="btn small primary">Continue setup</Link>
        </div>
      )}
      <div className="tiles" style={{ marginBottom: 16 }}>
        <Tile label="Ingest rate" value={`${data.rate.toFixed(1)}/s`} foot={`${fmt(data.received)} lines since start`} to="/logs" />
        <Tile label="Parsed into templates" value={`${parsed}%`} foot={`${fmt(pending)} pending`} to="/templates" />
        <Tile label="Agents online" value={`${data.agents_online} / ${data.agents}`} foot={`${data.sources} sources`} to="/agents" />
        <Tile label="Templates" value={fmt(data.templates)} foot={`registry v${data.registry_version}`} to="/templates" />
        <Tile label="Open warnings" value={fmt(openWarnings)}
          foot={openWarnings ? `${data.warnings.new_source ?? 0} new sources · ${data.warnings.novel_template ?? 0} unseen templates` : "nothing to review"}
          to="/warnings" />
        <Tile label="Anomaly models" value={`${data.models} / ${data.streams}`} foot="streams with a trained LogBERT" to="/streams" />
      </div>
      <div className="grid cols-3">
        <Card title="Lines per minute" hint="last hour, by parse status" className="span-2" >
          {stats.data ? <MinuteChart data={stats.data.per_minute} height={220} /> : <Loading error={stats.error} />}
        </Card>
        <div className="stack">
          {data.jobs.length > 0 && (
            <Card title="Running jobs">
              <div className="stack">{data.jobs.map((j) => <JobRow key={j.id} job={j} />)}</div>
            </Card>
          )}
          <Card title="Recent anomaly alerts" actions={<Link to="/alerts" className="small">All alerts</Link>}>
            {data.recent_alerts.length === 0 ? (
              <Empty title="No alerts">{data.models ? "Sequences look normal." : "Alerts start once a stream has a model."}</Empty>
            ) : (
              <div className="stack" style={{ gap: 10 }}>
                {data.recent_alerts.map((a) => (
                  <div key={a.alert_id}>
                    <div className="spread"><strong>{a.key}</strong><span className="badge bad">{a.reasons.join(", ")}</span></div>
                    <div className="muted small">{time(a.window_start)} – {time(a.window_end)} · miss {Math.round(a.miss_ratio * 100)}%</div>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      </div>
    </>
  );
}
