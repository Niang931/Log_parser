import { Link, useNavigate } from "react-router-dom";
import { usePoll, type Agent, type Facets, type Histogram, type Overview as O } from "../api";
import { LEVELS, PARSE, Timeline, TimeRange, Toplist, useRange } from "../charts";
import { agentProblem } from "./Agents";
import { Card, Empty, JobRow, Loading, PageHead, TimeAgo, fmt, time } from "../ui";

function Value({ label, value, foot, to, tone }: { label: string; value: string; foot?: string; to: string; tone?: "bad" | "warn" | "good" }) {
  return (
    <Link className={`qv ${tone ?? ""}`} to={to}>
      <span className="qv-label">{label}</span>
      <span className="qv-value num">{value}</span>
      {foot && <span className="qv-foot">{foot}</span>}
    </Link>
  );
}

export default function Overview() {
  const [range, minutes, setRange] = useRange("1h");
  const nav = useNavigate();
  const { data, error } = usePoll<O>("/api/overview");
  const byLevel = usePoll<Histogram>(`/api/logs/histogram?minutes=${minutes}&by=level`, 5000).data;
  const byParse = usePoll<Histogram>(`/api/logs/histogram?minutes=${minutes}&by=status`, 5000).data;
  const facets = usePoll<Facets>(`/api/logs/facets?minutes=${minutes}`, 5000).data;
  const errors = usePoll<Facets>(`/api/logs/facets?minutes=${minutes}&level=error`, 5000).data;
  const agents = usePoll<{ agents: Agent[] }>("/api/agents", 5000).data?.agents;
  if (!data) return <><PageHead title="Overview" /><Loading error={error} /></>;

  const final = data.status.FINAL ?? 0;
  const pending = data.status.PENDING ?? 0;
  const parsed = final + pending ? Math.round((final / (final + pending)) * 1000) / 10 : 0;
  const openWarnings = Object.values(data.warnings).reduce((a, b) => a + b, 0);
  const levels = Object.fromEntries(facets?.level ?? []) as Record<string, number>;
  const inRange = Object.values(levels).reduce((a, b) => a + b, 0);
  const errorPct = inRange ? ((levels.error ?? 0) / inRange) * 100 : 0;

  return (
    <>
      <PageHead title="Overview" sub="What is arriving, how much of it is understood, and what needs a look."
        actions={<TimeRange value={range} onChange={setRange} />} />
      {!data.settings.calibrated && (
        <div className="callout info" style={{ marginBottom: 12 }}>
          <div className="grow"><strong>Not calibrated yet.</strong> Logs are stored but not parsed into templates until you finish the setup guide.</div>
          <Link to="/setup" className="btn small primary">Continue setup</Link>
        </div>
      )}
      <div className="dash">
        <div className="w-12 qvs">
          <Value label="Ingest rate" value={`${data.rate.toFixed(1)}/s`} foot={`${fmt(data.received)} lines since start`} to="/logs" />
          <Value label="Error logs" value={`${errorPct.toFixed(errorPct < 10 ? 1 : 0)}%`} foot={`${fmt(levels.error ?? 0)} of ${fmt(inRange)} in range`}
            to={`/logs?range=${range}&level=error`} tone={errorPct >= 5 ? "bad" : errorPct >= 1 ? "warn" : undefined} />
          <Value label="Parsed into templates" value={`${parsed}%`} foot={`${fmt(pending)} pending`} to="/templates" tone={data.settings.calibrated && parsed < 90 ? "warn" : undefined} />
          <Value label="Agents online" value={`${data.agents_online}/${data.agents}`} foot={`${data.sources} sources`} to="/agents"
            tone={data.agents_online < data.agents ? "bad" : data.agents ? "good" : undefined} />
          <Value label="Open warnings" value={fmt(openWarnings)} to="/warnings" tone={openWarnings ? "warn" : undefined}
            foot={openWarnings ? `${data.warnings.new_source ?? 0} sources · ${data.warnings.novel_template ?? 0} templates` : "nothing to review"} />
          <Value label="Anomaly alerts" value={fmt(data.recent_alerts.length)} foot={`${data.models}/${data.streams} streams modelled`} to="/alerts"
            tone={data.recent_alerts.length ? "bad" : undefined} />
        </div>

        {data.jobs.length > 0 && (
          <Card title="Running jobs" className="w-12">
            <div className="stack">{data.jobs.map((j) => <JobRow key={j.id} job={j} />)}</div>
          </Card>
        )}

        <Card title="Log volume by status" className="w-8" actions={<Link to={`/logs?range=${range}`} className="small">Open in explorer</Link>}>
          <Timeline data={byLevel} series={[...LEVELS].reverse()} height={200} />
        </Card>
        <Card title="Errors by service" className="w-4">
          {errors ? <Toplist rows={errors.service.map(([s, n]) => [s || "(none)", n, s])} color="var(--lvl-error-soft)" unit="errors"
            onPick={(s) => nav(`/logs?range=${range}&level=error&service=${encodeURIComponent(s)}`)} /> : <Loading />}
        </Card>

        <Card title="Parsing coverage" className="w-8"><Timeline data={byParse} series={PARSE} height={150} /></Card>
        <Card title="Recent anomaly alerts" className="w-4" actions={<Link to="/alerts" className="small">All alerts</Link>}>
          {data.recent_alerts.length === 0 ? (
            <Empty title="No alerts">{data.models ? "Sequences look normal." : "Alerts start once a stream has a model."}</Empty>
          ) : (
            <ul className="events">
              {data.recent_alerts.map((a) => (
                <li key={a.alert_id}>
                  <span className="lvl lvl-error">alert</span>
                  <div><strong>{a.key}</strong> <span className="muted">{a.reasons.join(", ")}</span>
                    <div className="muted small">{time(a.window_start)} – {time(a.window_end)} · miss {Math.round(a.miss_ratio * 100)}%</div></div>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card title="Hosts" className="w-12" actions={<Link to="/agents" className="small">Fleet</Link>}>
          {!agents ? <Loading /> : agents.length === 0 ? <Empty title="No agents yet"><Link to="/setup">Install one</Link></Empty> : (
            <div className="hostmap">
              {agents.map((a) => {
                const problem = agentProblem(a);
                return (
                  <Link key={a.hostname} to={`/agents/${encodeURIComponent(a.hostname)}`}
                    className={`host ${!a.online ? "off" : problem ? "warn" : "ok"}`}
                    title={`${a.hostname}: ${!a.online ? "offline" : problem ? "reporting a problem" : "online"}`}>
                    <span className="host-name">{a.hostname}</span>
                    <span className="host-meta">{!a.online ? <>offline · <TimeAgo iso={a.last_seen} /></> : `${a.status.sources?.length ?? 0} sources · ${fmt(a.status.shipper.sent)} sent`}</span>
                  </Link>
                );
              })}
            </div>
          )}
        </Card>
      </div>
    </>
  );
}
