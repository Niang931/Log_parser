// Anomaly alerts, grouped into incidents (one service's alerts close together in time). The list
// says what to look at; an incident's page shows the lines the model found unusual and what it
// expected instead, what else broke at the same time, an explanation, and what was decided.
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, useAction, usePoll, type Incident, type IncidentDetail, type IncidentStatus } from "../api";
import { Card, Empty, JobRow, Loading, PageHead, Switch, TimeAgo, time } from "../ui";

export const STATUS: Record<IncidentStatus, { label: string; badge: string }> = {
  open: { label: "Open", badge: "bad" },
  acknowledged: { label: "Acknowledged", badge: "warn" },
  resolved: { label: "Resolved", badge: "good" },
  false_positive: { label: "False positive", badge: "" },
};

function span(start: string, end: string) {
  const s = Math.max(0, Math.round((new Date(end).getTime() - new Date(start).getTime()) / 1000));
  return s < 60 ? `${s}s` : s < 3600 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

const FILTERS = { active: (i: Incident) => i.status === "open" || i.status === "acknowledged", all: () => true,
  resolved: (i: Incident) => i.status === "resolved" || i.status === "false_positive" };

export default function Alerts() {
  const { data, error } = usePoll<{ incidents: Incident[] }>("/api/incidents", 5000);
  const [filter, setFilter] = useState<keyof typeof FILTERS>("active");
  const go = useNavigate();
  const shown = data?.incidents.filter(FILTERS[filter]) ?? [];
  return (
    <>
      <PageHead title="Anomaly alerts"
        sub={<>Stretches of a service's logs its LogBERT model finds unusual, grouped into incidents. Kinds of lines a model has never seen are not alerts: they are listed under <Link to="/maintenance">Maintenance</Link>.</>} />
      <Card actions={
        <div className="seg" role="group" aria-label="Show">
          {(Object.keys(FILTERS) as (keyof typeof FILTERS)[]).map((f) => (
            <button key={f} type="button" className={filter === f ? "on" : ""} aria-pressed={filter === f} onClick={() => setFilter(f)}>
              {f === "active" ? "Needs attention" : f === "all" ? "All" : "Closed"} {data?.incidents.filter(FILTERS[f]).length ?? ""}
            </button>
          ))}
        </div>}>
        {!data ? <Loading error={error} /> : shown.length === 0 ? <Empty title={filter === "active" ? "Nothing needs attention" : "No incidents"}>Nothing unusual so far.</Empty> : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>Status</th><th>Service</th><th>When</th><th className="right">Alerts</th><th className="right">Worst</th><th>What happened</th></tr></thead>
              <tbody>
                {shown.map((i) => (
                  <tr key={i.id} className="clickable" tabIndex={0} onClick={() => go(`/alerts/${i.id}`)}
                    onKeyDown={(e) => e.key === "Enter" && go(`/alerts/${i.id}`)}>
                    <td><span className={`badge ${STATUS[i.status].badge}`}>{STATUS[i.status].label}</span></td>
                    <td><strong>{i.service || i.host}</strong> <span className="muted small">on {i.host}</span></td>
                    <td className="small nowrap">{time(i.start)} <span className="muted">· {span(i.start, i.end)}</span></td>
                    <td className="right num">{i.alerts}</td>
                    <td className="right num">{Math.round(i.peak_miss * 100)}%</td>
                    <td className="small ellipsis" style={{ maxWidth: 380 }}>{i.explanation?.summary ?? <span className="muted">Open it to see the unusual lines</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  );
}

function Decide({ inc }: { inc: IncidentDetail }) {
  const [note, setNote] = useState(inc.note);
  const set = useAction((status: IncidentStatus) => api(`/api/incidents/${inc.id}`, { method: "PUT", json: { status, note } }));
  const button = (status: IncidentStatus, label: string, cls = "") => inc.status !== status && (
    <button className={`btn small ${cls}`} disabled={set.isPending} onClick={() => set.mutate(status)}>{label}</button>
  );
  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="row">
        {button("acknowledged", "Acknowledge")}
        {button("resolved", "Resolve", "primary")}
        {button("false_positive", "Not a problem (false positive)")}
        {inc.status !== "open" && button("open", "Reopen", "ghost")}
      </div>
      <input placeholder="Note: what you found, what you did" aria-label="Note" value={note} onChange={(e) => setNote(e.target.value)} />
      <p className="muted small" style={{ margin: 0 }}>
        A false positive is how this service normally behaves: its stretches are added to the training data, so the next retrain of <Link to="/streams">{inc.stream}</Link> stops flagging them.
      </p>
      {set.error && <div className="error-text small">{set.error.message}</div>}
    </div>
  );
}

function Explanation({ inc }: { inc: IncidentDetail }) {
  const run = useAction(() => api(`/api/incidents/${inc.id}/explain`, { method: "POST" }));
  const job = inc.explain_job;
  const busy = run.isPending || job?.status === "queued" || job?.status === "running";
  const e = inc.explanation;
  return (
    <Card title="What happened" actions={
      <button className="btn small" disabled={busy} onClick={() => run.mutate()}>{e ? "Explain again" : "Explain with AI"}</button>}>
      {e ? (
        <div className="stack" style={{ gap: 8 }}>
          <p style={{ margin: 0 }}>{e.summary}</p>
          <div><span className="muted small">Likely cause</span><p style={{ margin: "2px 0 0" }}>{e.likely_cause}</p></div>
          {!!e.next_steps.length && <div><span className="muted small">Next steps</span>
            <ol style={{ margin: "4px 0 0", paddingLeft: 20 }}>{e.next_steps.map((s, i) => <li key={i}>{s}</li>)}</ol></div>}
          <span className="muted small">Written <TimeAgo iso={e.at} />{e.note ? ` · ${e.note}` : ""}</span>
        </div>
      ) : !job && <p className="muted" style={{ margin: 0 }}>The LLM reads the unusual lines below and what other services logged meanwhile, and writes a summary, a likely cause and next steps.</p>}
      {job && <div style={{ marginTop: 10 }}><JobRow job={job} /></div>}
      {run.error && <div className="error-text small">{run.error.message}</div>}
    </Card>
  );
}

const MARK = { expected: "", unexpected: "unexpected", unseen: "never seen" };

export function IncidentPage() {
  const { id } = useParams();
  const { data: inc, error } = usePoll<IncidentDetail>(`/api/incidents/${id}`, 4000);
  const [onlyUnusual, setOnlyUnusual] = useState(false);
  if (!inc) return <><PageHead title="Incident" /><Loading error={error} /></>;
  const unusual = inc.events.filter((e) => e.status !== "expected").length;
  const events = onlyUnusual ? inc.events.filter((e) => e.status !== "expected") : inc.events;
  return (
    <>
      <PageHead title={`${inc.service || inc.host} on ${inc.host}`}
        sub={<><span className={`badge ${STATUS[inc.status].badge}`}>{STATUS[inc.status].label}</span> {time(inc.start)} to {time(inc.end)} · {inc.alerts} alert{inc.alerts === 1 ? "" : "s"}, up to {Math.round(inc.peak_miss * 100)}% of events mispredicted · <Link to="/alerts">All incidents</Link></>} />
      <div className="stack">
        <Card title="Decision"><Decide inc={inc} /></Card>
        <Explanation inc={inc} />
        <Card title={`The lines (${unusual} of ${inc.events.length} unusual)`}
          hint={inc.model_current ? "Each line masked in turn: did the model expect it there, and what it expected instead."
            : "The model was retrained since this incident: marks come from the current model."}
          actions={<Switch checked={onlyUnusual} onChange={setOnlyUnusual} label="Only unusual" />}>
          {events.length === 0 ? <Empty title="No lines">The lines of this stretch are no longer stored.</Empty> : (
            <ol className="incident-lines">
              {events.map((e, i) => (
                <li key={i} className={`ev-${e.status}`}>
                  <span className="mono small muted nowrap">{new Date(e.timestamp).toLocaleTimeString()}</span>
                  <div className="grow">
                    <div className="mono small ev-msg" title={e.template}>{e.message}</div>
                    {e.status !== "expected" && (
                      <div className="small">
                        <span className={`badge ${e.status === "unseen" ? "warn" : "bad"}`}>{MARK[e.status]}</span>
                        {e.status === "unexpected" && e.expected.length > 0 && <span className="muted"> expected here: {e.expected.map((x, j) => (
                          <span key={x.template_id}>{j > 0 && " or "}<span className="mono" style={{ color: "var(--ink)" }} title={x.template}>{x.words}</span></span>))}</span>}
                        {e.status === "unseen" && <span className="muted"> a kind of line this model never learned</span>}
                      </div>
                    )}
                  </div>
                </li>
              ))}
            </ol>
          )}
          <div className="small" style={{ marginTop: 8 }}><Link to={`/logs?host=${encodeURIComponent(inc.host)}&service=${encodeURIComponent(inc.service)}`}>Open this service in the Log Explorer</Link></div>
        </Card>
        <Card title="Elsewhere at the same time" hint="Warnings and errors other services logged from a minute before to a minute after">
          {inc.around.length === 0 ? <Empty title="Quiet">No other service logged warnings or errors then.</Empty> : (
            <ol className="incident-lines">
              {inc.around.map((a, i) => (
                <li key={i}>
                  <span className="mono small muted nowrap">{new Date(a.timestamp).toLocaleTimeString()}</span>
                  <span className={`badge ${a.level === "error" ? "bad" : "warn"}`}>{a.service || a.host}</span>
                  <div className="grow mono small ev-msg">{a.message}</div>
                </li>
              ))}
            </ol>
          )}
        </Card>
      </div>
    </>
  );
}
