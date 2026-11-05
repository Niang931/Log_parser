import { Fragment, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, useAction, usePoll, type Job, type Maintenance } from "../api";
import { Card, Empty, JOB_NAMES, JobDetail, JobStatus, Loading, Meter, PageHead, Progress, TimeAgo, jobTarget, sourceLabel } from "../ui";

function duration(job: Job, now: number) {
  if (!job.started_at) return "–";
  const ms = (job.finished_at ? new Date(job.finished_at).getTime() : now) - new Date(job.started_at).getTime();
  const s = Math.max(0, Math.round(ms / 1000));
  return s < 60 ? `${s}s` : s < 3600 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

/** What to do about a failure, in words. */
function hint(error: string) {
  if (/windows of parsed logs|need \d+/.test(error)) return "Not enough parsed lines yet. It trains by itself once there are.";
  if (/server restarted/.test(error)) return "The server stopped while this ran. Retry it.";
  if (/no log sources/.test(error)) return "Connect an agent and track a file first.";
  return null;
}

function Actions({ job }: { job: Job }) {
  const cancel = useAction(() => api(`/api/jobs/${job.id}/cancel`, { method: "POST" }));
  const retry = useAction(() => api(`/api/jobs/${job.id}/retry`, { method: "POST" }));
  const err = cancel.error ?? retry.error;
  return (
    <span className="row" style={{ flexWrap: "nowrap", justifyContent: "flex-end" }} onClick={(e) => e.stopPropagation()}>
      {job.status === "queued" && <button className="btn small ghost" disabled={cancel.isPending} onClick={() => cancel.mutate()}>Cancel</button>}
      {(job.status === "failed" || job.status === "cancelled") && !["calibrate", "recalibrate"].includes(job.kind) &&
        <button className="btn small" disabled={retry.isPending || retry.isSuccess} onClick={() => retry.mutate()}>{retry.isSuccess ? "Queued" : "Retry"}</button>}
      {err && <span className="error-text small">{err.message}</span>}
    </span>
  );
}

function WaitingLane() {
  const { data } = usePoll<Maintenance>("/api/maintenance", 5000);
  if (!data || (!data.collecting.length && !data.waiting.length)) return null;
  return (
    <Card title="Waiting for data" hint="These start by themselves once there is enough. Nothing to do.">
      <div className="table-wrap">
        <table>
          <thead><tr><th>Starts</th><th>For</th><th style={{ width: "40%" }}>Collected</th></tr></thead>
          <tbody>
            {data.collecting.map((c) => (
              <tr key={c.id}>
                <td>Draft templates</td>
                <td>{sourceLabel(c.id)}</td>
                <td><Meter have={c.have} need={c.need} unit="lines" /></td>
              </tr>
            ))}
            {data.waiting.map((s) => (
              <tr key={s.id}>
                <td>Train LogBERT</td>
                <td><Link to="/streams">{s.name}</Link></td>
                <td>{s.data ? <Meter have={s.data.have} need={s.data.need} unit="windows" /> : <span className="muted">checking…</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

const FILTERS = { all: () => true, active: (j: Job) => j.status === "queued" || j.status === "running", failed: (j: Job) => j.status === "failed" };

export default function Jobs() {
  const { data, error } = usePoll<{ jobs: Job[] }>("/api/jobs", 1500);
  const [filter, setFilter] = useState<keyof typeof FILTERS>("all");
  const [open, setOpen] = useState<string | null>(null);
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const t = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(t); }, []);
  const jobs = data?.jobs.filter(FILTERS[filter]) ?? [];
  const count = (f: keyof typeof FILTERS) => data?.jobs.filter(FILTERS[f]).length ?? 0;

  return (
    <>
      <PageHead title="Jobs" sub="Template drafting, applying and model training run one at a time in the background." />
      <div className="stack">
        <WaitingLane />
        <Card title="History" actions={
          <div className="seg" role="group" aria-label="Show">
            {(Object.keys(FILTERS) as (keyof typeof FILTERS)[]).map((f) => (
              <button key={f} type="button" className={filter === f ? "on" : ""} aria-pressed={filter === f} onClick={() => setFilter(f)}>
                {f[0].toUpperCase() + f.slice(1)} {count(f)}
              </button>
            ))}
          </div>}>
          {!data ? <Loading error={error} /> : jobs.length === 0 ? <Empty title={filter === "all" ? "No jobs yet" : `No ${filter} jobs`} /> : (
            <div className="table-wrap">
              <table className="jobs">
                <thead><tr><th style={{ width: 120 }}>Status</th><th>Job</th><th>Target</th><th style={{ width: "22%" }}>Progress</th><th>Started</th><th className="right">Duration</th><th /></tr></thead>
                <tbody>
                  {jobs.map((j) => {
                    const expanded = open === j.id;
                    return (
                      <Fragment key={j.id}>
                        <tr className={`clickable ${expanded ? "sel" : ""}`} onClick={() => setOpen(expanded ? null : j.id)} tabIndex={0}
                          onKeyDown={(e) => e.key === "Enter" && setOpen(expanded ? null : j.id)} aria-expanded={expanded}>
                          <td><JobStatus job={j} /></td>
                          <td><strong>{JOB_NAMES[j.kind]}</strong></td>
                          <td className="small ellipsis" style={{ maxWidth: 260 }} title={jobTarget(j)}>{jobTarget(j) || "–"}</td>
                          <td>
                            {j.status === "running" || j.status === "queued" ? <Progress value={j.progress} /> : null}
                            <div className={`small ellipsis ${j.error ? "error-text" : "muted"}`} style={{ maxWidth: 320 }}>{j.error ?? j.message}</div>
                          </td>
                          <td className="small">{j.started_at ? <TimeAgo iso={j.started_at} /> : <span className="muted">queued <TimeAgo iso={j.created_at} /></span>}</td>
                          <td className="right num small">{duration(j, now)}</td>
                          <td><Actions job={j} /></td>
                        </tr>
                        {expanded && (
                          <tr className="job-detail">
                            <td colSpan={7}>
                              {j.error && (
                                <div className="callout bad" style={{ marginBottom: 8 }}>
                                  <div className="grow"><strong>{j.error}</strong>{hint(j.error) && <div className="small">{hint(j.error)}</div>}</div>
                                </div>
                              )}
                              {j.log?.length ? (
                                <ol className="job-log mono small">
                                  {j.log.map(([t, m], i) => <li key={i}><span className="muted">{new Date(t).toLocaleTimeString()}</span>{m}</li>)}
                                </ol>
                              ) : <div className="muted small">No steps recorded.</div>}
                              <JobDetail job={j} steps={false} />
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>
    </>
  );
}
