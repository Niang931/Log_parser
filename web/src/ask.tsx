// Ask: a question becomes a plan of read-only queries; each round is shown (what it reads and the
// exact SQL or pipeline) and runs only once approved, unless approval is switched off in Settings.
// The answer cites the lines it rests on (L3, T12); hover or click a cite to find its line.
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, useAction, usePoll } from "./api";
import { Card, Empty, Switch, TimeAgo, time } from "./ui";

export interface AskQuery {
  id: string; tool: string; title: string; why: string; compiled: string; engine: string;
  status: "pending" | "approved" | "rejected" | "done" | "failed"; error?: string | null; result_summary?: string | null;
}
interface Evidence { ref: string; timestamp: string; service?: string; host?: string; level?: string; template_id?: number | null; message: string; why: string; score?: number | null }
interface Candidate { template_id: number; service: string; count: number; first: string; template: string; example: string; score: number; anomaly: number; rarity: number; proximity: number; why: string; new: boolean; earliest_in_chain?: boolean }
interface Claim { cause?: string; what?: string; at?: string; confidence?: string; reasoning?: string; cites: string[] }
interface Answer { summary: string; timeline: Claim[]; hypotheses: Claim[]; leads?: Claim[]; insufficient_evidence: boolean; missing?: string; next_steps?: string[]; dropped?: { in: string; text?: string }[]; by?: string; note?: string }
export interface AskResult {
  kind: "general" | "clarify" | "lines" | "table" | "report";
  answer?: Answer | string; note?: string; message?: string;
  lines?: { log_id: string; timestamp: string; host: string; service: string; level: string; template_id: number; message: string }[];
  templates?: { template_id: number; service: string; count: number; first: string; last: string; text: string }[];
  columns?: string[]; rows?: unknown[][];
  anchor?: { kind: string; at: string; service?: string | null; host?: string | null; why: string } | null;
  evidence?: Evidence[]; candidates?: Candidate[]; notes?: string[]; at?: string;
}
export interface AskPlan {
  id: string; question: string; via: string; notes: string[]; spec: Record<string, unknown>; require_approval: boolean;
  status: "waiting_approval" | "approved" | "running" | "done" | "failed" | "rejected" | "expired";
  created_at: string; expires_at?: string; rounds: { n: number; purpose: string; queries: AskQuery[] }[]; result: AskResult | null;
}

const STATUS: Record<AskPlan["status"], [string, string]> = {
  waiting_approval: ["warn", "Waiting for your approval"], approved: ["info", "Approved, queued"], running: ["info", "Running"],
  done: ["ok", "Answered"], failed: ["bad", "Failed"], rejected: ["", "Rejected"], expired: ["", "Expired"],
};
const ANCHOR: Record<string, string> = {
  user: "the time you gave", silence: "a service went quiet", shutdown: "a shutdown message", logbert: "LogBERT alerts",
  new_error: "a new error", range: "the whole range",
};

function Cites({ cites }: { cites: string[] }) {
  return <>{cites.map((c) => <a key={c} href={`#ev-${c}`} className="badge cite mono" title={`Evidence ${c}`}>{c}</a>)}</>;
}

/** One round: what each read is for and the exact text that will run. Pending ones can be ticked. */
function RoundView({ plan, round, last }: { plan: AskPlan; round: AskPlan["rounds"][number]; last: boolean }) {
  const waiting = last && plan.status === "waiting_approval";
  const [picked, setPicked] = useState<Set<string>>(() => new Set(round.queries.map((q) => q.id)));
  const approve = useAction(() => api(`/api/ask/${plan.id}/approve`, { method: "POST", json: { queries: [...picked] } }));
  const reject = useAction(() => api(`/api/ask/${plan.id}/reject`, { method: "POST" }));
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div className="spread"><strong>Round {round.n}: {round.purpose}</strong>
        <span className="muted small">{round.queries.length} read{round.queries.length === 1 ? "" : "s"}, all read-only</span></div>
      {round.queries.map((q) => (
        <details key={q.id} className="ask-query" open={waiting}>
          <summary className="row" style={{ gap: 8 }}>
            {waiting && <input type="checkbox" aria-label={`Approve ${q.title}`} checked={picked.has(q.id)}
              onClick={(e) => e.stopPropagation()}
              onChange={(e) => { const s = new Set(picked); if (e.target.checked) s.add(q.id); else s.delete(q.id); setPicked(s); }} />}
            <span>{q.title}</span>
            <span className={`badge ${q.status === "done" ? "ok" : q.status === "failed" ? "bad" : q.status === "rejected" ? "" : "info"}`}>{q.status}</span>
            <span className="muted small">{q.engine === "local" ? "server state, no log data" : q.engine}</span>
            {q.result_summary && <span className="muted small">· {q.result_summary}</span>}
          </summary>
          <p className="muted small" style={{ margin: "4px 0" }}>{q.why}</p>
          <pre className="code-block mono small">{q.compiled}</pre>
          {q.error && <div className="error-text small">{q.error}</div>}
        </details>
      ))}
      {waiting && (
        <div className="row">
          <button className="btn primary" disabled={!picked.size || approve.isPending} onClick={() => approve.mutate()}>
            Run {picked.size === round.queries.length ? "all" : `${picked.size} of ${round.queries.length}`}</button>
          <button className="btn ghost" disabled={reject.isPending} onClick={() => reject.mutate()}>Reject</button>
          {plan.expires_at && <span className="muted small">Expires <TimeAgo iso={plan.expires_at} /></span>}
          {(approve.error || reject.error) && <span className="error-text small">{(approve.error ?? reject.error)!.message}</span>}
        </div>
      )}
    </div>
  );
}

function SpecEditor({ plan }: { plan: AskPlan }) {
  const [text, setText] = useState(() => JSON.stringify(plan.spec, null, 2));
  const edit = useAction(() => api(`/api/ask/${plan.id}/edit`, { method: "POST", json: { spec: JSON.parse(text) } }));
  const editable = plan.status === "waiting_approval" && plan.rounds.length === 1;
  return (
    <details>
      <summary className="small">Query spec ({plan.via === "fast" ? "read without the LLM" : plan.via === "llm" ? "from the LLM" : plan.via})</summary>
      {editable ? (
        <form className="stack" style={{ gap: 6, marginTop: 6 }} onSubmit={(e) => { e.preventDefault(); edit.mutate(); }}>
          <textarea rows={10} className="mono small" value={text} spellCheck={false} aria-label="Query spec" onChange={(e) => setText(e.target.value)} />
          <div className="row">
            <button className="btn small" disabled={edit.isPending}>Re-plan with this spec</button>
            <span className="muted small">The new queries wait for approval again.</span>
            {edit.error && <span className="error-text small">{edit.error.message}</span>}
          </div>
        </form>
      ) : <pre className="code-block mono small">{JSON.stringify(plan.spec, null, 2)}</pre>}
      {plan.notes.length > 0 && <p className="muted small">{plan.notes.join(" · ")}</p>}
    </details>
  );
}

function Lines({ lines }: { lines: NonNullable<AskResult["lines"]> }) {
  if (!lines.length) return <Empty title="No matching lines" />;
  return (
    <div className="table-wrap"><table className="table">
      <thead><tr><th>Time</th><th>Service</th><th>Level</th><th>Line</th></tr></thead>
      <tbody>{lines.map((l) => (
        <tr key={l.log_id}><td className="mono small nowrap">{time(l.timestamp)}</td><td>{l.service || l.host}</td>
          <td><span className={`badge ${l.level === "error" ? "bad" : l.level === "warn" ? "warn" : ""}`}>{l.level}</span></td>
          <td className="mono small">{l.message}</td></tr>))}</tbody>
    </table></div>
  );
}

function Table({ columns, rows }: { columns: string[]; rows: unknown[][] }) {
  return (
    <div className="table-wrap"><table className="table">
      <thead><tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j} className="mono small">{String(v)}</td>)}</tr>)}</tbody>
    </table></div>
  );
}

export function ReportView({ r }: { r: AskResult }) {
  const a = typeof r.answer === "object" ? r.answer : undefined;
  const claims = a?.hypotheses.length ? a.hypotheses : a?.leads ?? [];
  return (
    <div className="stack">
      {r.anchor && <p className="small" style={{ margin: 0 }}><span className="muted">Anchored on {ANCHOR[r.anchor.kind] ?? r.anchor.kind}:</span> {r.anchor.why} <span className="mono">({time(r.anchor.at)})</span></p>}
      {a && (
        <>
          <p style={{ margin: 0 }}>{a.summary}</p>
          {a.insufficient_evidence && <div className="callout warn">Not enough evidence to name a cause{a.missing ? `: ${a.missing}` : "."}</div>}
          {claims.length > 0 && (
            <div><h3>{a.hypotheses.length ? "Likely causes" : "Leads (ranked by score, not a diagnosis)"}</h3>
              <ol style={{ margin: 0, paddingLeft: 20 }}>{claims.map((h, i) => (
                <li key={i}><strong>{h.cause}</strong> {h.confidence && h.confidence !== "ranked" && <span className="badge">{h.confidence}</span>} <Cites cites={h.cites} />
                  {h.reasoning && <div className="muted small">{h.reasoning}</div>}</li>))}</ol></div>
          )}
          {a.timeline.length > 0 && (
            <div><h3>Timeline</h3>
              <ol className="plain" style={{ margin: 0, paddingLeft: 0, listStyle: "none" }}>{a.timeline.map((t, i) => (
                <li key={i} className="small"><span className="mono">{t.at ? time(t.at) : ""}</span> {t.what} <Cites cites={t.cites} /></li>))}</ol></div>
          )}
          {!!a.next_steps?.length && <div><h3>Next steps</h3><ol style={{ margin: 0, paddingLeft: 20 }}>{a.next_steps.map((s, i) => <li key={i}>{s}</li>)}</ol></div>}
          {!!a.dropped?.length && <p className="muted small">{a.dropped.length} claim{a.dropped.length === 1 ? "" : "s"} dropped for citing nothing in the evidence.</p>}
          {a.note && <p className="muted small">{a.note}</p>}
        </>
      )}
      {!!r.evidence?.length && (
        <details open={!a || claims.length === 0}>
          <summary><strong>Evidence ({r.evidence.length} lines)</strong> <span className="muted small">all the LLM saw</span></summary>
          <div className="table-wrap"><table className="table">
            <thead><tr><th>Ref</th><th>Time</th><th>Service</th><th>Line</th><th>Why it's here</th></tr></thead>
            <tbody>{r.evidence.map((e) => (
              <tr key={e.ref} id={`ev-${e.ref}`}><td className="mono">{e.ref}</td><td className="mono small nowrap">{time(e.timestamp)}</td>
                <td>{e.service || e.host}{e.template_id != null && <div className="muted small mono">T{e.template_id}</div>}</td>
                <td className="mono small">{e.message}</td><td className="small muted">{e.why}</td></tr>))}</tbody>
          </table></div>
        </details>
      )}
      {!!r.candidates?.length && (
        <details>
          <summary><strong>Ranking</strong> <span className="muted small">score = anomaly × rarity × proximity</span></summary>
          <div className="table-wrap"><table className="table">
            <thead><tr><th>Template</th><th>Service</th><th>Lines</th><th>Score</th><th>Anomaly</th><th>Rarity</th><th>Proximity</th><th>Why</th></tr></thead>
            <tbody>{r.candidates.map((c) => (
              <tr key={`${c.template_id}-${c.service}`} id={`ev-T${c.template_id}`}>
                <td className="mono small">T{c.template_id} {c.earliest_in_chain && <span className="badge info">earliest</span>}<div>{c.example.split("\n")[0].slice(0, 160)}</div></td>
                <td>{c.service}</td><td>{c.count}</td><td className="mono">{c.score.toFixed(3)}</td>
                <td className="mono">{c.anomaly.toFixed(2)}</td><td className="mono">{c.rarity.toFixed(2)}</td><td className="mono">{c.proximity.toFixed(2)}</td>
                <td className="small muted">{c.why}</td></tr>))}</tbody>
          </table></div>
        </details>
      )}
      {!!r.notes?.length && <p className="muted small">{r.notes.join(" · ")}</p>}
    </div>
  );
}

export function ResultBody({ r }: { r: AskResult }) {
  if (r.kind === "general") return <><p style={{ margin: 0 }}>{String(r.answer)}</p><p className="muted small">{r.note}</p></>;
  if (r.kind === "clarify") return <div className="callout warn">{r.message}</div>;
  if (r.kind === "lines") return <>{r.note && <p className="muted small">{r.note}</p>}<Lines lines={r.lines ?? []} />
    {!!r.templates?.length && <Table columns={["template", "service", "count", "first", "last"]}
      rows={r.templates.map((t) => [`T${t.template_id} ${t.text.slice(0, 80)}`, t.service, t.count, time(t.first), time(t.last)])} />}</>;
  if (r.kind === "table") return r.rows?.length ? <Table columns={r.columns ?? []} rows={r.rows} /> : <Empty title="No rows" />;
  return <ReportView r={r} />;
}

export function PlanView({ id }: { id: string }) {
  const { data: plan, error } = usePoll<AskPlan>(`/api/ask/${id}`, 1500);
  if (!plan) return error ? <div className="error-text">{error.message}</div> : <p className="muted">Loading…</p>;
  const [badge, label] = STATUS[plan.status];
  return (
    <div className="stack">
      <div className="spread"><span>{plan.question}</span><span className={`badge ${badge}`}>{label}</span></div>
      <SpecEditor key={plan.rounds.length} plan={plan} />
      {plan.rounds.map((r, i) => <RoundView key={`${r.n}-${plan.status}`} plan={plan} round={r} last={i === plan.rounds.length - 1} />)}
      {plan.result && <ResultBody r={plan.result} />}
    </div>
  );
}

export function AskPanel({ initialPlan }: { initialPlan?: string }) {
  const [question, setQuestion] = useState("");
  const [current, setCurrent] = useState<string | undefined>(initialPlan);
  const recent = usePoll<{ plans: { id: string; question: string; status: string; created_at: string; intent: string }[]; require_approval: boolean }>("/api/ask", 10000).data;
  const serverRequires = recent?.require_approval ?? true;
  const [mine, setMine] = useState(true);  // only matters when the server lets queries run unapproved
  const ask = useAction(() => api<AskPlan>("/api/ask", {
    method: "POST", json: { question, require_approval: serverRequires || mine, tz_offset: -new Date().getTimezoneOffset() } }));
  useEffect(() => { document.getElementById("ask-q")?.focus(); }, []);
  return (
    <div className="stack">
      <form className="stack" onSubmit={(e) => { e.preventDefault(); ask.mutate(undefined, { onSuccess: (p) => setCurrent(p.id) }); }}>
        <input id="ask-q" type="search" value={question} onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. why did checkout fail around 14:05?  ·  errors from postgres in the last hour  ·  lines per service today" aria-label="Question" />
        <div className="row">
          <button className="btn primary" disabled={!question.trim() || ask.isPending}>{ask.isPending ? "Planning…" : "Ask"}</button>
          <Switch checked={serverRequires || mine} disabled={serverRequires} onChange={setMine} label="Approve each query before it runs"
            title={serverRequires ? "Required by the server (Settings → Ask)" : "The server lets queries run unapproved; keep this on to review them anyway"} />
        </div>
        {ask.error && <div className="callout warn">{ask.error.message}</div>}
      </form>
      {current && <PlanView id={current} />}
      {!!recent?.plans.length && (
        <details><summary className="small">Recent questions</summary>
          <ul className="side-list">{recent.plans.map((p) => (
            <li key={p.id}><button type="button" onClick={() => setCurrent(p.id)}>{p.question}</button>
              <span className="muted small"> {p.status} · <TimeAgo iso={p.created_at} /></span></li>))}</ul>
        </details>
      )}
    </div>
  );
}

/** The incident page's "What happened": a root-cause Ask anchored at the incident's start. */
export function IncidentAsk({ incident, plan }: { incident: string; plan: AskPlan | null }) {
  const start = useAction(() => api<AskPlan>(`/api/incidents/${incident}/explain`, { method: "POST" }));
  const busy = start.isPending || (plan && ["approved", "running", "waiting_approval"].includes(plan.status));
  return (
    <Card title="What happened" actions={
      <button className="btn small" disabled={!!busy} onClick={() => start.mutate()}>{plan ? "Investigate again" : "Investigate with AI"}</button>}>
      {plan ? <PlanView id={plan.id} /> : (
        <p className="muted" style={{ margin: 0 }}>Plans read-only queries around the incident's start (shown for your approval), ranks
          what changed against the day before, and has the LLM explain it, citing the lines. <Link to="/query?tab=ask">Ask anything else</Link>.</p>)}
      {start.error && <div className="error-text small">{start.error.message}</div>}
    </Card>
  );
}
