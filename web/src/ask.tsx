// Ask: a question becomes a plan of read-only queries; each round is shown (what it reads and the
// exact SQL or pipeline) and runs only once approved, unless approval is switched off in Settings.
// The answer cites the lines it rests on (L3, T12): hover a cite to light its line up, click to go to it.
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { api, useAction, usePoll } from "./api";
import { Card, CopyBlock, Empty, Switch, TimeAgo, time } from "./ui";

export interface AskQuery {
  id: string; tool: string; title: string; why: string; compiled: string; engine: string;
  status: "pending" | "approved" | "rejected" | "done" | "failed"; error?: string | null; result_summary?: string | null;
}
interface Evidence { ref: string; timestamp: string; service?: string | null; host?: string | null; level?: string | null; template_id?: number | null; message: string; why: string; score?: number | null }
interface Candidate { template_id: number; service: string | null; count: number; first: string; template: string; example: string; score: number; anomaly: number; rarity: number; proximity: number; why: string; new: boolean; earliest_in_chain?: boolean }
interface Cause { cause: string; confidence: string; reasoning: string; cites: string[] }
interface Moment { at: string; what: string; cites: string[] }
interface Answer { summary: string; timeline: Moment[]; hypotheses: Cause[]; leads: Cause[]; insufficient_evidence: boolean; missing: string; next_steps: string[]; dropped: { section: string; text?: string | null }[]; by: "llm" | "rules"; note: string; llm_error?: string }
interface Anchor { kind: string; at: string; service?: string | null; host?: string | null; why: string }
interface Hint { ref: string; field: string; value: string; errors: number; error_share: number; other_share: number; ratio: number }
interface Series { ref: string; service: string | null; what: string; start: string; step_seconds: number; counts: number[] }
interface AiStep {
  step: string; schema: string; status: "running" | "done" | "failed"; started_at: string; finished_at?: string; seconds?: number;
  messages: { at: string; text: string }[]; prompt: string; reply: string | null; outcome: string; timeout_seconds?: number;
}
export interface AskResult {
  kind: "general" | "clarify" | "lines" | "table" | "report";
  answer?: Answer | string; note?: string; message?: string;
  lines?: { log_id: string | null; timestamp: string; host: string | null; service: string | null; level: string | null; template_id: number | null; message: string }[];
  templates?: { template_id: number; service: string | null; count: number; first: string; last: string; text: string }[];
  columns?: string[]; rows?: unknown[][]; compiled?: string;
  anchor?: Anchor; evidence?: Evidence[]; candidates?: Candidate[]; notes?: string[]; groups?: Record<string, string[]>; at?: string;
  hints?: Hint[]; series?: Series[];
}
export interface AskPlan {
  id: string; question: string; via: string; notes: string[]; spec: Record<string, unknown>; require_approval: boolean;
  status: "waiting_approval" | "approved" | "running" | "done" | "failed" | "rejected" | "expired";
  created_at: string; expires_at?: string; rounds: { n: number; purpose: string; queries: AskQuery[] }[]; result: AskResult | null;
  llm_steps?: AiStep[]; error?: string;
}

const ANCHOR: Record<string, string> = {
  user: "the time you gave", silence: "a service went quiet", shutdown: "a shutdown message", logbert: "repeated LogBERT alerts",
  new_error: "an error not seen the day before", range: "the whole range",
};
const INTENT: Record<string, string> = {
  general: "General question", keyword: "Search", event: "Find an event", semantic: "Search by meaning",
  rca: "Root cause", summary: "Summary",
};
const EXAMPLES = ["why did checkout fail in the last hour?", "errors from postgres in the last 15 minutes",
  "lines per service today", "anything about running out of memory"];

/** Which evidence ref is lit up (hovering a cite), shared by the answer and the evidence list. */
const Lit = createContext<{ lit: string | null; setLit: (r: string | null) => void; goTo: (r: string) => void }>(
  { lit: null, setLit: () => {}, goTo: () => {} });

function Cites({ cites }: { cites: string[] }) {
  const { setLit, goTo } = useContext(Lit);
  if (!cites.length) return null;
  return (
    <span className="ask-cites">
      {cites.map((c) => (
        <button key={c} type="button" className="ask-cite" title={`Show ${c} in the evidence`}
          onMouseEnter={() => setLit(c)} onMouseLeave={() => setLit(null)} onFocus={() => setLit(c)} onBlur={() => setLit(null)}
          onClick={() => goTo(c)}>{c}</button>
      ))}
    </span>
  );
}

/** What the question was read as, in words: one chip per thing the spec asks for. */
function SpecChips({ spec, via }: { spec: Record<string, unknown>; via: string }) {
  const chips: [string, string][] = [];
  const list = (k: string) => (spec[k] as unknown[] | undefined) ?? [];
  if (spec.start && spec.end) chips.push(["time", `${time(String(spec.start))} – ${time(String(spec.end))}`]);
  else if (spec.anchor_time) chips.push(["around", time(String(spec.anchor_time))]);
  else if (spec.intent !== "general") chips.push(["time", `last ${fmtMinutes(Number(spec.minutes ?? 60))}`]);
  for (const [k, label] of [["groups", "group"], ["services", "service"], ["hosts", "host"]] as const)
    if (list(k).length) chips.push([label, list(k).join(", ")]);
  if (spec.min_level) chips.push(["level", `${spec.min_level} and above`]);
  if (list("keywords").length) chips.push(["contains", list("keywords").map((k) => `"${k}"`).join(" and ")]);
  if (list("template_ids").length) chips.push(["template", list("template_ids").map((t) => `T${t}`).join(", ")]);
  const agg = spec.aggregate as { fn: string; field?: string | null; group_by: string[]; over_time: boolean } | null | undefined;
  if (agg) chips.push(["measure", `${agg.fn}${agg.field ? ` of ${agg.field}` : ""}${agg.group_by.length ? ` per ${agg.group_by.join(", ")}` : ""}${agg.over_time ? ", over time" : ""}`]);
  if (spec.semantic_query) chips.push(["about", String(spec.semantic_query)]);
  return (
    <div className="ask-chips">
      <span className="ask-intent">{INTENT[String(spec.intent)] ?? String(spec.intent)}</span>
      {chips.map(([k, v]) => <span key={k + v} className="ask-chip"><span className="muted">{k}</span> {v}</span>)}
      <span className="muted small">read {via === "fast" ? "by rules, no LLM" : via === "llm" ? "by the LLM" : `(${via})`}</span>
    </div>
  );
}

function fmtMinutes(m: number) {
  return m % 1440 === 0 ? `${m / 1440} day${m === 1440 ? "" : "s"}` : m % 60 === 0 ? `${m / 60} h` : `${m} min`;
}

/** Where the plan is: read → approve → run → answer. */
function Steps({ plan }: { plan: AskPlan }) {
  const writing = plan.status === "running" && plan.llm_steps?.at(-1)?.status === "running" && plan.llm_steps.at(-1)?.schema === "AnswerOut";
  const at = writing ? 3 : { waiting_approval: 1, approved: 2, running: 2, done: 3, failed: 2, rejected: 1, expired: 1 }[plan.status];
  const broke = ["failed", "rejected", "expired"].includes(plan.status);
  const names = ["Question read", plan.require_approval ? "Your approval" : "Approval off", "Read-only queries", "Answer"];
  return (
    <ol className="ask-steps" aria-label="Progress">
      {names.map((n, i) => {
        const state = i < at || plan.status === "done" ? "done" : i === at ? (broke ? "stopped" : "now") : "next";
        const busy = state === "now" && (plan.status === "running" || plan.status === "approved");
        return <li key={n} className={state} aria-current={state === "now" ? "step" : undefined}>
          {busy ? <span className="spin" aria-hidden="true" /> : <span className="dot" />}{n}</li>;
      })}
    </ol>
  );
}

/** One round: what each read is for and the exact text that will run. Pending ones can be ticked. */
function RoundView({ plan, round, waiting }: { plan: AskPlan; round: AskPlan["rounds"][number]; waiting: boolean }) {
  const [picked, setPicked] = useState<Set<string>>(() => new Set(round.queries.map((q) => q.id)));
  const [open, setOpen] = useState<Set<string>>(new Set());
  const approve = useAction(() => api(`/api/ask/${plan.id}/approve`, { method: "POST", json: { queries: [...picked] } }));
  const reject = useAction(() => api(`/api/ask/${plan.id}/reject`, { method: "POST" }));
  const running = ["approved", "running"].includes(plan.status) && round.queries.some((q) => q.status === "approved");
  const done = round.queries.filter((q) => q.status === "done").length;
  const flip = (set: Set<string>, id: string) => { const s = new Set(set); if (s.has(id)) s.delete(id); else s.add(id); return s; };
  const body = (
    <>
      <ul className="ask-queries">
        {round.queries.map((q) => (
          <li key={q.id} className={`q-${q.status}`}>
            <div className="ask-q-head">
              {waiting ? <input type="checkbox" aria-label={`Run ${q.title}`} checked={picked.has(q.id)} onChange={() => setPicked(flip(picked, q.id))} />
                : <span className={`ask-q-dot q-${q.status}`} aria-hidden="true" />}
              <div className="grow">
                <div className="ask-q-title">{q.title}</div>
                <div className="muted small">{q.why}</div>
              </div>
              <span className="ask-q-meta">
                {q.result_summary && <span className="small">{q.result_summary}</span>}
                <span className="badge" title={q.engine === "local" ? "Reads the server's own state, no log data" : "Read-only, on the log store"}>
                  {q.engine === "local" ? "server state" : q.engine}</span>
                <span className={`badge ${STATUS_BADGE[q.status]}`}>{QUERY_STATUS[q.status]}</span>
                <button type="button" className="btn small ghost" aria-expanded={open.has(q.id)} onClick={() => setOpen(flip(open, q.id))}>
                  {open.has(q.id) ? "Hide query" : "Show query"}</button>
              </span>
            </div>
            {open.has(q.id) && <div className="ask-q-code"><CopyBlock text={q.compiled} /></div>}
            {q.error && <div className="error-text small">{q.error}</div>}
          </li>
        ))}
      </ul>
      {waiting && (
        <div className="ask-turn" role="region" aria-label="Approve this round">
          <div className="grow">
            <strong>Your turn.</strong> {picked.size === round.queries.length ? `Run these ${round.queries.length} queries?` : `Run ${picked.size} of ${round.queries.length}? The others are skipped.`}
            <div className="muted small">All read-only. Open a query to check exactly what runs.
              {plan.expires_at && <> <Until iso={plan.expires_at} /></>}</div>
          </div>
          <button className="btn ghost" disabled={reject.isPending} onClick={() => reject.mutate()}>Stop here</button>
          <button className="btn primary" disabled={!picked.size || approve.isPending} onClick={() => approve.mutate()}>
            {approve.isPending ? <><span className="spin" aria-hidden="true" />Starting…</> : picked.size === round.queries.length ? "Run all" : `Run ${picked.size}`}</button>
          {(approve.error || reject.error) && <div className="error-text small" style={{ flexBasis: "100%" }}>{(approve.error ?? reject.error)!.message}</div>}
        </div>
      )}
    </>
  );
  const head = (
    <span className="ask-round-head">
      <span className="ask-round-n">Round {round.n}</span>
      <span className="grow">{round.purpose}</span>
      <span className="muted small">{running ? `running, ${done} of ${round.queries.length} done` : waiting ? "waiting for you" : summarize(round.queries)}</span>
    </span>
  );
  return (
    <section className={`ask-round${waiting ? " is-waiting" : ""}${running ? " is-running" : ""}`}>
      {waiting || running ? <>{head}{running && <div className="ask-progress" aria-hidden="true" />}{body}</>
        : <details><summary>{head}</summary>{body}</details>}
    </section>
  );
}

/** "Expires in 14 min", counting down. */
function Until({ iso }: { iso: string }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const t = setInterval(() => setNow(Date.now()), 15_000); return () => clearInterval(t); }, []);
  const min = Math.round((Date.parse(iso) - now) / 60_000);
  return <>{min > 0 ? `Expires in ${min} min.` : "Expires in under a minute."}</>;
}

const SLOW = 60;  // seconds: past this an LLM call is slower than usual, and the UI says so

function useSecondsSince(since: string) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const t = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(t); }, []);
  return Math.max(0, Math.round((now - Date.parse(since)) / 1000));
}

function duration(s: number) {
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`;
}

/** Seconds since ``since``, ticking: "12 s", "1 min 04 s". */
export function Elapsed({ since }: { since: string }) {
  return <span className="num">{duration(useSecondsSince(since))}</span>;
}

/** A running LLM call's clock, and once it's slow, how long until it's given up on. */
function Waiting({ since, limit }: { since: string; limit?: number }) {
  const s = useSecondsSince(since);
  const slow = s >= SLOW;
  return (
    <span className={slow ? "ask-slow" : "muted"}>
      <span className="num">{duration(s)}</span>
      {slow && <> · slower than usual{limit ? `; stops waiting in ${duration(Math.max(0, Math.round(limit - s)))}` : ""}</>}
    </span>
  );
}

/** What the LLM is doing for this plan, as it happens: each call, the adapter's own steps, the
 *  prompt it got and what it answered. Open while a call runs; folds away once it's done. */
function AiActivity({ steps }: { steps: AiStep[] }) {
  const running = steps.find((s) => s.status === "running");
  const [open, setOpen] = useState(!!running);
  const runningSince = running?.started_at;
  useEffect(() => { if (runningSince) setOpen(true); }, [runningSince]);  // a new call opens it again
  const total = steps.reduce((a, s) => a + (s.seconds ?? 0), 0);
  const failed = steps.filter((s) => s.status === "failed").length;
  return (
    <details className={`ask-ai${running ? " is-running" : ""}`} open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>
        {running ? <span className="spin" aria-hidden="true" /> : <span className="ask-ai-mark" aria-hidden="true">AI</span>}
        <span className="grow">{running ? <><strong>{running.step}…</strong> <Waiting since={running.started_at} limit={running.timeout_seconds} /></>
          : <>LLM activity <span className="muted">· {steps.length} call{steps.length === 1 ? "" : "s"}, {Math.round(total)} s</span>
            {failed > 0 && <span className="badge bad" style={{ marginLeft: 6 }}>{failed} failed</span>}</>}</span>
        <span className="muted small">{open ? "Hide" : "Show"}</span>
      </summary>
      <ol className="ask-ai-steps">
        {steps.map((s, i) => <AiStepView key={i} s={s} />)}
      </ol>
    </details>
  );
}

function AiStepView({ s }: { s: AiStep }) {
  const [show, setShow] = useState<"prompt" | "reply" | null>(null);
  return (
    <li className={`ai-${s.status}`}>
      <div className="ask-ai-head">
        {s.status === "running" ? <span className="spin" aria-hidden="true" /> : <span className={`ask-q-dot q-${s.status === "done" ? "done" : "failed"}`} aria-hidden="true" />}
        <strong className="grow">{s.step}</strong>
        <span className="small">{s.status === "running" ? <Waiting since={s.started_at} limit={s.timeout_seconds} /> : <span className="muted">{s.seconds ?? 0} s</span>}</span>
      </div>
      {s.outcome && <div className={`small ask-ai-outcome${s.status === "failed" ? " error-text" : ""}`}>{s.outcome}</div>}
      {s.messages.length > 0 && (
        <ul className="ask-ai-log">{s.messages.map((m, i) => (
          <li key={i}><span className="ask-time">+{Math.max(0, Math.round((Date.parse(m.at) - Date.parse(s.started_at)) / 1000))} s</span> {m.text}</li>))}
        </ul>
      )}
      <div className="row" style={{ gap: 6 }}>
        <button type="button" className="btn small ghost" aria-expanded={show === "prompt"} onClick={() => setShow(show === "prompt" ? null : "prompt")}>
          {show === "prompt" ? "Hide prompt" : `Prompt (${Math.round(s.prompt.length / 4)} tokens)`}</button>
        {s.reply && <button type="button" className="btn small ghost" aria-expanded={show === "reply"} onClick={() => setShow(show === "reply" ? null : "reply")}>
          {show === "reply" ? "Hide reply" : "Reply"}</button>}
      </div>
      {show && <div className="ask-q-code"><CopyBlock text={show === "prompt" ? s.prompt : s.reply ?? ""} /></div>}
    </li>
  );
}

const QUERY_STATUS: Record<AskQuery["status"], string> = { pending: "to approve", approved: "running", rejected: "skipped", done: "done", failed: "failed" };
const STATUS_BADGE: Record<AskQuery["status"], string> = { pending: "warn", approved: "info", rejected: "", done: "good", failed: "bad" };

function summarize(qs: AskQuery[]) {
  const n = (s: AskQuery["status"]) => qs.filter((q) => q.status === s).length;
  return [n("done") && `${n("done")} done`, n("rejected") && `${n("rejected")} skipped`, n("failed") && `${n("failed")} failed`]
    .filter(Boolean).join(", ") || `${qs.length} queries`;
}

function SpecEditor({ plan, onClose }: { plan: AskPlan; onClose: () => void }) {
  const [text, setText] = useState(() => JSON.stringify(plan.spec, null, 2));
  const [bad, setBad] = useState("");
  const edit = useAction((spec: unknown) => api(`/api/ask/${plan.id}/edit`, { method: "POST", json: { spec } }));
  return (
    <form className="stack ask-spec" style={{ gap: 8 }} onSubmit={(e) => {
      e.preventDefault();
      try { edit.mutate(JSON.parse(text), { onSuccess: onClose }); setBad(""); } catch { setBad("That isn't valid JSON."); }
    }}>
      <textarea rows={12} value={text} spellCheck={false} aria-label="Query spec" onChange={(e) => setText(e.target.value)} />
      <div className="row">
        <button className="btn primary small" disabled={edit.isPending}>Plan again with this</button>
        <button type="button" className="btn ghost small" onClick={onClose}>Cancel</button>
        <span className="muted small">The new queries wait for your approval, whatever the setting.</span>
        {(bad || edit.error) && <span className="error-text small">{bad || edit.error!.message}</span>}
      </div>
    </form>
  );
}

/** A time the LLM copied from a line, in local time; one without a zone is UTC, like the bundle. */
function clock(at: string) {
  const iso = /^\d{4}-\d\d-\d\dT[\d:.]+$/.test(at) ? `${at}Z` : at;
  return at && !Number.isNaN(Date.parse(iso)) ? new Date(iso).toLocaleTimeString() : at;
}

function LevelPill({ level }: { level?: string | null }) {
  return level ? <span className={`lvl lvl-${level}`}>{level}</span> : null;
}

function LinesList({ lines, cited }: { lines: { key: string; ref?: string; timestamp: string; service?: string | null; host?: string | null; level?: string | null; message: string; why?: string }[]; cited?: boolean }) {
  const { lit } = useContext(Lit);
  return (
    <ol className="ask-lines">
      {lines.map((l) => (
        <li key={l.key} id={l.ref ? `ev-${l.ref}` : undefined} className={l.ref && lit === l.ref ? "lit" : ""}>
          {cited && <span className="ask-ref">{l.ref}</span>}
          <time className="ask-time" dateTime={l.timestamp}>{new Date(l.timestamp).toLocaleTimeString()}</time>
          <span className="ask-svc">{l.service || l.host || "–"}</span>
          <LevelPill level={l.level} />
          <div className="grow">
            <div className="ask-msg">{l.message}</div>
            {l.why && <div className="muted small">{l.why}</div>}
          </div>
        </li>
      ))}
    </ol>
  );
}

function Table({ columns, rows }: { columns: string[]; rows: unknown[][] }) {
  return (
    <div className="table-wrap"><table className="table">
      <thead><tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j} className={typeof v === "number" ? "num" : "mono small"}>{String(v)}</td>)}</tr>)}</tbody>
    </table></div>
  );
}

const CONFIDENCE: Record<string, string> = { high: "bad", medium: "warn", low: "", ranked: "info" };

export function ReportView({ r }: { r: AskResult }) {
  const a = typeof r.answer === "object" ? r.answer : undefined;
  const causes = a?.hypotheses.length ? a.hypotheses : a?.leads ?? [];
  const [tab, setTab] = useState<"answer" | "evidence" | "ranking">("answer");
  const [lit, setLit] = useState<string | null>(null);
  const evidence = r.evidence ?? [];
  const hints = r.hints ?? [];
  const series = r.series ?? [];
  const goTo = (ref: string) => {
    const line = ref.startsWith("T") ? evidence.find((e) => `T${e.template_id}` === ref)?.ref : ref;
    setTab("evidence");
    if (line) {
      setLit(line);
      setTimeout(() => document.getElementById(`ev-${line}`)?.scrollIntoView({ block: "center", behavior: "smooth" }), 30);
    }
  };
  return (
    <Lit.Provider value={{ lit, setLit, goTo }}>
      <div className="ask-report">
        <div className="ask-verdict">
          {a && <p className="ask-summary">{a.summary}</p>}
          {r.anchor && (
            <p className="ask-anchor">
              <span className="muted">{r.anchor.kind === "range" ? "Covers" : "Broke at"}</span>{" "}
              <strong>{r.anchor.kind === "range" ? "the whole range" : time(r.anchor.at)}</strong>
              <span className="muted"> · found by {ANCHOR[r.anchor.kind] ?? r.anchor.kind}:</span> {r.anchor.why}
            </p>
          )}
          {a && <span className={`badge ${a.by === "llm" ? "info" : ""}`}>{a.by === "llm" ? "Written by the LLM, checked against the evidence" : "Ranked by rules (no LLM)"}</span>}
        </div>
        {a?.llm_error && (
          <div className="callout bad"><div><strong>The LLM failed, so this answer is the rules' ranking.</strong> {a.llm_error}
            <div className="muted small">The evidence below is complete; only the written explanation is missing. Ask again to retry the LLM.</div></div></div>
        )}
        {a?.insufficient_evidence && !a.llm_error && <div className="callout warn"><div><strong>Not enough evidence to name a cause.</strong> {a.missing}</div></div>}
        <div className="tabs" role="tablist">
          {(["answer", "evidence", "ranking"] as const).map((t) => (
            <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
              {t === "answer" ? "Answer" : t === "evidence" ? `Evidence (${evidence.length + hints.length + series.length})` : `Ranking (${r.candidates?.length ?? 0})`}
            </button>
          ))}
        </div>
        {tab === "answer" && a && (
          <div className="ask-answer">
            {causes.length > 0 && (
              <section>
                <h3>{a.hypotheses.length ? "Likely causes" : "Leads"}{!a.hypotheses.length && <span className="muted small"> the top of the ranking, not a diagnosis</span>}</h3>
                <ol className="ask-causes">
                  {causes.map((h, i) => (
                    <li key={i}>
                      <div className="ask-cause-head">
                        <span className="ask-cause-n">{i + 1}</span>
                        <strong className="grow">{h.cause}</strong>
                        {h.confidence !== "ranked" && <span className={`badge ${CONFIDENCE[h.confidence] ?? ""}`}>{h.confidence}</span>}
                      </div>
                      {h.reasoning && <p className="ask-reason">{h.reasoning}</p>}
                      <Cites cites={h.cites} />
                    </li>
                  ))}
                </ol>
              </section>
            )}
            {a.hypotheses.length > 0 && a.insufficient_evidence && a.leads.length > 0 && (
              <section>
                <h3>Leads from the ranking<span className="muted small"> the rules' own top picks, for when the answer above is unsure</span></h3>
                <ol className="ask-causes">
                  {a.leads.map((h, i) => (
                    <li key={i}>
                      <div className="ask-cause-head"><span className="ask-cause-n">{i + 1}</span><strong className="grow">{h.cause}</strong></div>
                      {h.reasoning && <p className="ask-reason">{h.reasoning}</p>}
                      <Cites cites={h.cites} />
                    </li>
                  ))}
                </ol>
              </section>
            )}
            {a.timeline.length > 0 && (
              <section>
                <h3>Timeline</h3>
                <ol className="ask-timeline">
                  {a.timeline.map((t, i) => (
                    <li key={i}><time className="ask-time">{clock(t.at)}</time>
                      <span className="grow">{t.what}</span><Cites cites={t.cites} /></li>
                  ))}
                </ol>
              </section>
            )}
            {a.next_steps.length > 0 && <section><h3>Next steps</h3><ol className="ask-next">{a.next_steps.map((s, i) => <li key={i}>{s}</li>)}</ol></section>}
            <p className="muted small" style={{ margin: 0 }}>
              {a.dropped.length > 0 && <>{a.dropped.length} claim{a.dropped.length === 1 ? " was" : "s were"} dropped for citing nothing in the evidence. </>}
              {!a.llm_error && a.note}{r.notes?.length ? ` ${r.notes.join(" · ")}` : ""}
            </p>
          </div>
        )}
        {tab === "evidence" && (evidence.length + hints.length + series.length
          ? <>
            {hints.length > 0 && <Hints hints={hints} />}
            {series.length > 0 && <Rates series={series} />}
            {evidence.length > 0 && <section className="stack" style={{ gap: 6 }}>
              <h3 style={{ margin: 0 }}>Lines <span className="muted small">everything the LLM saw, oldest first; L1 is the earliest</span></h3>
              <LinesList cited lines={evidence.map((e) => ({ key: e.ref, ...e }))} /></section>}
          </>
          : <Empty title="No evidence">Nothing stood out in the window.</Empty>)}
        {tab === "ranking" && <Ranking candidates={r.candidates ?? []} />}
      </div>
    </Lit.Provider>
  );
}

/** Where the errors concentrate: a value far more common on error lines than on the others. */
function Hints({ hints }: { hints: Hint[] }) {
  const { lit } = useContext(Lit);
  return (
    <section className="stack" style={{ gap: 6 }}>
      <h3 style={{ margin: 0 }}>Where the errors concentrate <span className="muted small">share of error/warn lines carrying the value, against the other lines</span></h3>
      <ul className="ask-hints">
        {hints.map((h) => (
          <li key={h.ref} id={`ev-${h.ref}`} className={lit === h.ref ? "lit" : ""}>
            <span className="ask-ref">{h.ref}</span>
            <span className="grow"><span className="mono">{h.field.replace(/^@/, "")}=<strong>{h.value}</strong></span>
              <span className="muted small"> on {h.errors} error lines, {h.ratio}× more often than elsewhere</span></span>
            <span className="ask-share" title={`${Math.round(h.error_share * 100)}% of error lines, ${Math.round(h.other_share * 100)}% of the others`}>
              <span className="bar err" style={{ width: `${h.error_share * 100}%` }} /><span className="bar oth" style={{ width: `${h.other_share * 100}%` }} />
            </span>
            <span className="num small">{Math.round(h.error_share * 100)}% vs {Math.round(h.other_share * 100)}%</span>
          </li>))}
      </ul>
    </section>
  );
}

/** Lines per step that a follow-up read counted, as a row of bars. */
function Rates({ series }: { series: Series[] }) {
  const { lit } = useContext(Lit);
  return (
    <section className="stack" style={{ gap: 6 }}>
      <h3 style={{ margin: 0 }}>Rates <span className="muted small">read by follow-ups</span></h3>
      <ul className="ask-hints">
        {series.map((s) => {
          const max = Math.max(...s.counts, 1);
          const w = 4;
          return (
            <li key={s.ref} id={`ev-${s.ref}`} className={lit === s.ref ? "lit" : ""}>
              <span className="ask-ref">{s.ref}</span>
              <span className="grow"><strong>{s.service ?? "–"}</strong> <span className="muted small">{s.what} per {s.step_seconds / 60} min from {time(s.start)}, peak {max}</span></span>
              <svg className="ask-spark" width={s.counts.length * w} height={22} role="img" aria-label={`${s.what} of ${s.service}: ${s.counts.join(", ")}`}>
                {s.counts.map((c, i) => <rect key={i} x={i * w} y={22 - Math.max(c ? 2 : 0, (c / max) * 22)} width={w - 1} height={Math.max(c ? 2 : 0, (c / max) * 22)} />)}
              </svg>
            </li>);
        })}
      </ul>
    </section>
  );
}

function Ranking({ candidates }: { candidates: Candidate[] }) {
  if (!candidates.length) return <Empty title="Nothing ranked">No template was seen in the window.</Empty>;
  const best = Math.max(...candidates.map((c) => c.score), 1e-9);
  return (
    <>
      <p className="muted small" style={{ margin: 0 }}>Score = anomaly × rarity × proximity. Each row says which factor set its value.</p>
      <div className="table-wrap"><table className="table ask-rank">
        <thead><tr><th>Template</th><th>Service</th><th className="right">Lines</th><th>Score</th><th className="right">Anomaly</th><th className="right">Rarity</th><th className="right">Proximity</th></tr></thead>
        <tbody>{candidates.map((c) => (
          <tr key={`${c.template_id}-${c.service}`}>
            <td><span className="mono small">T{c.template_id}</span> {c.earliest_in_chain && <span className="badge info">earliest in chain</span>} {c.new && <span className="badge warn">new</span>}
              <div className="ask-msg">{c.example.split("\n")[0].slice(0, 200)}</div><div className="muted small">{c.why}</div></td>
            <td>{c.service}</td><td className="num">{c.count}</td>
            <td><span className="ask-score"><span style={{ width: `${(c.score / best) * 100}%` }} /></span><span className="num small">{c.score.toFixed(3)}</span></td>
            <td className="num">{c.anomaly.toFixed(2)}</td><td className="num">{c.rarity.toFixed(2)}</td><td className="num">{c.proximity.toFixed(2)}</td>
          </tr>))}</tbody>
      </table></div>
    </>
  );
}

export function ResultBody({ r }: { r: AskResult }) {
  if (r.kind === "general") return <div className="ask-general"><p>{String(r.answer)}</p><span className="badge">{r.note}</span></div>;
  if (r.kind === "clarify") return <div className="callout warn"><div><strong>I need more to go on.</strong> {r.message}{r.note && <div className="muted small">{r.note}</div>}</div></div>;
  if (r.kind === "lines") return (
    <div className="stack" style={{ gap: 10 }}>
      {r.note && <p className="muted small" style={{ margin: 0 }}>{r.note}</p>}
      {r.lines?.length ? <LinesList lines={r.lines.map((l, i) => ({ key: l.log_id ?? String(i), ...l }))} /> : <Empty title="No matching lines">Try a longer time range or fewer filters.</Empty>}
      {!!r.templates?.length && <><h3>When each kind of line occurred</h3><Table columns={["template", "service", "lines", "first", "last"]}
        rows={r.templates.map((t) => [`T${t.template_id} ${t.text.slice(0, 80)}`, t.service, t.count, time(t.first), time(t.last)])} /></>}
    </div>
  );
  if (r.kind === "table") return r.rows?.length ? <Table columns={r.columns ?? []} rows={r.rows} /> : <Empty title="No rows">Nothing matched in that range.</Empty>;
  return <ReportView r={r} />;
}

const DONE_TEXT: Partial<Record<AskPlan["status"], ReactNode>> = {
  rejected: "You stopped this plan; nothing else ran.",
  expired: "This plan expired before it was approved. Ask again to get fresh queries.",
};

export function PlanView({ id }: { id: string }) {
  const { data: plan, error } = usePoll<AskPlan>(`/api/ask/${id}`, 1500);
  const [editing, setEditing] = useState(false);
  if (!plan) return error ? <div className="error-text">{error.message}</div> : <div className="ask-skeleton" aria-busy="true" />;
  const editable = plan.status === "waiting_approval" && plan.rounds.length === 1;
  const last = plan.rounds.length - 1;
  return (
    <article className="ask-plan">
      <header className="ask-plan-head">
        <h2 className="ask-question">{plan.question}</h2>
        <div className="spread">
          <SpecChips spec={plan.spec} via={plan.via} />
          {editable && !editing && <button type="button" className="btn small ghost" onClick={() => setEditing(true)}>Edit what it searches</button>}
        </div>
        {editing && <SpecEditor plan={plan} onClose={() => setEditing(false)} />}
        {plan.notes.length > 0 && <p className="muted small" style={{ margin: 0 }}>{plan.notes.join(" · ")}</p>}
        {plan.rounds.length > 0 && <Steps plan={plan} />}
      </header>
      {!!plan.llm_steps?.length && <AiActivity steps={plan.llm_steps} />}
      {plan.rounds.length > 0 && (
        <div className="ask-rounds">
          {plan.rounds.map((r, i) => <RoundView key={`${r.n}-${plan.status}`} plan={plan} round={r} waiting={i === last && plan.status === "waiting_approval"} />)}
        </div>
      )}
      {DONE_TEXT[plan.status] && <div className="callout"><div>{DONE_TEXT[plan.status]}</div></div>}
      {plan.status === "failed" && (
        <div className="callout bad" role="alert"><div><strong>This run stopped.</strong> {plan.error ?? "A query failed: see the round above."}
          <div className="muted small">Nothing else will run for it. Ask again to start over.</div></div></div>
      )}
      {plan.result && <ResultBody r={plan.result} />}
    </article>
  );
}

/** While the question is being read (by rules, or by the LLM, which can take a while through a
 *  web chat): the question, a running clock and what happens next. */
function Pending({ question, what = "Reading your question", limit }: { question: string; what?: string; limit?: number }) {
  const [since] = useState(() => new Date().toISOString());
  return (
    <article className="ask-plan ask-pending" aria-busy="true" aria-live="polite">
      <h2 className="ask-question">{question}</h2>
      <div className="ask-ai is-running ask-ai-static">
        <span className="spin" aria-hidden="true" />
        <span className="grow"><strong>{what}…</strong> <Waiting since={since} limit={limit} /></span>
      </div>
      <p className="muted small" style={{ margin: 0 }}>Plain questions are read by rules in a moment. Anything else goes to the LLM, which can take up to a minute;
        then you see the queries it plans before any of them runs.</p>
    </article>
  );
}

interface Recent { id: string; question: string; status: AskPlan["status"]; created_at: string; intent: string }

export function AskPanel({ initialPlan }: { initialPlan?: string }) {
  const [question, setQuestion] = useState("");
  const [current, setCurrent] = useState<string | undefined>(initialPlan);
  const recent = usePoll<{ plans: Recent[]; require_approval: boolean; llm_timeout_seconds: number }>("/api/ask", 10000).data;
  const serverRequires = recent?.require_approval ?? true;
  const [mine, setMine] = useState(true);  // only matters when the server lets queries run unapproved
  const ask = useAction((q: string) => api<AskPlan>("/api/ask", {
    method: "POST", json: { question: q, require_approval: serverRequires || mine, tz_offset: -new Date().getTimezoneOffset() } }));
  useEffect(() => { document.getElementById("ask-q")?.focus(); }, []);
  const submit = (q: string) => { setQuestion(q); ask.mutate(q, { onSuccess: (p) => setCurrent(p.id) }); };
  return (
    <div className="ask">
      <form className="ask-composer" onSubmit={(e) => { e.preventDefault(); if (question.trim()) submit(question); }}>
        <div className="ask-input">
          <input id="ask-q" type="search" value={question} onChange={(e) => setQuestion(e.target.value)} autoComplete="off"
            placeholder="Ask about your logs: why did checkout fail around 14:05?" aria-label="Question" />
          <button className="btn primary" disabled={!question.trim() || ask.isPending} aria-busy={ask.isPending}>
            {ask.isPending ? <><span className="spin" aria-hidden="true" />Reading…</> : "Ask"}</button>
        </div>
        <div className="spread">
          <Switch checked={serverRequires || mine} disabled={serverRequires} onChange={setMine} label="Approve each query before it runs"
            title={serverRequires ? "Required by the server (Settings → Ask)" : "The server lets queries run unapproved; keep this on to review them anyway"} />
          {serverRequires && <span className="muted small">Required in <Link to="/settings#ask">Settings</Link>. Every query is read-only.</span>}
        </div>
        {ask.error && <div className="callout warn"><div>{ask.error.message}</div></div>}
      </form>
      {ask.isPending && <Pending question={ask.variables ?? question} limit={recent?.llm_timeout_seconds} />}
      {!current && !ask.isPending && (
        <div className="ask-empty">
          <p className="muted small" style={{ margin: 0 }}>Plain questions are read by rules. Anything else goes to the LLM, which only fills in a search form, never SQL. Try:</p>
          <div className="row">{EXAMPLES.map((q) => <button key={q} type="button" className="ask-example" onClick={() => submit(q)}>{q}</button>)}</div>
        </div>
      )}
      {current && !ask.isPending && <PlanView key={current} id={current} />}
      {!!recent?.plans.length && (
        <details className="ask-recent" open={!current}>
          <summary>Recent questions ({recent.plans.length})</summary>
          <ul>{recent.plans.map((p) => (
            <li key={p.id} className={p.id === current ? "on" : ""}>
              <button type="button" onClick={() => setCurrent(p.id)}>
                <span className={`ask-q-dot q-${p.status === "done" ? "done" : p.status === "waiting_approval" ? "pending" : ["failed"].includes(p.status) ? "failed" : p.status === "approved" || p.status === "running" ? "approved" : "rejected"}`} aria-hidden="true" />
                <span className="grow">{p.question}</span>
                <span className="muted small">{INTENT[p.intent] ?? p.intent} · <TimeAgo iso={p.created_at} /></span>
              </button>
            </li>))}</ul>
        </details>
      )}
    </div>
  );
}

/** The incident page's "What happened": a root-cause Ask anchored at the incident's start. */
export function IncidentAsk({ incident, plan }: { incident: string; plan: AskPlan | null }) {
  const start = useAction(() => api<AskPlan>(`/api/incidents/${incident}/explain`, { method: "POST" }));
  const working = plan && ["approved", "running"].includes(plan.status);
  const busy = start.isPending || working || plan?.status === "waiting_approval";
  const label = start.isPending ? "Planning…" : working ? "Investigating…" : plan?.status === "waiting_approval" ? "Waiting for your approval"
    : plan ? "Investigate again" : "Investigate";
  return (
    <Card title="What happened" actions={
      <button className="btn small primary" disabled={!!busy} aria-busy={start.isPending || !!working} onClick={() => start.mutate()}>
        {(start.isPending || working) && <span className="spin" aria-hidden="true" />}{label}</button>}>
      {start.isPending ? <Pending question="What happened in this incident?" what="Planning the first queries" />
        : plan ? <PlanView id={plan.id} /> : (
        <p className="muted" style={{ margin: 0 }}>Plans read-only queries around the incident's start for your approval, ranks what changed
          against the day before, and explains it citing the lines. <Link to="/query?tab=ask">Ask anything else</Link>.</p>)}
      {start.error && <div className="error-text small">{start.error.message}</div>}
    </Card>
  );
}
