// Template review: drafted templates wait here until a person approves, edits, merges or rejects
// them; only approved ones reach the parser. Also lists what the online learner added by itself.
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, useAction, usePoll, type AutoTemplate, type Batch, type Draft } from "../api";
import { Card, Empty, Loading, PageHead, TemplateText, TimeAgo, fmt, sourceLabel } from "../ui";

const SLOT = /<\*>|<VAR:[A-Z0-9]+>/g;
const slotCount = (tokens: string[]) => tokens.reduce((n, t) => n + (t.match(SLOT)?.length ?? 0), 0);

type Checked = { count: number; samples: string[]; checked?: number };

/** Template text + one field name per variable; re-checks against the sample as you type. */
function TemplateEditor({ tokens, keys, check, onSave, onCancel, saving, error }: {
  tokens: string[]; keys: string[]; saving: boolean; error?: Error | null;
  check?: (tokens: string[], keys: string[]) => Promise<Checked>;
  onSave: (tokens: string[], keys: string[]) => void; onCancel: () => void;
}) {
  const [text, setText] = useState(tokens.join(" "));
  const [names, setNames] = useState(keys);
  const [result, setResult] = useState<Checked | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const toks = text.trim().split(/\s+/).filter(Boolean);
  const slots = slotCount(toks);
  const fields = Array.from({ length: slots }, (_, i) => names[i] ?? `var_${i + 1}`);

  useEffect(() => {
    if (!check) return;
    const t = setTimeout(() => {
      check(toks, fields).then((r) => { setResult(r); setProblem(null); }, (e: Error) => { setResult(null); setProblem(e.message); });
    }, 350);
    return () => clearTimeout(t);
  }, [text, fields.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <form className="editor stack" onSubmit={(e) => { e.preventDefault(); onSave(toks, fields); }}>
      <label className="field">
        Template <span className="muted small">use <code>&lt;*&gt;</code> for a variable part; words are separated by spaces</span>
        <textarea className="mono" rows={2} value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} />
      </label>
      {slots > 0 && (
        <div className="field">
          Field names <span className="muted small">lower_snake_case, one per variable, in order</span>
          <div className="row">
            {fields.map((f, i) => (
              <input key={i} aria-label={`Field ${i + 1}`} className="mono" value={f} style={{ width: 140 }}
                onChange={(e) => setNames(Object.assign([...fields], { [i]: e.target.value }))} />
            ))}
          </div>
        </div>
      )}
      {check && (
        <div className={`small ${problem ? "error-text" : "muted"}`} role="status">
          {problem ?? (result ? `Matches ${result.count.toLocaleString()} of ${(result.checked ?? 0).toLocaleString()} recent lines` : "Checking…")}
        </div>
      )}
      {result && result.samples.length > 0 && (
        <div className="samples mono">{result.samples.slice(0, 3).map((s, i) => <div key={i}>{s}</div>)}</div>
      )}
      {error && <div className="error-text">{error.message}</div>}
      <div className="row">
        <button className="btn small primary" disabled={saving || !!problem}>Save</button>
        <button type="button" className="btn small ghost" onClick={onCancel}>Cancel</button>
      </div>
    </form>
  );
}

function DraftRow({ d, total, selected, onSelect, batch }: {
  d: Draft; total: number; selected: boolean; onSelect: (on: boolean) => void; batch: string;
}) {
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const save = useAction((body: Partial<Draft>) => api<Draft>(`/api/drafts/${d.id}`, { method: "PUT", json: body }));
  const decide = (status: Draft["status"]) => save.mutate({ status: d.status === status ? "pending" : status });
  return (
    <li className={`draft draft-${d.status}`}>
      <div className="draft-head">
        <input type="checkbox" aria-label="Select for merging" checked={selected} onChange={(e) => onSelect(e.target.checked)} />
        <button type="button" className="draft-text" onClick={() => setOpen(!open)} aria-expanded={open}>
          <TemplateText tokens={d.tokens} keys={d.keys} />
          {d.edited && <span className="badge" style={{ marginLeft: 6 }}>edited</span>}
        </button>
        <span className="draft-count num" title={`${d.count} of ${total} recent lines`}>
          {fmt(d.count)}<span className="muted small"> · {total ? ((d.count / total) * 100).toFixed(1) : 0}%</span>
        </span>
        <div className="seg draft-decide" role="group" aria-label="Decision">
          <button type="button" className={d.status === "approved" ? "on good" : ""} aria-pressed={d.status === "approved"} onClick={() => decide("approved")} disabled={save.isPending}>Approve</button>
          <button type="button" className={d.status === "rejected" ? "on bad" : ""} aria-pressed={d.status === "rejected"} onClick={() => decide("rejected")} disabled={save.isPending}>Reject</button>
        </div>
        <button type="button" className="btn small ghost" onClick={() => { setOpen(true); setEditing(true); }}>Edit</button>
      </div>
      {open && (
        <div className="draft-body">
          {editing ? (
            <TemplateEditor tokens={d.tokens} keys={d.keys} saving={save.isPending} error={save.error}
              check={(tokens, keys) => api<Checked>(`/api/review/${batch}/check`, { method: "POST", json: { tokens, keys } })}
              onCancel={() => setEditing(false)}
              onSave={(tokens, keys) => save.mutate({ tokens, keys }, { onSuccess: () => setEditing(false) })} />
          ) : (
            <>
              <div className="muted small">Example lines</div>
              {d.samples.length ? <div className="samples mono">{d.samples.map((s, i) => <div key={i}>{s}</div>)}</div>
                : <div className="muted small">No recent line matches this template.</div>}
            </>
          )}
        </div>
      )}
      {save.error && !editing && <div className="error-text small" style={{ padding: "0 12px 8px" }}>{save.error.message}</div>}
    </li>
  );
}

/** One batch of drafts, as the setup guide and the Maintenance page show it. */
export function ReviewBatch({ batchId, onApplied }: { batchId: string; onApplied?: () => void }) {
  const { data, error } = usePoll<Batch & { drafts: Draft[] }>(`/api/review/${batchId}`, 5000);
  const [filter, setFilter] = useState<"all" | Draft["status"]>("all");
  const [picked, setPicked] = useState<string[]>([]);
  const all = useAction((status: string) => api(`/api/review/${batchId}/decide-all`, { method: "POST", json: { status } }));
  const merge = useAction(() => api(`/api/drafts/${picked[0]}/merge`, { method: "POST", json: { others: picked.slice(1) } }));
  const apply = useAction(() => api(`/api/review/${batchId}/apply`, { method: "POST" }));
  const discard = useAction(() => api(`/api/review/${batchId}/discard`, { method: "POST" }));
  if (!data) return <Loading error={error} />;

  const drafts = data.drafts.filter((d) => d.status !== "merged");
  const n = (s: Draft["status"]) => drafts.filter((d) => d.status === s).length;
  const shown = filter === "all" ? drafts : drafts.filter((d) => d.status === filter);
  const covered = drafts.filter((d) => d.status === "approved").reduce((a, d) => a + d.count, 0);
  const err = all.error ?? merge.error ?? apply.error ?? discard.error;
  if (data.status !== "review") {
    return <div className="callout good"><div className="grow">This review is {data.status}{data.note ? `: ${data.note}` : ""}.</div></div>;
  }

  return (
    <div className="stack">
      <div className="review-summary">
        <div><span className="qv-label">Drafted</span><strong className="num">{drafts.length}</strong></div>
        <div><span className="qv-label">Approved</span><strong className="num good-text">{n("approved")}</strong></div>
        <div><span className="qv-label">Rejected</span><strong className="num">{n("rejected")}</strong></div>
        <div><span className="qv-label">Undecided</span><strong className={`num ${n("pending") ? "warn-text" : ""}`}>{n("pending")}</strong></div>
        <div><span className="qv-label">Lines covered</span><strong className="num">{data.lines ? ((covered / data.lines) * 100).toFixed(0) : 0}%</strong>
          <span className="muted small"> of {fmt(data.lines)} recent</span></div>
      </div>
      <p className="muted small" style={{ margin: 0 }}>
        Drafted <TimeAgo iso={data.created_at} /> from {data.sources.length > 3 ? `${data.sources.length} sources` : data.sources.map(sourceLabel).join(", ")}.
        Click a template to see the lines it matches. Highlighted parts are variables, named by their field.
      </p>

      <div className="spread">
        <div className="seg" role="group" aria-label="Show">
          {(["all", "pending", "approved", "rejected"] as const).map((f) => (
            <button key={f} type="button" className={filter === f ? "on" : ""} aria-pressed={filter === f} onClick={() => setFilter(f)}>
              {f === "all" ? `All ${drafts.length}` : f === "pending" ? `Undecided ${n("pending")}` : `${f[0].toUpperCase()}${f.slice(1)} ${n(f)}`}
            </button>
          ))}
        </div>
        <div className="row">
          {picked.length > 1 && (
            <button className="btn small" onClick={() => merge.mutate(undefined, { onSuccess: () => setPicked([]) })} disabled={merge.isPending}>
              Merge {picked.length} selected
            </button>
          )}
          <button className="btn small" disabled={!n("pending") || all.isPending} onClick={() => all.mutate("approved")}>Approve undecided</button>
          <button className="btn small ghost" disabled={!n("pending") || all.isPending} onClick={() => all.mutate("rejected")}>Reject undecided</button>
        </div>
      </div>

      {shown.length === 0 ? <Empty title="Nothing here" /> : (
        <ul className="drafts">
          {shown.map((d) => (
            <DraftRow key={d.id} d={d} total={data.lines} batch={batchId} selected={picked.includes(d.id)}
              onSelect={(on) => { merge.reset(); setPicked(on ? [...picked, d.id] : picked.filter((p) => p !== d.id)); }} />
          ))}
        </ul>
      )}
      {picked.length === 1 && <div className="muted small">Select another template to merge it into the first one. Words that differ become variables.</div>}

      <div className="review-foot">
        <span className={n("pending") ? "warn-text small" : "muted small"}>
          {n("pending") ? `Decide on ${n("pending")} more template${n("pending") > 1 ? "s" : ""} to continue.`
            : `${n("approved")} template${n("approved") === 1 ? "" : "s"} will be added to the parser; stored lines are back-filled and the affected models retrained.`}
        </span>
        <div className="row">
          {!data.initial && <button className="btn ghost" onClick={() => discard.mutate()} disabled={discard.isPending}>Skip, let the online learner handle it</button>}
          <button className="btn primary" disabled={!!n("pending") || (data.initial && !n("approved")) || apply.isPending}
            onClick={() => apply.mutate(undefined, { onSuccess: onApplied })}>
            {data.initial ? `Apply ${n("approved")} templates and train` : `Add ${n("approved")} templates`}
          </button>
        </div>
      </div>
      {err && <div className="error-text">{err.message}</div>}
    </div>
  );
}

function AutoRow({ t, all }: { t: AutoTemplate; all: AutoTemplate[] }) {
  const [mode, setMode] = useState<"" | "edit" | "same">("");
  const [target, setTarget] = useState("");
  const act = useAction((body: object) => api(`/api/templates/${t.template_id}/review`, { method: "POST", json: body }));
  return (
    <li className="draft">
      <div className="draft-head">
        <span className="muted small num">#{t.template_id}</span>
        <span className="draft-text"><TemplateText tokens={t.tokens} keys={t.keys} /></span>
        <span className="draft-count num">{fmt(t.count)}<span className="muted small"> lines</span></span>
        <button className="btn small" onClick={() => act.mutate({ keep: true })} disabled={act.isPending}>Keep</button>
        <button className="btn small ghost" onClick={() => setMode(mode === "edit" ? "" : "edit")}>Edit</button>
        <button className="btn small ghost" onClick={() => setMode(mode === "same" ? "" : "same")}>Same as…</button>
      </div>
      {mode === "edit" && (
        <div className="draft-body">
          <TemplateEditor tokens={t.tokens} keys={t.keys} saving={act.isPending} error={act.error}
            onCancel={() => setMode("")} onSave={(tokens, keys) => act.mutate({ tokens, keys })} />
        </div>
      )}
      {mode === "same" && (
        <form className="draft-body row" onSubmit={(e) => { e.preventDefault(); act.mutate({ same_as: Number(target) }); }}>
          <input aria-label="Template id" placeholder="template id, e.g. 12" value={target} onChange={(e) => setTarget(e.target.value)}
            list={`same-${t.template_id}`} style={{ width: 200 }} />
          <datalist id={`same-${t.template_id}`}>{all.filter((o) => o.template_id !== t.template_id).map((o) => <option key={o.template_id} value={o.template_id}>{o.tokens.join(" ")}</option>)}</datalist>
          <button className="btn small primary" disabled={!target || act.isPending}>Merge into it</button>
          {act.error && <span className="error-text">{act.error.message}</span>}
        </form>
      )}
      {act.error && mode === "" && <div className="error-text small" style={{ padding: "0 12px 8px" }}>{act.error.message}</div>}
    </li>
  );
}

export function AutoTemplates({ templates }: { templates: AutoTemplate[] }) {
  const keepAll = useAction(() => Promise.all(templates.map((t) => api(`/api/templates/${t.template_id}/review`, { method: "POST", json: { keep: true } }))));
  if (!templates.length) return <Empty title="Nothing to look at">Every template in the parser has been reviewed.</Empty>;
  return (
    <div className="stack">
      <div className="spread">
        <span className="muted small">Already live, so parsing never waits. Keep them, fix their text or field names, or merge one into an existing template.</span>
        <button className="btn small" onClick={() => keepAll.mutate()} disabled={keepAll.isPending}>Keep all {templates.length}</button>
      </div>
      <ul className="drafts">{templates.map((t) => <AutoRow key={t.template_id} t={t} all={templates} />)}</ul>
    </div>
  );
}

/** /review/:id — one batch on its own page. */
export default function ReviewPage() {
  const { id } = useParams();
  return (
    <>
      <PageHead title="Review templates" sub={<>Nothing reaches the parser until you apply it. <Link to="/maintenance">Back to maintenance</Link></>} />
      <Card><ReviewBatch batchId={id!} /></Card>
    </>
  );
}
