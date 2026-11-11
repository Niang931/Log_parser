// Template review: drafted templates wait here until a person approves, edits, merges or rejects
// them; only approved ones reach the parser. Also lists what the online learner added by itself.
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, useAction, usePoll, type AutoTemplate, type Batch, type Draft } from "../api";
import { RuleEditor, TemplatePicker, type RuleCheck } from "../rules";
import { Card, Empty, Loading, PageHead, TemplateText, TimeAgo, fmt, sourceLabel } from "../ui";

function DraftRow({ d, total, selected, onSelect }: {
  d: Draft; total: number; selected: boolean; onSelect: (on: boolean) => void;
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
          {d.by && <span className="badge" style={{ marginLeft: 6 }} title="Last changed by an AI assistant key">by {d.by}</span>}
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
            <RuleEditor initial={d.rule} saving={save.isPending} error={save.error}
              check={(rule) => api<RuleCheck>(`/api/drafts/${d.id}/check`, { method: "POST", json: { rule } })}
              onCancel={() => setEditing(false)}
              onSave={(rule) => save.mutate({ rule }, { onSuccess: () => setEditing(false) })} />
          ) : (
            <>
              {d.rule_edited && <div className="small"><span className="muted">Rule:</span> <code>{d.rule}</code></div>}
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
  const [q, setQ] = useState("");
  const all = useAction((status: string) => api(`/api/review/${batchId}/decide-all`, { method: "POST", json: { status } }));
  const some = useAction((status: Draft["status"]) => api(`/api/review/${batchId}/decide-all`, { method: "POST", json: { status, ids: picked } }));
  const finish = useAction(async () => {
    await api(`/api/review/${batchId}/decide-all`, { method: "POST", json: { status: "approved" } });
    return api(`/api/review/${batchId}/apply`, { method: "POST" });
  });
  const merge = useAction(() => api(`/api/drafts/${picked[0]}/merge`, { method: "POST", json: { others: picked.slice(1) } }));
  const apply = useAction(() => api(`/api/review/${batchId}/apply`, { method: "POST" }));
  const discard = useAction(() => api(`/api/review/${batchId}/discard`, { method: "POST" }));
  if (!data) return <Loading error={error} />;

  const drafts = data.drafts.filter((d) => d.status !== "merged");
  const n = (s: Draft["status"]) => drafts.filter((d) => d.status === s).length;
  const needle = q.trim().toLowerCase();
  const shown = (filter === "all" ? drafts : drafts.filter((d) => d.status === filter)).filter((d) => !needle
    || d.tokens.join(" ").toLowerCase().includes(needle) || d.samples.some((x) => x.toLowerCase().includes(needle)));
  const allShown = shown.length > 0 && shown.every((d) => picked.includes(d.id));
  const toggleShown = () => setPicked(allShown ? picked.filter((p) => !shown.some((d) => d.id === p))
    : [...new Set([...picked, ...shown.map((d) => d.id)])]);
  const bulk = (status: Draft["status"]) => some.mutate(status, { onSuccess: () => setPicked([]) });
  const covered = drafts.filter((d) => d.status === "approved").reduce((a, d) => a + d.count, 0);
  const err = all.error ?? some.error ?? finish.error ?? merge.error ?? apply.error ?? discard.error;
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
          <input type="search" placeholder="Search templates and lines" aria-label="Search templates and example lines"
            value={q} onChange={(e) => setQ(e.target.value)} style={{ width: 220 }} />
          <button className="btn small" disabled={!n("pending") || all.isPending} onClick={() => all.mutate("approved")}>Approve undecided</button>
          <button className="btn small ghost" disabled={!n("pending") || all.isPending} onClick={() => all.mutate("rejected")}>Reject undecided</button>
        </div>
      </div>

      <div className="bulk-bar">
        <label className="row" style={{ gap: 6 }}>
          <input type="checkbox" checked={allShown} onChange={toggleShown} disabled={!shown.length} />
          {picked.length ? `${picked.length} selected` : `Select all ${shown.length} shown`}
        </label>
        {picked.length > 0 && (
          <div className="row">
            <button className="btn small" onClick={() => bulk("approved")} disabled={some.isPending}>Approve</button>
            <button className="btn small" onClick={() => bulk("rejected")} disabled={some.isPending}>Reject</button>
            <button className="btn small ghost" onClick={() => bulk("pending")} disabled={some.isPending}>Undecide</button>
            {picked.length > 1 && (
              <button className="btn small" onClick={() => merge.mutate(undefined, { onSuccess: () => setPicked([]) })} disabled={merge.isPending}
                title="One template for all of them: words that differ become variables">Merge</button>
            )}
            <button className="btn small ghost" onClick={() => setPicked([])}>Clear</button>
          </div>
        )}
      </div>

      {shown.length === 0 ? <Empty title="Nothing here" /> : (
        <ul className="drafts">
          {shown.map((d) => (
            <DraftRow key={d.id} d={d} total={data.lines} selected={picked.includes(d.id)}
              onSelect={(on) => { merge.reset(); setPicked(on ? [...picked, d.id] : picked.filter((p) => p !== d.id)); }} />
          ))}
        </ul>
      )}

      <div className="review-foot">
        <span className={n("pending") ? "warn-text small" : "muted small"}>
          {n("pending") ? `${n("pending")} template${n("pending") > 1 ? "s" : ""} undecided: decide them, or approve the rest in one go.`
            : `${n("approved")} template${n("approved") === 1 ? "" : "s"} will be added to the parser; stored lines are back-filled and the affected models retrained.`}
        </span>
        <div className="row">
          {!data.initial && <button className="btn ghost" onClick={() => discard.mutate()} disabled={discard.isPending}>Skip, let the online learner handle it</button>}
          {n("pending") > 0 ? (
            <button className="btn primary" disabled={finish.isPending} onClick={() => finish.mutate(undefined, { onSuccess: onApplied })}
              title="Approves every undecided template, then applies">
              {data.initial ? `Approve the rest and train (${n("approved") + n("pending")})` : `Approve the rest and add (${n("approved") + n("pending")})`}
            </button>
          ) : (
            <button className="btn primary" disabled={(data.initial && !n("approved")) || apply.isPending}
              onClick={() => apply.mutate(undefined, { onSuccess: onApplied })}>
              {data.initial ? `Apply ${n("approved")} templates and train` : `Add ${n("approved")} templates`}
            </button>
          )}
        </div>
      </div>
      {err && <div className="error-text">{err.message}</div>}
    </div>
  );
}

function AutoRow({ t }: { t: AutoTemplate }) {
  const [mode, setMode] = useState<"" | "edit" | "link">("");
  const act = useAction((body: object) => api(`/api/templates/${t.template_id}/review`, { method: "POST", json: body }));
  return (
    <li className="draft">
      <div className="draft-head">
        <span className="muted small num">#{t.template_id}</span>
        <span className="draft-text"><TemplateText tokens={t.tokens} keys={t.keys} /></span>
        <span className="draft-count num">{fmt(t.count)}<span className="muted small"> lines</span></span>
        <button className="btn small" onClick={() => act.mutate({ keep: true })} disabled={act.isPending}>Keep</button>
        <button className="btn small ghost" aria-pressed={mode === "edit"} onClick={() => setMode(mode === "edit" ? "" : "edit")}>Edit rule</button>
        <button className="btn small ghost" aria-pressed={mode === "link"} onClick={() => setMode(mode === "link" ? "" : "link")}>Same event as…</button>
      </div>
      {mode === "edit" && (
        <div className="draft-body">
          <RuleEditor initial={t.rule} saving={act.isPending} error={act.error} saveLabel="Save and re-extract stored lines"
            check={(rule) => api<RuleCheck>(`/api/templates/${t.template_id}/check`, { method: "POST", json: { rule } })}
            onCancel={() => setMode("")} onSave={(rule) => act.mutate({ rule }, { onSuccess: () => setMode("") })} />
        </div>
      )}
      {mode === "link" && (
        <div className="draft-body">
          <TemplatePicker exclude={t.template_id} pending={act.isPending} error={act.error}
            onPick={(target) => act.mutate({ same_as: target }, { onSuccess: () => setMode("") })} onCancel={() => setMode("")} />
        </div>
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
      <ul className="drafts">{templates.map((t) => <AutoRow key={t.template_id} t={t} />)}</ul>
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
