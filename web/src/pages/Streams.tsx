import { useState } from "react";
import { Link } from "react-router-dom";
import { api, useAction, usePoll, type Selector, type Settings, type Stream } from "../api";
import { Card, Empty, Loading, PageHead, TimeAgo, sourceLabel } from "../ui";

interface Draft { name: string; selectors: Selector[]; retrain_days: number | null }

function StreamForm({ initial, onDone }: { initial?: Stream; onDone: () => void }) {
  const [d, setD] = useState<Draft>({
    name: initial?.name ?? "", selectors: initial?.selectors.length ? initial.selectors : [{}],
    retrain_days: initial?.retrain_days ?? null,
  });
  const save = useAction((draft: Draft) => api(initial ? `/api/streams/${initial.id}` : "/api/streams",
    { method: initial ? "PUT" : "POST", json: draft }));
  const setSel = (i: number, k: keyof Selector, v: string) =>
    setD({ ...d, selectors: d.selectors.map((s, j) => (j === i ? { ...s, [k]: v } : s)) });
  return (
    <form className="stack" style={{ gap: 12 }} onSubmit={(e) => { e.preventDefault(); save.mutate(d, { onSuccess: onDone }); }}>
      <div className="row">
        <label className="field">Name<input value={d.name} onChange={(e) => setD({ ...d, name: e.target.value })} required /></label>
        <label className="field">Retrain every (days)
          <input type="number" min={1} placeholder="default" value={d.retrain_days ?? ""} style={{ width: 120 }}
            onChange={(e) => setD({ ...d, retrain_days: e.target.value ? Number(e.target.value) : null })} />
        </label>
      </div>
      <div>
        <h3>Sources it includes</h3>
        <p className="muted small">A source joins when it matches any row. Wildcards: <code>*</code> and <code>?</code>; empty means any.</p>
        <div className="stack" style={{ gap: 6 }}>
          {d.selectors.map((s, i) => (
            <div className="row" key={i}>
              <input placeholder="host, e.g. gw-*" aria-label="Host pattern" value={s.host ?? ""} onChange={(e) => setSel(i, "host", e.target.value)} />
              <input placeholder="service" aria-label="Service pattern" value={s.service ?? ""} onChange={(e) => setSel(i, "service", e.target.value)} />
              <input placeholder="source" aria-label="Source pattern" value={s.source ?? ""} onChange={(e) => setSel(i, "source", e.target.value)} />
              {d.selectors.length > 1 && <button type="button" className="btn small ghost" onClick={() => setD({ ...d, selectors: d.selectors.filter((_, j) => j !== i) })}>Remove</button>}
            </div>
          ))}
        </div>
        <button type="button" className="btn small ghost" onClick={() => setD({ ...d, selectors: [...d.selectors, {}] })}>+ Add row</button>
      </div>
      <div className="row">
        <button className="btn primary" disabled={save.isPending}>Save stream</button>
        <button type="button" className="btn ghost" onClick={onDone}>Cancel</button>
        {save.error && <span className="error-text">{save.error.message}</span>}
      </div>
    </form>
  );
}

function SettingsCard({ s }: { s: Settings }) {
  const [cap, setCap] = useState(s.sample_cap);
  const [days, setDays] = useState(s.retrain_days);
  const save = useAction(() => api("/api/settings", { method: "PUT", json: { sample_cap: cap, retrain_days: days } }));
  return (
    <Card title="Learning settings" hint="Apply to calibration, recalibration and every retrain">
      <form className="row" onSubmit={(e) => { e.preventDefault(); save.mutate(); }}>
        <label className="field">Sample cap (lines per source)<input type="number" min={100} max={200000} step={100} value={cap} onChange={(e) => setCap(Number(e.target.value))} /></label>
        <label className="field">Default retrain period (days)<input type="number" min={1} value={days} onChange={(e) => setDays(Number(e.target.value))} /></label>
        <button className="btn" style={{ alignSelf: "flex-end" }} disabled={save.isPending}>{save.isSuccess ? "Saved" : "Save"}</button>
        {save.error && <span className="error-text">{save.error.message}</span>}
      </form>
      <p className="muted small" style={{ marginTop: 10, marginBottom: 0 }}>
        A stream retrains on its own once its model is older than its period <em>and</em> it has templates its model hasn't seen.
      </p>
    </Card>
  );
}

function StreamCard({ s, defaultDays }: { s: Stream; defaultDays: number }) {
  const [editing, setEditing] = useState(false);
  const train = useAction(() => api(`/api/streams/${s.id}/train`, { method: "POST" }));
  const del = useAction(() => api(`/api/streams/${s.id}`, { method: "DELETE" }));
  if (editing) return <Card title={`Edit ${s.name}`}><StreamForm initial={s} onDone={() => setEditing(false)} /></Card>;
  return (
    <Card title={s.name}
      hint={s.selectors.map((sel) => Object.entries(sel).map(([k, v]) => `${k}=${v}`).join(" ") || "everything").join("  ·  ")}
      actions={
        <>
          <button className="btn small primary" onClick={() => train.mutate()} disabled={train.isPending || !s.sources.length}>{train.isSuccess ? "Queued" : s.has_model ? "Retrain now" : "Train"}</button>
          <button className="btn small" onClick={() => setEditing(true)}>Edit</button>
          {s.id !== "all" && <button className="btn small ghost danger" onClick={() => confirm(`Delete stream ${s.name}?`) && del.mutate()}>Delete</button>}
        </>
      }>
      <div className="grid cols-2">
        <dl className="kv">
          <dt>Model</dt>
          <dd>{s.has_model ? <><span className="badge good">trained</span> <span className="mono small">{s.model_version}</span></> : <span className="badge">none yet</span>}</dd>
          {s.trained_at && <><dt>Trained</dt><dd><TimeAgo iso={s.trained_at} /> on {s.windows} windows, {s.vocab} templates</dd></>}
          {s.miss_threshold !== undefined && <><dt>Alert when</dt><dd>over {Math.round(s.miss_threshold * 100)}% of masked events are mispredicted</dd></>}
          <dt>Retrains</dt><dd>every {s.retrain_days ?? defaultDays} days if it has unseen templates</dd>
          {s.model_error && <><dt>Last attempt</dt><dd className="error-text">{s.model_error}</dd></>}
        </dl>
        <div>
          <h3>{s.sources.length} source{s.sources.length === 1 ? "" : "s"}</h3>
          <div className="row">{s.sources.slice(0, 12).map((id) => <span key={id} className="badge">{sourceLabel(id)}</span>)}
            {s.sources.length > 12 && <span className="muted small">+{s.sources.length - 12} more</span>}</div>
        </div>
      </div>
      {(train.error || del.error) && <div className="error-text">{(train.error ?? del.error)!.message}</div>}
      {train.isSuccess && <div className="small" style={{ marginTop: 8 }}><Link to="/jobs">Follow the training job</Link></div>}
    </Card>
  );
}

export default function Streams() {
  const { data, error } = usePoll<{ streams: Stream[] }>("/api/streams", 5000);
  const settings = usePoll<Settings>("/api/settings", 10000).data;
  const [adding, setAdding] = useState(false);
  return (
    <>
      <PageHead title="Streams & models"
        sub="A stream groups sources that behave alike. Each one trains its own LogBERT anomaly model; all of them share the template registry."
        actions={!adding && <button className="btn primary" onClick={() => setAdding(true)}>New stream</button>} />
      <div className="stack">
        {adding && <Card title="New stream"><StreamForm onDone={() => setAdding(false)} /></Card>}
        {!data ? <Loading error={error} /> : data.streams.length === 0 ? <Card><Empty title="No streams" /></Card>
          : data.streams.map((s) => <StreamCard key={s.id} s={s} defaultDays={settings?.retrain_days ?? 7} />)}
        {settings && <SettingsCard s={settings} />}
      </div>
    </>
  );
}
