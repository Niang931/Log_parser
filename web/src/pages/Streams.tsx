import { useState } from "react";
import { Link } from "react-router-dom";
import { api, useAction, usePoll, type Group, type Selector, type Settings, type Source, type Stream } from "../api";
import { Card, Loading, PageHead, TimeAgo, fmt, sourceLabel, Meter } from "../ui";

interface Draft { name: string; picked: string[]; selectors: Selector[]; retrain_days: number | null; group: string | null }

/** Name, the sources it watches (ticked; each source is in one stream), and, folded away,
 *  patterns for sources that don't exist yet and the retrain period. */
function StreamForm({ initial, streams, groups, onDone }: { initial?: Stream; streams: Stream[]; groups: Group[]; onDone: () => void }) {
  const sources = usePoll<{ sources: Source[] }>("/api/sources", 5000).data?.sources.filter((s) => !s.alias_of);
  const [d, setD] = useState<Draft>({
    name: initial?.name ?? "", picked: initial?.picked ?? initial?.sources ?? [],
    selectors: initial?.selectors ?? [], retrain_days: initial?.retrain_days ?? null, group: initial?.group ?? null,
  });
  const save = useAction((draft: Draft) => api(initial ? `/api/streams/${initial.id}` : "/api/streams",
    { method: initial ? "PUT" : "POST", json: { ...draft, selectors: draft.selectors.filter((s) => s.host || s.service || s.source) } }));
  const setSel = (i: number, k: keyof Selector, v: string) =>
    setD({ ...d, selectors: d.selectors.map((s, j) => (j === i ? { ...s, [k]: v } : s)) });
  const toggle = (id: string) => setD({ ...d, picked: d.picked.includes(id) ? d.picked.filter((p) => p !== id) : [...d.picked, id] });
  const owner = (id: string) => streams.find((s) => s.sources.includes(id));
  return (
    <form className="stack" style={{ gap: 12 }} onSubmit={(e) => { e.preventDefault(); save.mutate(d, { onSuccess: onDone }); }}>
      <div className="row" style={{ alignItems: "flex-end" }}>
        <label className="field grow">Name<input value={d.name} onChange={(e) => setD({ ...d, name: e.target.value })} required placeholder="e.g. Payments" /></label>
        <label className="field">Group
          <select value={d.group ?? ""} onChange={(e) => setD({ ...d, group: e.target.value || null })}>
            <option value="">No group</option>
            {groups.map((g) => <option key={g.id} value={g.id}>{g.name}</option>)}
          </select>
        </label>
      </div>
        <div>
          <h3>Sources it watches</h3>
          <p className="muted small" style={{ margin: "0 0 6px" }}>Ticking a source moves it here. A source you untick goes back to a stream of the same service, or gets its own; a stream left empty is removed.</p>
          {!sources ? <Loading /> : sources.length === 0 ? <div className="muted small">No sources yet.</div> : (
            <ul className="pick-list">
              {sources.map((src) => {
                const now = owner(src.id);
                const moves = !d.picked.includes(src.id) ? null : now && now.id !== initial?.id ? now.name : null;
                return (
                  <li key={src.id}>
                    <label>
                      <input type="checkbox" checked={d.picked.includes(src.id)} onChange={() => toggle(src.id)} />
                      <span className="grow">{sourceLabel(src.id)} <span className="muted small">{src.source}</span></span>
                      {moves ? <span className="badge warn">moves from {moves}</span>
                        : now && now.id !== initial?.id && <span className="muted small">in {now.name}</span>}
                      <span className="num muted small">{fmt(src.count)} lines</span>
                    </label>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      <details className="advanced">
        <summary>Advanced: future sources and retraining</summary>
        <div className="stack" style={{ gap: 8, marginTop: 8 }}>
            <p className="muted small" style={{ margin: 0 }}>Also add sources that appear later and match a pattern. Wildcards: <code>*</code> and <code>?</code>. A ticked source always wins over a pattern.</p>
            {d.selectors.map((s, i) => (
              <div className="row" key={i}>
                <input placeholder="host, e.g. gw-*" aria-label="Host pattern" value={s.host ?? ""} onChange={(e) => setSel(i, "host", e.target.value)} />
                <input placeholder="service" aria-label="Service pattern" value={s.service ?? ""} onChange={(e) => setSel(i, "service", e.target.value)} />
                <input placeholder="source" aria-label="Source pattern" value={s.source ?? ""} onChange={(e) => setSel(i, "source", e.target.value)} />
                <button type="button" className="btn small ghost" onClick={() => setD({ ...d, selectors: d.selectors.filter((_, j) => j !== i) })}>Remove</button>
              </div>
            ))}
            <button type="button" className="btn small ghost" style={{ alignSelf: "flex-start" }} onClick={() => setD({ ...d, selectors: [...d.selectors, {}] })}>+ Add pattern</button>
          <label className="field">Retrain every (days)
            <input type="number" min={1} placeholder="default" value={d.retrain_days ?? ""} style={{ width: 120 }}
              onChange={(e) => setD({ ...d, retrain_days: e.target.value ? Number(e.target.value) : null })} />
          </label>
        </div>
      </details>
      <div className="row">
        <button className="btn primary" disabled={save.isPending}>Save stream</button>
        <button type="button" className="btn ghost" onClick={onDone}>Cancel</button>
        {save.error && <span className="error-text">{save.error.message}</span>}
      </div>
    </form>
  );
}

/** The stream list with add/edit in place: the Streams page and the setup guide's step. */
export function StreamsEditor({ compact = false }: { compact?: boolean }) {
  const { data, error } = usePoll<{ streams: Stream[] }>("/api/streams", 5000);
  const groups = usePoll<{ groups: Group[] }>("/api/groups", 5000).data?.groups ?? [];
  const settings = usePoll<Settings>("/api/settings", 10000).data;
  const [editing, setEditing] = useState<string | null>(null);  // stream id, or "new"
  if (!data) return <Loading error={error} />;
  const form = (s?: Stream) => <StreamForm initial={s} streams={data.streams} groups={groups} onDone={() => setEditing(null)} />;
  const groupName = (s: Stream) => groups.find((g) => g.id === s.group)?.name;
  return (
    <div className="stack">
      {data.streams.map((s) => editing === s.id ? <Card key={s.id} title={`Edit ${s.name}`}>{form(s)}</Card>
        : compact ? <StreamRow key={s.id} s={s} onEdit={() => setEditing(s.id)} />
        : <StreamCard key={s.id} s={s} group={groupName(s)} defaultDays={settings?.retrain_days ?? 7} onEdit={() => setEditing(s.id)} />)}
      {editing === "new" ? <Card title="New stream">{form()}</Card>
        : <button className="btn" style={{ alignSelf: "flex-start" }} onClick={() => setEditing("new")}>+ New stream</button>}
    </div>
  );
}

function StreamRow({ s, onEdit }: { s: Stream; onEdit: () => void }) {
  const del = useAction(() => api(`/api/streams/${s.id}`, { method: "DELETE" }));
  return (
    <div className="stream-row">
      <div className="grow">
        <strong>{s.name}</strong> <span className="muted small">{s.sources.length} source{s.sources.length === 1 ? "" : "s"}</span>
        <div className="row" style={{ marginTop: 4 }}>
          {s.sources.length ? s.sources.slice(0, 8).map((id) => <span key={id} className="badge">{sourceLabel(id)}</span>)
            : <span className="muted small">No sources yet{s.selectors.length ? ": waiting for one matching its pattern" : ""}.</span>}
          {s.sources.length > 8 && <span className="muted small">+{s.sources.length - 8} more</span>}
        </div>
      </div>
      <button className="btn small" onClick={onEdit}>Edit</button>
      {!s.sources.length && <button className="btn small ghost danger" disabled={del.isPending}
        onClick={() => del.mutate()}>Delete</button>}
    </div>
  );
}

function StreamCard({ s, group, defaultDays, onEdit }: { s: Stream; group?: string; defaultDays: number; onEdit: () => void }) {
  const train = useAction(() => api<{ waiting?: string[] }>(`/api/streams/${s.id}/train`, { method: "POST" }));
  const short = s.waiting && s.data && s.data.have < s.data.need;
  const del = useAction(() => api(`/api/streams/${s.id}`, { method: "DELETE" }));
  return (
    <Card title={<>{s.name} {group && <span className="badge info" title="The group this stream is in">{group}</span>}</>}
      hint={s.selectors.length ? `also future sources matching ${s.selectors.map((sel) => Object.entries(sel).map(([k, v]) => `${k}=${v}`).join(" ")).join(" or ")}` : undefined}
      actions={
        <>
          <button className="btn small primary" onClick={() => train.mutate()} disabled={train.isPending || !s.sources.length || !!short}
            title={short ? "Not enough parsed lines yet: it trains by itself once there are" : ""}>
            {train.isSuccess && !train.data?.waiting ? "Queued" : s.has_model ? "Retrain now" : "Train"}
          </button>
          <button className="btn small" onClick={onEdit}>Edit</button>
          {!s.sources.length && <button className="btn small ghost danger" onClick={() => del.mutate()}>Delete</button>}
        </>
      }>
      <div className="grid cols-2">
        <dl className="kv">
          <dt>Model</dt>
          <dd>{s.has_model ? <><span className="badge good">trained</span> <span className="mono small">{s.model_version}</span></>
            : s.waiting ? <span className="badge warn">waiting for data</span> : <span className="badge">none yet</span>}</dd>
          {s.waiting && s.data && <><dt>Training data</dt><dd><Meter have={s.data.have} need={s.data.need} unit="windows" />
            <span className="muted small">Trains by itself once it has enough; checked every minute.</span></dd></>}
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
      {train.isSuccess && !train.data?.waiting && <div className="small" style={{ marginTop: 8 }}><Link to="/jobs">Follow the training job</Link></div>}
    </Card>
  );
}

export default function Streams() {
  return (
    <>
      <PageHead title="Streams & models"
        sub="A stream is a set of sources that behave alike: each source is in exactly one stream, and each stream trains its own LogBERT model. A group puts several streams together, e.g. mail = ldap + proxy, so you can ask about all of them at once." />
      <div className="stack">
        <GroupsCard />
        <StreamsEditor />
        <p className="muted small">Sample cap, retrain period and when templates are drafted: <Link to="/settings#learning">Settings → Learning</Link>.</p>
      </div>
    </>
  );
}

/** Groups: each one names several streams (a stream is in at most one). Ask can be asked about a
 *  group ("why is mail down?"), which covers the services of all its streams. */
function GroupsCard() {
  const groups = usePoll<{ groups: Group[] }>("/api/groups", 5000);
  const streams = usePoll<{ streams: Stream[] }>("/api/streams", 5000).data?.streams ?? [];
  const [editing, setEditing] = useState<string | null>(null);  // group id, or "new"
  if (!groups.data) return <Loading error={groups.error} />;
  const list = groups.data.groups;
  return (
    <Card title="Groups" hint="Streams managed together. Ask covers all of a group's services when you name it."
      actions={editing !== "new" && <button className="btn small" onClick={() => setEditing("new")}>+ New group</button>}>
      <div className="stack" style={{ gap: 8 }}>
        {list.length === 0 && editing !== "new" && <p className="muted small" style={{ margin: 0 }}>No groups yet. Make one for each system you run, e.g. <em>mail</em> with the ldap and proxy streams.</p>}
        {list.map((g) => editing === g.id ? <GroupForm key={g.id} initial={g} streams={streams} onDone={() => setEditing(null)} />
          : <GroupRow key={g.id} g={g} onEdit={() => setEditing(g.id)} />)}
        {editing === "new" && <GroupForm streams={streams} onDone={() => setEditing(null)} />}
      </div>
    </Card>
  );
}

function GroupRow({ g, onEdit }: { g: Group; onEdit: () => void }) {
  const del = useAction(() => api(`/api/groups/${g.id}`, { method: "DELETE" }));
  return (
    <div className="stream-row">
      <div className="grow">
        <strong>{g.name}</strong> {g.description && <span className="muted small">{g.description}</span>}
        <div className="row" style={{ marginTop: 4 }}>
          {g.streams.length ? g.streams.map((s) => <span key={s.id} className="badge">{s.name}</span>)
            : <span className="muted small">No streams yet.</span>}
        </div>
        {g.services.length > 0 && <div className="muted small" style={{ marginTop: 4 }}>Services: {g.services.join(", ")}</div>}
      </div>
      <button className="btn small" onClick={onEdit}>Edit</button>
      <button className="btn small ghost danger" disabled={del.isPending} onClick={() => del.mutate()}
        title="Its streams stay; they just leave the group">Delete</button>
      {del.error && <span className="error-text small">{del.error.message}</span>}
    </div>
  );
}

function GroupForm({ initial, streams, onDone }: { initial?: Group; streams: Stream[]; onDone: () => void }) {
  const [name, setName] = useState(initial?.name ?? "");
  const [description, setDescription] = useState(initial?.description ?? "");
  const [picked, setPicked] = useState<string[]>(initial?.streams.map((s) => s.id) ?? []);
  const save = useAction(() => api(initial ? `/api/groups/${initial.id}` : "/api/groups",
    { method: initial ? "PUT" : "POST", json: { name, description, streams: picked } }));
  const toggle = (id: string) => setPicked(picked.includes(id) ? picked.filter((p) => p !== id) : [...picked, id]);
  return (
    <form className="stack" style={{ gap: 10, padding: 12, border: "1px solid var(--line)", borderRadius: "var(--radius)" }}
      onSubmit={(e) => { e.preventDefault(); save.mutate(undefined, { onSuccess: onDone }); }}>
      <div className="row" style={{ alignItems: "flex-end" }}>
        <label className="field">Name<input value={name} onChange={(e) => setName(e.target.value)} required maxLength={60} placeholder="e.g. mail" /></label>
        <label className="field grow">Description<input value={description} onChange={(e) => setDescription(e.target.value)} maxLength={300} placeholder="optional, e.g. ldap + proxy in front of the mail servers" /></label>
      </div>
      <div>
        <h3>Streams in it</h3>
        <ul className="pick-list">
          {streams.map((s) => {
            const elsewhere = s.group && s.group !== initial?.id;
            return (
              <li key={s.id}><label>
                <input type="checkbox" checked={picked.includes(s.id)} onChange={() => toggle(s.id)} />
                <span className="grow">{s.name} <span className="muted small">{s.sources.length} source{s.sources.length === 1 ? "" : "s"}</span></span>
                {elsewhere && picked.includes(s.id) && <span className="badge warn">moves from its group</span>}
              </label></li>
            );
          })}
        </ul>
      </div>
      <div className="row">
        <button className="btn primary small" disabled={save.isPending}>Save group</button>
        <button type="button" className="btn ghost small" onClick={onDone}>Cancel</button>
        {save.error && <span className="error-text small">{save.error.message}</span>}
      </div>
    </form>
  );
}
