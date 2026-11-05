// Maintenance: the to-do list that keeps parsing and models up to date. Possible anomalies are
// not here; they are on the Anomaly alerts page.
import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { api, useAction, usePoll, type AutoTemplate, type Maintenance as M, type Source, type Warning } from "../api";
import { Card, Empty, Loading, Meter, PageHead, TimeAgo, fmt, sourceLabel } from "../ui";
import { TemplatePicker } from "../rules";
import { AutoTemplates } from "./Review";

function Item({ tag, tone = "", title, children, actions }: {
  tag: string; tone?: string; title: ReactNode; children?: ReactNode; actions?: ReactNode;
}) {
  return (
    <li className="todo">
      <span className={`badge ${tone}`}>{tag}</span>
      <div className="todo-body">
        <div className="todo-title">{title}</div>
        {children && <div className="todo-detail">{children}</div>}
      </div>
      {actions && <div className="row todo-actions">{actions}</div>}
    </li>
  );
}

function Section({ title, hint, count, children }: { title: string; hint: string; count: number; children: ReactNode }) {
  if (!count) return null;
  return (
    <Card title={<>{title} <span className="count-pill">{count}</span></>} hint={hint}>
      <ul className="todos">{children}</ul>
    </Card>
  );
}

function SameAsSource({ sid, sources }: { sid: string; sources: Source[] }) {
  const [target, setTarget] = useState("");
  const act = useAction(() => api(`/api/warnings/${encodeURIComponent(`new_source:${sid}`)}/same-as`, { method: "POST", json: { target } }));
  return (
    <form className="row" onSubmit={(e) => { e.preventDefault(); act.mutate(); }}>
      <select aria-label="Behaves like" value={target} onChange={(e) => setTarget(e.target.value)} style={{ maxWidth: 260 }}>
        <option value="">Same as another source…</option>
        {sources.filter((s) => s.id !== sid && !s.alias_of && s.calibrated).map((s) => <option key={s.id} value={s.id}>{sourceLabel(s.id)}</option>)}
      </select>
      {target && <button className="btn small">Use its templates</button>}
      {act.error && <span className="error-text">{act.error.message}</span>}
    </form>
  );
}

function NewTemplate({ w }: { w: M["warnings"][number] }) {
  const retrain = useAction(() => api(`/api/warnings/${encodeURIComponent(w.id)}/recalibrate`, { method: "POST" }));
  const dismiss = useAction(() => api(`/api/warnings/${encodeURIComponent(w.id)}/resolve`, { method: "POST", json: { note: "dismissed" } }));
  const same = useAction((target: number) => api(`/api/warnings/${encodeURIComponent(w.id)}/same-as`, { method: "POST", json: { target } }));
  const [linking, setLinking] = useState(false);
  const ready = (w.readiness ?? []).every((r) => (r.have ?? 0) >= (r.need ?? 1));
  return (
    <Item tag="Retrain" tone="info"
      title={<>Template <Link to={`/logs?template_id=${w.subject}`}>#{w.subject}</Link> is new to {w.streams.length > 1 ? `${w.streams.length} models` : `the ${w.readiness?.[0]?.name ?? w.streams[0]} model`}</>}
      actions={<>
        <button className="btn small primary" disabled={!ready || retrain.isPending || retrain.isSuccess} onClick={() => retrain.mutate()}
          title={ready ? "" : "Not enough parsed lines yet: it trains by itself once there are"}>
          {retrain.isSuccess ? "Queued" : "Retrain now"}
        </button>
        <button className="btn small" aria-pressed={linking} onClick={() => setLinking(!linking)}
          title="A reworded line (a firmware update, say) of an event the model already knows: no retrain needed">Same event as…</button>
        <button className="btn small ghost" disabled={dismiss.isPending} onClick={() => dismiss.mutate()}>Dismiss</button>
      </>}>
      <div className="mono small ellipsis">{w.detail}</div>
      <div className="muted small">{fmt(w.count)} lines since <TimeAgo iso={w.first_seen} />. Not an anomaly: the model scores around it and learns it at the next retrain, or link it to the event it is a new wording of.</div>
      {linking && <TemplatePicker exclude={Number(w.subject)} pending={same.isPending} error={same.error}
        onPick={(target) => same.mutate(target, { onSuccess: () => setLinking(false) })} onCancel={() => setLinking(false)} />}
      {(retrain.error || dismiss.error) && <div className="error-text">{(retrain.error ?? dismiss.error)!.message}</div>}
    </Item>
  );
}

function DraftNow({ sid, have, min }: { sid: string; have: number; min: number }) {
  const act = useAction(() => api(`/api/warnings/${encodeURIComponent(`new_source:${sid}`)}/recalibrate`, { method: "POST" }));
  return (
    <>
      <button className="btn small" disabled={have < min || act.isPending || act.isSuccess} onClick={() => act.mutate()}
        title={have < min ? `Needs at least ${min} lines` : "Draft from what has arrived so far"}>
        {act.isSuccess ? "Drafting…" : "Draft now"}
      </button>
      {act.error && <span className="error-text">{act.error.message}</span>}
    </>
  );
}

function History() {
  const { data } = usePoll<{ warnings: Warning[] }>("/api/warnings", 10000);
  const done = data?.warnings.filter((w) => w.status === "resolved") ?? [];
  if (!done.length) return <Empty title="Nothing done yet" />;
  return (
    <ul className="todos">
      {done.map((w) => (
        <Item key={w.id} tag={w.kind === "new_source" ? "Source" : "Template"} title={w.title}>
          <span className="muted small">{w.resolution} · <TimeAgo iso={w.resolved_at} /></span>
        </Item>
      ))}
    </ul>
  );
}

export default function Maintenance() {
  const { data, error } = usePoll<M>("/api/maintenance", 4000);
  const sources = usePoll<{ sources: Source[] }>("/api/sources", 10000).data?.sources ?? [];
  const auto = usePoll<{ auto_templates: AutoTemplate[] }>("/api/review", 10000).data?.auto_templates ?? [];
  const [tab, setTab] = useState<"todo" | "done">("todo");
  const novel = data?.warnings.filter((w) => w.kind === "novel_template") ?? [];
  const total = data ? data.review.length + data.collecting.length + data.waiting.length + novel.length + (auto.length ? 1 : 0) : 0;

  return (
    <>
      <PageHead title="Maintenance"
        sub={<>What keeps parsing and the anomaly models up to date. Most of it runs by itself; this is what needs you or is still waiting. Possible anomalies are under <Link to="/alerts">Anomaly alerts</Link>.</>} />
      <div className="tabs" role="tablist">
        <button role="tab" aria-selected={tab === "todo"} className={tab === "todo" ? "active" : ""} onClick={() => setTab("todo")}>To do ({total})</button>
        <button role="tab" aria-selected={tab === "done"} className={tab === "done" ? "active" : ""} onClick={() => setTab("done")}>History</button>
      </div>
      {tab === "done" ? <Card><History /></Card> : !data ? <Loading error={error} /> : total === 0 ? (
        <Card><Empty title="All caught up">Every source has reviewed templates and every model knows its templates.</Empty></Card>
      ) : (
        <div className="stack">
          <Section title="Review templates" hint="Drafted from a sample of new sources. Nothing reaches the parser until you approve it." count={data.review.length}>
            {data.review.map((b) => (
              <Item key={b.id} tag="Review" tone="warn"
                title={<>{b.drafts as number} templates drafted for {b.sources.length > 2 ? `${b.sources.length} sources` : b.sources.map(sourceLabel).join(", ")}</>}
                actions={<Link className="btn small primary" to={`/review/${b.id}`}>Review</Link>}>
                <span className="muted small">{b.pending} undecided · {b.approved} approved · drafted <TimeAgo iso={b.created_at} /></span>
              </Item>
            ))}
          </Section>

          <Section title="New sources collecting lines" count={data.collecting.length}
            hint="Templates are drafted by themselves once a source has enough lines, or after it has been collecting for a while.">
            {data.collecting.map((c) => (
              <Item key={c.id} tag="Collecting" title={<>{sourceLabel(c.id)} <span className="muted small">{c.source}</span></>}
                actions={<><DraftNow sid={c.id} have={c.have} min={50} /><SameAsSource sid={c.id} sources={sources} /></>}>
                <Meter have={c.have} need={c.need} unit="lines" />
                <span className="muted small">Drafts at {fmt(c.need)} lines or <TimeAgo iso={c.draft_by} />{new Date(c.draft_by) > new Date() ? "" : " (once it has 50 lines)"}, whichever comes first.</span>
              </Item>
            ))}
          </Section>

          <Section title="Models waiting for data" count={data.waiting.length}
            hint="A model needs enough parsed log sequences to train. These start training by themselves when they have them.">
            {data.waiting.map((s) => (
              <Item key={s.id} tag="Waiting" title={<Link to="/streams">{s.name}</Link>}>
                {s.data && <Meter have={s.data.have} need={s.data.need} unit="windows" />}
                <span className="muted small">A window is 20 consecutive parsed lines from one source. Checked every minute.</span>
              </Item>
            ))}
          </Section>

          <Section title="Models to retrain" count={novel.length}
            hint="Templates a model hasn't seen yet. Retrain on your schedule; they are retrained automatically every few days.">
            {novel.map((w) => <NewTemplate key={w.id} w={w} />)}
          </Section>

          {auto.length > 0 && (
            <Card title={<>Added by the online learner <span className="count-pill">{auto.length}</span></>}
              hint="Templates learned from unknown lines as they arrived.">
              <AutoTemplates templates={auto} />
            </Card>
          )}
        </div>
      )}
    </>
  );
}
