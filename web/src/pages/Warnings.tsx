import { useState } from "react";
import { Link } from "react-router-dom";
import { api, useAction, usePoll, type Source, type Template, type Warning } from "../api";
import { Card, Empty, Loading, PageHead, TimeAgo, fmt, sourceLabel } from "../ui";

function SameAs({ w, sources, templates, onDone }: {
  w: Warning; sources: Source[]; templates: Template[]; onDone: () => void;
}) {
  const [target, setTarget] = useState("");
  const act = useAction((t: string) => api(`/api/warnings/${encodeURIComponent(w.id)}/same-as`, { method: "POST", json: { target: t } }));
  const options = w.kind === "new_source"
    ? sources.filter((s) => s.id !== w.subject && !s.alias_of).map((s) => [s.id, sourceLabel(s.id)])
    : templates.filter((t) => String(t.template_id) !== w.subject && t.alias_of === null)
      .map((t) => [String(t.template_id), `#${t.template_id} ${t.text.slice(0, 80)}`]);
  return (
    <form className="row" onSubmit={(e) => { e.preventDefault(); act.mutate(target, { onSuccess: onDone }); }}>
      <select aria-label="Same as" value={target} onChange={(e) => setTarget(e.target.value)} style={{ maxWidth: 420 }}>
        <option value="">{w.kind === "new_source" ? "Pick the source it behaves like…" : "Pick the template it is the same event as…"}</option>
        {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
      <button className="btn small primary" disabled={!target || act.isPending}>Confirm</button>
      <button type="button" className="btn small ghost" onClick={onDone}>Cancel</button>
      {act.error && <span className="error-text">{act.error.message}</span>}
    </form>
  );
}

function WarningCard({ w, sources, templates }: { w: Warning; sources: Source[]; templates: Template[] }) {
  const [sameAs, setSameAs] = useState(false);
  const resolve = useAction(() => api(`/api/warnings/${encodeURIComponent(w.id)}/resolve`, { method: "POST", json: {} }));
  const recal = useAction(() => api(`/api/warnings/${encodeURIComponent(w.id)}/recalibrate`, { method: "POST" }));
  const isSource = w.kind === "new_source";
  return (
    <Card>
      <div className="stack" style={{ gap: 10 }}>
        <div className="spread">
          <div className="row">
            <span className={`badge ${w.status === "open" ? "warn" : ""}`}>
              {isSource ? "New source" : "Unseen template"}
            </span>
            <h2>{w.title}</h2>
          </div>
          <span className="muted small">first seen <TimeAgo iso={w.first_seen} /> · {fmt(w.count)} lines · last <TimeAgo iso={w.last_seen} /></span>
        </div>
        {isSource ? (
          <p className="ink-2" style={{ margin: 0 }}>{w.detail} Streams it feeds: {w.streams.join(", ") || "none"}.{" "}
            <Link to={`/logs?host=${encodeURIComponent(w.subject.split("|")[0])}&service=${encodeURIComponent(w.subject.split("|")[1])}`}>See its lines</Link></p>
        ) : (
          <>
            <div className="code-block mono">{w.detail}</div>
            <p className="ink-2 small" style={{ margin: 0 }}>
              Reported as novelty, not as an anomaly: models of {w.streams.join(", ")} keep scoring known events. It joins their vocabulary at the next retrain.{" "}
              <Link to={`/logs?template_id=${w.subject}`}>See its lines</Link>
            </p>
          </>
        )}
        {w.status === "resolved" ? (
          <div className="muted small">Resolved <TimeAgo iso={w.resolved_at} />: {w.resolution}</div>
        ) : sameAs ? (
          <SameAs w={w} sources={sources} templates={templates} onDone={() => setSameAs(false)} />
        ) : (
          <div className="row">
            <button className="btn primary small" onClick={() => recal.mutate()} disabled={recal.isPending || recal.isSuccess}>
              {recal.isSuccess ? "Job queued" : isSource ? "Recalibrate with it" : "Retrain now"}
            </button>
            <button className="btn small" onClick={() => setSameAs(true)}>{isSource ? "Same as another source…" : "Same as another template…"}</button>
            <button className="btn small ghost" onClick={() => resolve.mutate()} disabled={resolve.isPending}>Mark resolved</button>
            {recal.isSuccess && <Link to="/jobs" className="small">Follow progress</Link>}
            {(recal.error || resolve.error) && <span className="error-text">{(recal.error ?? resolve.error)!.message}</span>}
          </div>
        )}
      </div>
    </Card>
  );
}

export default function Warnings() {
  const { data, error } = usePoll<{ warnings: Warning[] }>("/api/warnings", 4000);
  const sources = usePoll<{ sources: Source[] }>("/api/sources", 10000).data?.sources ?? [];
  const templates = usePoll<{ templates: Template[] }>("/api/templates", 10000).data?.templates ?? [];
  const [tab, setTab] = useState<"open" | "resolved">("open");
  const list = data?.warnings.filter((w) => w.status === tab) ?? [];
  const openCount = data?.warnings.filter((w) => w.status === "open").length ?? 0;
  return (
    <>
      <PageHead title="Warnings"
        sub="Things the models don't cover yet: sources added after calibration and templates no LogBERT model has seen. Recalibrate, say what they are the same as, or mark them resolved." />
      <div className="tabs" role="tablist">
        <button role="tab" aria-selected={tab === "open"} className={tab === "open" ? "active" : ""} onClick={() => setTab("open")}>Open ({openCount})</button>
        <button role="tab" aria-selected={tab === "resolved"} className={tab === "resolved" ? "active" : ""} onClick={() => setTab("resolved")}>Resolved</button>
      </div>
      {!data ? <Loading error={error} /> : list.length === 0 ? (
        <Card><Empty title={tab === "open" ? "All clear" : "Nothing resolved yet"}>{tab === "open" ? "Every source and template is covered by a model." : null}</Empty></Card>
      ) : (
        <div className="stack">{list.map((w) => <WarningCard key={w.id} w={w} sources={sources} templates={templates} />)}</div>
      )}
    </>
  );
}
