import { Fragment, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { usePoll, type LogLine, type Source } from "../api";
import { Card, Drawer, Empty, Loading, PageHead, StatusBadge, time } from "../ui";

const FIELDS = ["host", "service", "source", "status", "template_id", "q"] as const;

export default function Logs() {
  const [params, setParams] = useSearchParams();
  const [live, setLive] = useState(true);
  const [open, setOpen] = useState<LogLine | null>(null);
  const [text, setText] = useState(params.get("q") ?? "");
  const query = new URLSearchParams();
  FIELDS.forEach((f) => params.get(f) && query.set(f, params.get(f)!));
  query.set("limit", "300");
  const { data, error, isFetching } = usePoll<{ logs: LogLine[] }>(`/api/logs?${query}`, live ? 2000 : 0);
  const sources = usePoll<{ sources: Source[] }>("/api/sources", 10000).data?.sources ?? [];
  const uniq = (k: "host" | "service" | "source") => [...new Set(sources.map((s) => s[k]))].filter(Boolean).sort();

  const set = (k: string, v: string) => {
    const next = new URLSearchParams(params);
    if (v) next.set(k, v); else next.delete(k);
    setParams(next, { replace: true });
  };
  const select = (k: "host" | "service" | "source", label: string) => (
    <select aria-label={label} value={params.get(k) ?? ""} onChange={(e) => set(k, e.target.value)}>
      <option value="">All {label}s</option>
      {uniq(k).map((v) => <option key={v}>{v}</option>)}
    </select>
  );
  const filtered = FIELDS.some((f) => params.get(f));

  return (
    <>
      <PageHead title="Live logs" sub="Newest first. Multiline events (stack traces) arrive as one entry."
        actions={
          <>
            <span className="muted small" aria-live="polite">{live ? (isFetching ? "updating…" : "live") : "paused"}</span>
            <button className="btn" onClick={() => setLive(!live)}>{live ? "Pause" : "Resume"}</button>
          </>
        } />
      <Card>
        <form className="filters" onSubmit={(e) => { e.preventDefault(); set("q", text); }}>
          <input type="search" placeholder="Search message text… (Enter)" value={text} onChange={(e) => setText(e.target.value)} aria-label="Search messages" />
          {select("host", "host")}
          {select("service", "service")}
          {select("source", "source")}
          <select aria-label="Status" value={params.get("status") ?? ""} onChange={(e) => set("status", e.target.value)}>
            <option value="">Any status</option>
            <option value="FINAL">Parsed</option>
            <option value="PENDING">Pending</option>
          </select>
          {params.get("template_id") && (
            <span className="badge info">template #{params.get("template_id")}
              <button className="btn small ghost" style={{ height: 18 }} onClick={() => set("template_id", "")} aria-label="Clear template filter">✕</button>
            </span>
          )}
          {filtered && <button type="button" className="btn small ghost" onClick={() => { setText(""); setParams({}); }}>Clear filters</button>}
        </form>
        {!data ? <Loading error={error} /> : data.logs.length === 0 ? (
          <Empty title={filtered ? "Nothing matches" : "No logs yet"}>
            {filtered ? "Try fewer filters." : <>Connect an agent from the <Link to="/setup">setup guide</Link>.</>}
          </Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>Time</th><th>Host</th><th>Service</th><th>Status</th><th style={{ width: "60%" }}>Message</th></tr></thead>
              <tbody>
                {data.logs.map((l) => (
                  <tr key={l.log_id} className="clickable" onClick={() => setOpen(l)} tabIndex={0}
                    onKeyDown={(e) => e.key === "Enter" && setOpen(l)}>
                    <td className="nowrap small num">{time(l.timestamp)}</td>
                    <td className="nowrap small">{l.host}</td>
                    <td className="nowrap small">{l.service}</td>
                    <td><StatusBadge status={l.status} /></td>
                    <td className="msg"><div className="clamp">{l.message}</div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
      {open && (
        <Drawer onClose={() => setOpen(null)}>
          <div className="stack">
            <div className="row"><StatusBadge status={open.status} /><span className="muted">{new Date(open.timestamp).toLocaleString()}</span></div>
            <div className="code-block"><pre style={{ whiteSpace: "pre-wrap" }}>{open.message}</pre></div>
            <dl className="kv">
              <dt>Host</dt><dd>{open.host}</dd>
              <dt>Service</dt><dd>{open.service || "–"}</dd>
              <dt>Source</dt><dd>{open.source || "–"}</dd>
              <dt>Tags</dt><dd>{open.tags.length ? open.tags.map((t) => <span key={t} className="badge" style={{ marginRight: 4 }}>{t}</span>) : "–"}</dd>
              <dt>Template</dt><dd>{open.status === "FINAL"
                ? <Link to={`/templates?id=${open.template_id}`} onClick={() => setOpen(null)}>#{open.template_id}</Link>
                : "not matched yet (learned online, then back-filled)"}</dd>
            </dl>
            {Object.keys(open.payload).length > 0 && (
              <div>
                <h3>Extracted fields</h3>
                <dl className="kv mono">
                  {Object.entries(open.payload).map(([k, v]) => <Fragment key={k}><dt>{k}</dt><dd>{v}</dd></Fragment>)}
                </dl>
              </div>
            )}
          </div>
        </Drawer>
      )}
    </>
  );
}
