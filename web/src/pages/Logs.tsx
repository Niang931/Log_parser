import { Fragment, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { usePoll, type Facets, type Histogram, type LogLine } from "../api";
import { LEVELS, LevelBadge, PARSE, Timeline, TimeRange, useRange } from "../charts";
import { Drawer, Empty, Loading, StatusBadge, fmt } from "../ui";

const FILTERS = ["level", "service", "host", "source", "status", "template_id"] as const;
const FACET_GROUPS: { key: keyof Facets; title: string }[] = [
  { key: "level", title: "Status" },
  { key: "service", title: "Service" },
  { key: "host", title: "Host" },
  { key: "source", title: "Source" },
  { key: "status", title: "Parsing" },
];
const valueLabel = (facet: string, v: string) =>
  facet === "level" ? LEVELS.find((l) => l.key === v)?.label ?? v
    : facet === "status" ? PARSE.find((p) => p.key === v)?.label ?? v : v || "(none)";
const swatch = (facet: string, v: string) =>
  facet === "level" ? LEVELS.find((l) => l.key === v)?.color : facet === "status" ? PARSE.find((p) => p.key === v)?.color : undefined;

function stamp(iso: string) {
  const d = new Date(iso);
  return `${d.toLocaleDateString([], { month: "short", day: "2-digit" })} ${d.toLocaleTimeString([], { hour12: false })}.${String(d.getMilliseconds()).padStart(3, "0")}`;
}

function FacetPanel({ facets, params, toggle }: { facets?: Facets; params: URLSearchParams; toggle: (k: string, v: string) => void }) {
  const [closed, setClosed] = useState<Record<string, boolean>>({});
  return (
    <aside className="facets" aria-label="Facets">
      {FACET_GROUPS.map(({ key, title }) => {
        const rows = facets?.[key] ?? [];
        const picked = params.get(key);
        return (
          <section key={key} className="facet">
            <button type="button" className="facet-head" aria-expanded={!closed[key]} onClick={() => setClosed({ ...closed, [key]: !closed[key] })}>
              <span className="chev">{closed[key] ? "▸" : "▾"}</span>{title}
            </button>
            {!closed[key] && (
              <ul>
                {rows.length === 0 && <li className="muted small facet-none">none in range</li>}
                {rows.map(([v, n]) => {
                  const on = picked === String(v);
                  const color = swatch(key, String(v));
                  return (
                    <li key={String(v)}>
                      <button type="button" className={`facet-val ${on ? "on" : ""} ${picked && !on ? "dim" : ""}`} aria-pressed={on}
                        onClick={() => toggle(key, on ? "" : String(v))}>
                        <span className={`check ${on ? "on" : ""}`} aria-hidden="true" />
                        {color && <i className="sw" style={{ background: color }} />}
                        <span className="facet-label">{valueLabel(key, String(v))}</span>
                        <span className="facet-count num">{fmt(n)}</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </section>
        );
      })}
    </aside>
  );
}

export default function Logs() {
  const [params, setParams] = useSearchParams();
  const [range, minutes, setRange] = useRange("15m");
  const [live, setLive] = useState(true);
  const [open, setOpen] = useState<LogLine | null>(null);
  const [text, setText] = useState(params.get("q") ?? "");

  const query = new URLSearchParams({ minutes: String(minutes) });
  [...FILTERS, "q"].forEach((f) => params.get(f) && query.set(f, params.get(f)!));
  const every = live ? 2000 : 0;
  const logs = usePoll<{ logs: LogLine[] }>(`/api/logs?${query}&limit=300`, every);
  const hist = usePoll<Histogram>(`/api/logs/histogram?${query}&by=level`, live ? 5000 : 0);
  const facets = usePoll<Facets>(`/api/logs/facets?${query}`, live ? 5000 : 0);

  const set = (k: string, v: string) => {
    const next = new URLSearchParams(params);
    if (v) next.set(k, v); else next.delete(k);
    setParams(next, { replace: true });
  };
  const active = [...FILTERS, "q"].filter((f) => params.get(f));
  const total = hist.data?.points.reduce((a, p) => a + LEVELS.reduce((b, l) => b + (p[l.key] ?? 0), 0), 0);

  return (
    <div className="explorer-page">
      <div className="toolbar">
        <h1>Log Explorer</h1>
        <div className="row">
          <button type="button" className={`live ${live ? "on" : ""}`} onClick={() => setLive(!live)} aria-pressed={live}>
            <span className="pulse" />{live ? "Live" : "Paused"}
          </button>
          <TimeRange value={range} onChange={setRange} />
        </div>
      </div>

      <form className="searchbar" onSubmit={(e) => { e.preventDefault(); set("q", text); }}>
        <svg viewBox="0 0 24 24" className="icon" aria-hidden="true"><path d="M11 4a7 7 0 100 14 7 7 0 000-14zm9 16l-4-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>
        {active.filter((f) => f !== "q").map((f) => (
          <span key={f} className="chip">
            <span className="muted">{f === "template_id" ? "template" : f === "level" ? "status" : f === "status" ? "parsing" : f}:</span>
            {f === "template_id" ? `#${params.get(f)}` : valueLabel(f, params.get(f)!)}
            <button type="button" aria-label={`Remove ${f} filter`} onClick={() => set(f, "")}>×</button>
          </span>
        ))}
        <input type="search" placeholder="Search log messages, then Enter" value={text} onChange={(e) => setText(e.target.value)} aria-label="Search messages" />
        {active.length > 0 && <button type="button" className="btn small ghost" onClick={() => { setText(""); setParams(new URLSearchParams({ range })); }}>Clear all</button>}
      </form>

      <div className="widget timeline-widget">
        <Timeline data={hist.data} series={[...LEVELS].reverse()} height={110} />
      </div>

      <div className="explorer">
        <FacetPanel facets={facets.data} params={params} toggle={set} />
        <section className="widget loglist" aria-label="Logs">
          <div className="loglist-head">
            <span><strong className="num">{total === undefined ? "…" : total.toLocaleString()}</strong> logs in range</span>
            <span className="muted small">{logs.data && `showing newest ${logs.data.logs.length}`}</span>
          </div>
          {!logs.data ? <div style={{ padding: 16 }}><Loading error={logs.error} /></div> : logs.data.logs.length === 0 ? (
            <Empty title={active.length ? "Nothing matches" : "No logs in this range"}>
              {active.length ? "Remove a filter or widen the time range." : <>Add an agent under <Link to="/agents">Agents</Link>.</>}
            </Empty>
          ) : (
            <table className="logs">
              <thead><tr><th className="col-date">Date</th><th className="col-host">Host</th><th className="col-svc">Service</th><th>Content</th></tr></thead>
              <tbody>
                {logs.data.logs.map((l) => (
                  <tr key={l.log_id} className={`lvl-row row-${l.level} ${open?.log_id === l.log_id ? "sel" : ""}`} onClick={() => setOpen(l)} tabIndex={0}
                    onKeyDown={(e) => e.key === "Enter" && setOpen(l)} aria-label={`${l.level} log from ${l.host}`}>
                    <td className="col-date num">{stamp(l.timestamp)}</td>
                    <td className="col-host">{l.host}</td>
                    <td className="col-svc">{l.service || <span className="muted">–</span>}</td>
                    <td className="content">{l.message.split("\n")[0]}{l.message.includes("\n") && <span className="more">+{l.message.split("\n").length - 1} lines</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </div>

      {open && (
        <Drawer onClose={() => setOpen(null)}>
          <div className="stack">
            <div className="row">
              <LevelBadge level={open.level} />
              <strong>{open.service || open.source || "log"}</strong>
              <span className="muted">on {open.host}</span>
              <span className="muted" style={{ marginLeft: "auto" }}>{new Date(open.timestamp).toLocaleString()}</span>
            </div>
            <div className="code-block"><pre style={{ whiteSpace: "pre-wrap" }}>{open.message}</pre></div>
            <div className="row">
              <button className="btn small" onClick={() => { set("host", open.host); setOpen(null); }}>Only this host</button>
              {open.service && <button className="btn small" onClick={() => { set("service", open.service); setOpen(null); }}>Only this service</button>}
              {open.status === "FINAL" && <button className="btn small" onClick={() => { set("template_id", String(open.template_id)); setOpen(null); }}>Same template</button>}
            </div>
            <h3>Attributes</h3>
            <dl className="kv">
              <dt>status</dt><dd><LevelBadge level={open.level} /></dd>
              <dt>host</dt><dd>{open.host}</dd>
              <dt>service</dt><dd>{open.service || "–"}</dd>
              <dt>source</dt><dd>{open.source || "–"}</dd>
              <dt>tags</dt><dd>{open.tags.length ? open.tags.map((t) => <span key={t} className="badge" style={{ marginRight: 4 }}>{t}</span>) : "–"}</dd>
              <dt>parsing</dt><dd><StatusBadge status={open.status} /></dd>
              <dt>template</dt><dd>{open.status === "FINAL"
                ? <Link to={`/templates?id=${open.template_id}`} onClick={() => setOpen(null)}>#{open.template_id}</Link>
                : "not matched yet (learned online, then back-filled)"}</dd>
            </dl>
            {Object.keys(open.payload).length > 0 && (
              <>
                <h3>Extracted fields</h3>
                <dl className="kv mono">
                  {Object.entries(open.payload).map(([k, v]) => <Fragment key={k}><dt>{k}</dt><dd>{v}</dd></Fragment>)}
                </dl>
              </>
            )}
          </div>
        </Drawer>
      )}
    </div>
  );
}
