// Query: build a question with dropdowns (time range, filters, a measure, group-by), ask it in
// plain English, or write SQL. A schema sidebar and saved queries sit on the left.
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useSearchParams } from "react-router-dom";
import { api, useAction, usePoll, type QueryResult } from "../api";
import { RANGES, TimeRange } from "../charts";
import { Card, Empty, PageHead, Switch, fmt } from "../ui";

interface Schema {
  engine: string; table: string; payload_access: string; measures: string[]; ops: string[];
  columns: { name: string; builder?: string; about: string }[]; fields: string[];
}
interface Saved { id: string; name: string; sql: string; spec?: Spec | null; saved_at: string }
interface Filter { field: string; op: string; value: string }
interface Spec { minutes: number; filters: Filter[]; measure: { fn: string; field?: string }; group_by: string[]; over_time: boolean; limit: number }

const BASE = ["host", "service", "source", "level", "status", "template_id", "message"];
const MEASURE_LABEL: Record<string, string> = { count: "Count of lines", count_distinct: "Distinct values of", avg: "Average of", min: "Minimum of", max: "Maximum of", sum: "Sum of", p95: "95th percentile of" };
// Categorical order, fixed: a series keeps its colour; past six, groups fold into the table only.
const PALETTE = ["#3b6fd4", "#e07b1a", "#2a9d8f", "#9b5de5", "#c2a40e", "#8a8f98"];
const EMPTY: Spec = { minutes: 60, filters: [], measure: { fn: "count" }, group_by: [], over_time: false, limit: 100 };

function FieldSelect({ value, onChange, fields, label, allowNone }: { value: string; onChange: (v: string) => void; fields: string[]; label: string; allowNone?: boolean }) {
  return (
    <select aria-label={label} value={value} onChange={(e) => onChange(e.target.value)}>
      {allowNone && <option value="">—</option>}
      <optgroup label="Log attributes">{BASE.map((f) => <option key={f} value={f}>{f}</option>)}</optgroup>
      {fields.length > 0 && <optgroup label="Extracted fields">{fields.map((f) => <option key={f} value={`@${f}`}>@{f}</option>)}</optgroup>}
    </select>
  );
}

function Builder({ spec, setSpec, schema, onRun, running }: { spec: Spec; setSpec: (s: Spec) => void; schema?: Schema; onRun: () => void; running: boolean }) {
  const fields = schema?.fields ?? [];
  const range = RANGES.find(([, m]) => m === spec.minutes)?.[0] ?? "1h";
  const setFilter = (i: number, f: Partial<Filter>) => setSpec({ ...spec, filters: spec.filters.map((x, j) => (j === i ? { ...x, ...f } : x)) });
  return (
    <form className="builder" onSubmit={(e) => { e.preventDefault(); onRun(); }}>
      <div className="b-row"><span className="b-label">Time</span><TimeRange value={range} onChange={(r) => setSpec({ ...spec, minutes: RANGES.find(([k]) => k === r)![1] })} /></div>
      <div className="b-row">
        <span className="b-label">Where</span>
        <div className="stack" style={{ gap: 6 }}>
          {spec.filters.map((f, i) => (
            <div key={i} className="row">
              <FieldSelect value={f.field} onChange={(v) => setFilter(i, { field: v })} fields={fields} label="Filter field" />
              <select aria-label="Operator" value={f.op} onChange={(e) => setFilter(i, { op: e.target.value })}>
                {(schema?.ops ?? ["="]).map((o) => <option key={o} value={o}>{o}</option>)}
              </select>
              <input aria-label="Value" value={f.value} placeholder={f.field === "level" ? "error" : "value"} onChange={(e) => setFilter(i, { value: e.target.value })} style={{ width: 180 }} />
              <button type="button" className="btn small ghost" aria-label="Remove filter" onClick={() => setSpec({ ...spec, filters: spec.filters.filter((_, j) => j !== i) })}>×</button>
            </div>
          ))}
          <button type="button" className="btn small ghost" style={{ alignSelf: "flex-start" }}
            onClick={() => setSpec({ ...spec, filters: [...spec.filters, { field: "level", op: "=", value: "error" }] })}>+ Add filter</button>
        </div>
      </div>
      <div className="b-row">
        <span className="b-label">Show</span>
        <div className="row">
          <select aria-label="Measure" value={spec.measure.fn} onChange={(e) => setSpec({ ...spec, measure: { fn: e.target.value, field: e.target.value === "count" ? undefined : spec.measure.field ?? (fields[0] ? `@${fields[0]}` : "host") } })}>
            {(schema?.measures ?? ["count"]).filter((m) => m !== "p95" || schema?.engine === "clickhouse").map((m) => <option key={m} value={m}>{MEASURE_LABEL[m]}</option>)}
          </select>
          {spec.measure.fn !== "count" && <FieldSelect value={spec.measure.field ?? ""} onChange={(v) => setSpec({ ...spec, measure: { ...spec.measure, field: v } })} fields={fields} label="Measure field" />}
        </div>
      </div>
      <div className="b-row">
        <span className="b-label">By</span>
        <div className="row">
          {[0, 1].map((i) => (
            <FieldSelect key={i} allowNone label={`Group by ${i + 1}`} fields={fields} value={spec.group_by[i] ?? ""}
              onChange={(v) => { const g = [...spec.group_by]; g[i] = v; setSpec({ ...spec, group_by: g.filter(Boolean) }); }} />
          ))}
          <Switch checked={spec.over_time} onChange={(on) => setSpec({ ...spec, over_time: on })} label="over time" />
        </div>
      </div>
      <div className="b-row"><span /><button className="btn primary" disabled={running}>{running ? "Running…" : "Run"}</button></div>
    </form>
  );
}

function ResultView({ r }: { r: QueryResult }) {
  const cols = r.columns;
  const numericLast = r.rows.length > 0 && r.rows.every((row) => typeof row[row.length - 1] === "number" || row[row.length - 1] === null);
  const timeCol = cols[0] === "time" && numericLast;
  const value = cols[cols.length - 1];
  let chart = null;
  if (timeCol) {
    const groupIdx = cols.length > 2 ? 1 : -1;
    const totals = new Map<string, number>();
    r.rows.forEach((row) => { const g = groupIdx < 0 ? value : String(row[groupIdx]); totals.set(g, (totals.get(g) ?? 0) + Number(row[row.length - 1] ?? 0)); });
    const series = [...totals.entries()].sort((a, b) => b[1] - a[1]).slice(0, PALETTE.length).map(([g]) => g);
    const byTime = new Map<string, Record<string, number | string>>();
    r.rows.forEach((row) => {
      const t = String(row[0]); const g = groupIdx < 0 ? value : String(row[groupIdx]);
      if (!series.includes(g)) return;
      byTime.set(t, { ...(byTime.get(t) ?? { time: t }), [g]: Number(row[row.length - 1] ?? 0) });
    });
    const data = [...byTime.values()];
    chart = (
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis dataKey="time" tick={{ fill: "var(--muted)", fontSize: 11 }} tickFormatter={(t: string) => new Date(t.replace(" ", "T") + (t.includes("Z") || t.includes("+") ? "" : "Z")).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} minTickGap={40} axisLine={{ stroke: "var(--line)" }} tickLine={false} />
          <YAxis width={44} tick={{ fill: "var(--muted)", fontSize: 11 }} tickFormatter={(n: number) => fmt(n)} axisLine={false} tickLine={false} />
          <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", fontSize: 12 }} />
          {series.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} />}
          {series.map((g, i) => <Line key={g} dataKey={g} stroke={PALETTE[i]} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />)}
        </LineChart>
      </ResponsiveContainer>
    );
  } else if (numericLast && cols.length === 2 && r.rows.length > 1) {
    const data = r.rows.slice(0, 20).map((row) => ({ k: String(row[0]), v: Number(row[1] ?? 0) }));
    chart = (
      <ResponsiveContainer width="100%" height={Math.max(120, data.length * 26)}>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid horizontal={false} stroke="var(--grid)" />
          <XAxis type="number" tick={{ fill: "var(--muted)", fontSize: 11 }} tickFormatter={(n: number) => fmt(n)} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="k" width={160} tick={{ fill: "var(--ink-2)", fontSize: 12 }} axisLine={false} tickLine={false} />
          <Tooltip cursor={{ fill: "var(--hover)" }} contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", fontSize: 12 }} />
          <Bar dataKey="v" name={value} fill={PALETTE[0]} radius={[0, 4, 4, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    );
  }
  return (
    <Card title={`${r.rows.length} row${r.rows.length === 1 ? "" : "s"}`} hint={`${r.engine} · ${r.table}`}>
      {r.rows.length === 0 ? <Empty title="No rows">Widen the time range or remove a filter.</Empty> : (
        <div className="stack">
          {chart}
          <div className="table-wrap" style={{ maxHeight: 420, overflow: "auto" }}>
            <table>
              <thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>
                {r.rows.map((row, i) => (
                  <tr key={i}>{row.map((v, j) => (
                    <td key={j} className={`mono small ${typeof v === "number" ? "right num" : ""}`}>{typeof v === "object" && v !== null ? JSON.stringify(v) : v === null ? <span className="muted">null</span> : String(v)}</td>
                  ))}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </Card>
  );
}

export default function Query() {
  const [params] = useSearchParams();
  const schema = usePoll<Schema>("/api/query/schema", 60000).data;
  const saved = usePoll<{ queries: Saved[] }>("/api/queries", 30000).data?.queries ?? [];
  const [tab, setTab] = useState<"builder" | "sql" | "ask">(params.get("sql") ? "sql" : "builder");
  const [spec, setSpec] = useState<Spec>(EMPTY);
  const [sql, setSql] = useState(params.get("sql") ?? "");
  const [question, setQuestion] = useState("");
  const [name, setName] = useState("");
  const [naming, setNaming] = useState(false);
  const build = useAction((s: Spec) => api<{ sql: string; result: QueryResult }>("/api/query/build", { method: "POST", json: s }));
  const run = useAction((q: string) => api<QueryResult>("/api/query", { method: "POST", json: { sql: q } }));
  const ask = useAction((q: string) => api<{ sql: string }>("/api/query/ask", { method: "POST", json: { question: q } }));
  const save = useAction(() => api("/api/queries", { method: "POST", json: { name, sql: tab === "builder" ? build.data?.sql : sql, spec: tab === "builder" ? spec : null } }));
  const del = useAction((id: string) => api(`/api/queries/${id}`, { method: "DELETE" }));
  const result = tab === "builder" ? build.data?.result : run.data;
  const error = tab === "builder" ? build.error : run.error ?? ask.error;
  const currentSql = tab === "builder" ? build.data?.sql : sql;

  const insert = (text: string) => { setTab("sql"); setSql((s) => (s ? `${s}${s.endsWith(" ") || s.endsWith("\n") ? "" : " "}${text}` : `SELECT ${text}\nFROM ${schema?.table ?? "iot_logs"}`)); };
  const load = (q: Saved) => {
    if (q.spec) { setTab("builder"); setSpec(q.spec); build.mutate(q.spec); } else { setTab("sql"); setSql(q.sql); run.mutate(q.sql); }
  };

  return (
    <>
      <PageHead title="Query" sub="Pick what you want to see, ask in plain English, or write SQL. Everything is read-only." />
      <div className="query-layout">
        <aside className="query-side">
          <Card title="Saved">
            {saved.length === 0 ? <div className="muted small">Run a query, then save it to find it here.</div> : (
              <ul className="side-list">
                {saved.map((q) => (
                  <li key={q.id}>
                    <button type="button" onClick={() => load(q)} title={q.sql}>{q.name}</button>
                    <button type="button" className="x" aria-label={`Delete ${q.name}`} onClick={() => del.mutate(q.id)}>×</button>
                  </li>
                ))}
              </ul>
            )}
          </Card>
          <Card title={<>Table <code>{schema?.table ?? "…"}</code></>} hint="Click to insert into SQL">
            <ul className="side-list schema">
              {schema?.columns.map((c) => (
                <li key={c.name}><button type="button" title={c.about} onClick={() => insert(c.name === "level (computed)" ? "level" : c.name)}>
                  <code>{c.name}</code><span className="muted small">{c.about}</span></button></li>
              ))}
            </ul>
            {schema && schema.fields.length > 0 && (
              <>
                <h3 style={{ marginTop: 10 }}>Extracted fields</h3>
                <div className="row" style={{ gap: 4 }}>
                  {schema.fields.map((f) => <button key={f} type="button" className="chip-btn" title={schema.payload_access.replace("name", f)}
                    onClick={() => insert(schema.payload_access.replace("name", f))}>@{f}</button>)}
                </div>
              </>
            )}
          </Card>
        </aside>

        <div className="stack">
          <Card>
            <div className="tabs" role="tablist" style={{ marginBottom: 12 }}>
              {(["builder", "ask", "sql"] as const).map((t) => (
                <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
                  {t === "builder" ? "Builder" : t === "ask" ? "Ask in English" : "SQL"}
                </button>
              ))}
            </div>
            {tab === "builder" && <Builder spec={spec} setSpec={setSpec} schema={schema} running={build.isPending} onRun={() => build.mutate(spec)} />}
            {tab === "ask" && (
              <form className="stack" onSubmit={(e) => { e.preventDefault(); ask.mutate(question, { onSuccess: (r) => { setSql(r.sql); setTab("sql"); } }); }}>
                <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="e.g. which hosts logged the most errors in the last hour?" aria-label="Question" />
                <div className="row">
                  <button className="btn primary" disabled={!question.trim() || ask.isPending}>{ask.isPending ? "Writing SQL…" : "Write the SQL"}</button>
                  <span className="muted small">The LLM writes a query; you review it in the SQL tab before it runs.</span>
                </div>
                {ask.error && <div className="callout warn">{ask.error.message}</div>}
              </form>
            )}
            {tab === "sql" && (
              <form className="stack" style={{ gap: 8 }} onSubmit={(e) => { e.preventDefault(); run.mutate(sql); }}>
                <textarea rows={8} className="mono" value={sql} spellCheck={false} aria-label="SQL" placeholder={`SELECT service, count() FROM ${schema?.table ?? "iot_logs"} GROUP BY service`}
                  onChange={(e) => setSql(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run.mutate(sql); }} />
                <div className="row">
                  <button className="btn primary" disabled={!sql.trim() || run.isPending}>{run.isPending ? "Running…" : "Run"}</button>
                  <span className="muted small">Ctrl/⌘ + Enter · at most 500 rows</span>
                </div>
              </form>
            )}
            {currentSql && tab === "builder" && (
              <details className="gen-sql"><summary className="small">View SQL</summary>
                <pre className="code-block mono small">{currentSql}</pre>
                <button type="button" className="btn small" onClick={() => { setSql(currentSql); setTab("sql"); }}>Edit as SQL</button>
              </details>
            )}
            {error && tab !== "ask" && <div className="callout bad" style={{ marginTop: 12 }}>{error.message}</div>}
            {result && currentSql && (
              <div className="row" style={{ marginTop: 12 }}>
                {naming ? (
                  <form className="row" onSubmit={(e) => { e.preventDefault(); save.mutate(undefined, { onSuccess: () => { setNaming(false); setName(""); } }); }}>
                    <input autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder="Name" aria-label="Query name" />
                    <button className="btn small primary" disabled={!name.trim() || save.isPending}>Save</button>
                    <button type="button" className="btn small ghost" onClick={() => setNaming(false)}>Cancel</button>
                  </form>
                ) : <button className="btn small" onClick={() => setNaming(true)}>Save query</button>}
                {save.error && <span className="error-text">{save.error.message}</span>}
              </div>
            )}
          </Card>
          {result && <ResultView r={result} />}
        </div>
      </div>
    </>
  );
}
