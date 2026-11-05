// The server's own health: throughput, parse latency, the learning backlog, LLM calls, CPU and
// memory, sampled every 10 s by the server itself (no Prometheus or Grafana needed).
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useState } from "react";
import { api, useAction, usePoll } from "../api";
import { clock, TimeRange, useRange } from "../charts";
import { Card, Empty, Loading, PageHead, TimeAgo, fmt } from "../ui";

type Meta = { label: string; unit: string; group: string };
export type SystemData = {
  metrics: Record<string, Meta>;
  current: Record<string, number>;
  sampled_at: number | null;
  every: number;
  cpus: number;
  step: number;
  series: Record<string, [number, number][]>;
};

function num(v: number | undefined, unit = "") {
  if (v === undefined) return "–";
  const n = Math.abs(v) >= 100 ? fmt(Math.round(v)) : v.toLocaleString(undefined, { maximumFractionDigits: v < 10 ? 2 : 1 });
  return unit ? `${n} ${unit}` : n;
}

/** One metric over the range; gaps where the server wasn't running stay gaps. */
export function MetricChart({ name, meta, points, step, height = 120 }: { name: string; meta: Meta; points?: [number, number][]; step: number; height?: number }) {
  const data = (points ?? []).map(([t, v]) => ({ t: t * 1000, v }));
  const id = `fill-${name}`;
  return (
    <div style={{ position: "relative" }}>
      {!data.length && <div className="chart-empty muted small" style={{ position: "absolute", inset: 0 }}>No samples in this range</div>}
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={data} margin={{ top: 6, right: 4, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.25} />
              <stop offset="100%" stopColor="var(--accent)" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} scale="time" tickFormatter={(t: number) => clock(t, step)}
            tick={{ fill: "var(--muted)", fontSize: 11 }} axisLine={{ stroke: "var(--line)" }} tickLine={false} minTickGap={40} />
          <YAxis width={44} tick={{ fill: "var(--muted)", fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(n: number) => num(n)} />
          <Tooltip isAnimationActive={false} cursor={{ stroke: "var(--line)" }}
            content={({ active, payload }) => active && payload?.length ? (
              <div className="chart-tip">
                <div className="muted">{clock(payload[0].payload.t, step)}</div>
                <strong className="num">{num(payload[0].payload.v, meta.unit)}</strong>
              </div>
            ) : null} />
          <Area dataKey="v" type="monotone" stroke="var(--accent)" strokeWidth={2} fill={`url(#${id})`} isAnimationActive={false}
            dot={false} activeDot={{ r: 4, stroke: "var(--surface)", strokeWidth: 2 }} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export type MetricAlert = {
  id: string; rule: string; label: string; metric: string; op: ">" | "<"; threshold: number; value: number; peak: number;
  started_at: number; last_at: number; status: "firing" | "resolved"; resolved_at?: number; resolution?: string;
};
type Rule = { id: string; metric: string; op: ">" | "<"; threshold: number; for_minutes: number; enabled: boolean; label: string };
const iso = (s: number) => new Date(s * 1000).toISOString();

function RuleRow({ rule, metrics }: { rule: Rule; metrics: Record<string, Meta> }) {
  const [threshold, setThreshold] = useState(String(rule.threshold));
  const [minutes, setMinutes] = useState(String(rule.for_minutes));
  const save = useAction((body: Partial<Rule>) => api(`/api/system/rules/${rule.id}`, { method: "PUT", json: body }));
  const remove = useAction(() => api(`/api/system/rules/${rule.id}`, { method: "DELETE" }));
  const dirty = threshold !== String(rule.threshold) || minutes !== String(rule.for_minutes);
  const unit = metrics[rule.metric]?.unit ?? "";
  return (
    <tr className={rule.enabled ? "" : "muted"}>
      <td><input type="checkbox" checked={rule.enabled} aria-label={`Enable ${rule.label}`} onChange={(e) => save.mutate({ enabled: e.target.checked })} /></td>
      <td>{rule.label}<div className="small muted">{metrics[rule.metric]?.label ?? rule.metric}</div></td>
      <td className="nowrap">{rule.op === ">" ? "above" : "below"}{" "}
        <input type="number" value={threshold} onChange={(e) => setThreshold(e.target.value)} aria-label="Threshold" style={{ width: 90 }} /> <span className="small muted">{unit}</span></td>
      <td className="nowrap">for <input type="number" min={0} value={minutes} onChange={(e) => setMinutes(e.target.value)} aria-label="Minutes" style={{ width: 60 }} /> min</td>
      <td className="right nowrap">
        {dirty && <button className="btn small primary" onClick={() => save.mutate({ threshold: Number(threshold), for_minutes: Number(minutes) })}>Save</button>}
        <button className="btn small ghost danger" onClick={() => confirm(`Delete the rule "${rule.label}"?`) && remove.mutate()}>Delete</button>
        {save.error && <span className="error-text">{save.error.message}</span>}
      </td>
    </tr>
  );
}

function AddRule({ metrics }: { metrics: Record<string, Meta> }) {
  const [metric, setMetric] = useState("cpu_pct");
  const [op, setOp] = useState(">");
  const [threshold, setThreshold] = useState("");
  const [minutes, setMinutes] = useState("5");
  const add = useAction(() => api("/api/system/rules", { method: "POST", json: { metric, op, threshold: Number(threshold), for_minutes: Number(minutes) } }));
  return (
    <form className="row" style={{ marginTop: 12 }} onSubmit={(e) => { e.preventDefault(); add.mutate(undefined, { onSuccess: () => setThreshold("") }); }}>
      <select value={metric} onChange={(e) => setMetric(e.target.value)} aria-label="Metric">
        {Object.entries(metrics).map(([k, m]) => <option key={k} value={k}>{m.label} ({m.unit})</option>)}
      </select>
      <select value={op} onChange={(e) => setOp(e.target.value)} aria-label="Above or below"><option value=">">above</option><option value="<">below</option></select>
      <input type="number" required placeholder="threshold" value={threshold} onChange={(e) => setThreshold(e.target.value)} aria-label="Threshold" style={{ width: 100 }} />
      <label className="row small">for <input type="number" min={0} value={minutes} onChange={(e) => setMinutes(e.target.value)} aria-label="Minutes" style={{ width: 60 }} /> min</label>
      <button className="btn">Add rule</button>
      {add.error && <span className="error-text">{add.error.message}</span>}
    </form>
  );
}

function Alerts({ metrics }: { metrics: Record<string, Meta> }) {
  const { data } = usePoll<{ alerts: MetricAlert[]; rules: Rule[] }>("/api/system/alerts", 5000);
  const [showRules, setShowRules] = useState(false);
  if (!data) return null;
  const firing = data.alerts.filter((a) => a.status === "firing");
  const recent = data.alerts.filter((a) => a.status === "resolved").slice(0, 5);
  const unit = (m: string) => metrics[m]?.unit ?? "";
  return (
    <Card title="Alerts" hint={`${data.rules.filter((r) => r.enabled).length} rules checked every sample`}
      actions={<button className="btn small" onClick={() => setShowRules(!showRules)} aria-expanded={showRules}>{showRules ? "Hide rules" : "Edit rules"}</button>}>
      {firing.length === 0 && recent.length === 0 ? <Empty title="No alerts">Every rule is within its threshold.</Empty> : (
        <ul className="todos">
          {[...firing, ...recent].map((a) => (
            <li key={a.id} className="row" style={{ justifyContent: "space-between" }}>
              <span>
                <span className={`badge ${a.status === "firing" ? "bad" : ""}`}>{a.status === "firing" ? "firing" : "resolved"}</span>{" "}
                <strong>{a.label}</strong>{" "}
                <span className="small muted">now {num(a.value, unit(a.metric))} (threshold {a.op} {num(a.threshold, unit(a.metric))}, worst {num(a.peak, unit(a.metric))})</span>
              </span>
              <span className="small muted">since <TimeAgo iso={iso(a.started_at)} />{a.resolved_at && <> · resolved <TimeAgo iso={iso(a.resolved_at)} /></>}</span>
            </li>
          ))}
        </ul>
      )}
      {showRules && (
        <div style={{ marginTop: 12 }}>
          <div className="table-wrap">
            <table>
              <thead><tr><th>On</th><th>Rule</th><th>When</th><th>For</th><th /></tr></thead>
              <tbody>{data.rules.map((r) => <RuleRow key={`${r.id}-${r.threshold}-${r.for_minutes}`} rule={r} metrics={metrics} />)}</tbody>
            </table>
          </div>
          <AddRule metrics={metrics} />
        </div>
      )}
    </Card>
  );
}

type DriftRow = { id: string; name: string; live_windows: number; has_baseline: boolean; trained_at: string | null; psi: number | null; alert_pct: number | null };

function psiLabel(psi: number | null, hasBaseline: boolean) {
  if (!hasBaseline) return <span className="muted small">no baseline: retrain once to start tracking</span>;
  if (psi === null) return <span className="muted small">collecting (needs 200 scored windows)</span>;
  const [tone, text] = psi < 0.1 ? ["good", "stable"] : psi < 0.25 ? ["warn", "moderate shift"] : ["bad", "drifted: retrain"];
  return <><span className="num">{psi.toFixed(3)}</span> <span className={`badge ${tone}`}>{text}</span></>;
}

function DriftCard() {
  const { data } = usePoll<{ streams: DriftRow[] }>("/api/drift", 10000);
  const retrain = useAction((id: string) => api(`/api/streams/${id}/train`, { method: "POST" }));
  if (!data) return null;
  return (
    <Card title="Model drift" hint="How far each stream's anomaly scores have moved from what its model learned (PSI over the last 2,000 windows)">
      {data.streams.length === 0 ? <Empty title="No streams yet" /> : (
        <div className="table-wrap">
          <table>
            <thead><tr><th>Stream</th><th>Drift (PSI)</th><th className="right">Live windows</th><th className="right">Alerted</th><th>Model trained</th><th /></tr></thead>
            <tbody>
              {data.streams.map((d) => (
                <tr key={d.id}>
                  <td>{d.name}</td>
                  <td>{psiLabel(d.psi, d.has_baseline)}</td>
                  <td className="right num">{fmt(d.live_windows)}</td>
                  <td className="right num">{d.alert_pct === null ? "–" : `${d.alert_pct}%`}</td>
                  <td className="small">{d.trained_at ? <TimeAgo iso={d.trained_at} /> : "never"}</td>
                  <td className="right"><button className="btn small" onClick={() => retrain.mutate(d.id)} disabled={retrain.isPending}>Retrain</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {retrain.error && <p className="error-text">{retrain.error.message}</p>}
    </Card>
  );
}

const HEADLINE: [string, string][] = [
  ["lines_per_s", "Throughput"], ["parse_p99_us", "Parse p99"], ["cpu_pct", "CPU"],
  ["rss_mb", "Memory"], ["pending_rows", "Waiting to be learned"], ["llm_errors_per_min", "LLM errors"],
];

export default function System() {
  const [range, minutes, setRange] = useRange("1h");
  const { data, error } = usePoll<SystemData>(`/api/system?minutes=${minutes}`, 10000);
  if (!data) return <><PageHead title="System" /><Loading error={error} /></>;
  const groups = [...new Set(Object.values(data.metrics).map((m) => m.group))];
  const stale = data.sampled_at === null;
  return (
    <>
      <PageHead title="System" sub={`How hard the server is working, sampled every ${data.every} s and kept 30 days. ${data.cpus} CPU cores on this host.`}
        actions={<TimeRange value={range} onChange={setRange} />} />
      {stale && <p className="muted">The first sample is taken about {data.every} s after the server starts.</p>}
      <div className="stack" style={{ marginBottom: 16 }}><Alerts metrics={data.metrics} /><DriftCard /></div>
      <div className="tiles" style={{ marginBottom: 16 }}>
        {HEADLINE.map(([k, label]) => (
          <div className="tile" key={k}>
            <div className="label">{label}</div>
            <div className="value num">{num(data.current[k], data.metrics[k].unit)}</div>
            <div className="foot">{data.metrics[k].label}</div>
          </div>
        ))}
      </div>
      <div className="stack">
        {groups.map((g) => (
          <section key={g}>
            <h2 className="section-title">{g}</h2>
            <div className="dash">
              {Object.entries(data.metrics).filter(([, m]) => m.group === g).map(([k, m]) => (
                <Card key={k} className="w-4" title={m.label} hint={`now ${num(data.current[k], m.unit)}`}>
                  <MetricChart name={k} meta={m} points={data.series[k]} step={data.step} />
                </Card>
              ))}
            </div>
          </section>
        ))}
      </div>
    </>
  );
}
