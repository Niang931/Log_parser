// The server's own health: throughput, parse latency, the learning backlog, LLM calls, CPU and
// memory, sampled every 10 s by the server itself (no Prometheus or Grafana needed).
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { usePoll } from "../api";
import { clock, TimeRange, useRange } from "../charts";
import { Card, Loading, PageHead, fmt } from "../ui";

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
