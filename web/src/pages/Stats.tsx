import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { usePoll, type Stats as S } from "../api";
import { Card, Loading, PageHead, RankBars, fmt } from "../ui";

const RANGES: [string, number][] = [["15 min", 15], ["1 hour", 60], ["6 hours", 360], ["24 hours", 1440], ["7 days", 10080]];

type Point = S["per_minute"][number];

function Tip({ active, payload, label }: { active?: boolean; payload?: { payload: Point }[]; label?: string }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="chart-tip">
      <div className="muted">{label?.replace("T", " ")}</div>
      <div className="legend" style={{ flexDirection: "column", gap: 2, marginTop: 4 }}>
        <span><i style={{ background: "var(--series-1)" }} />Parsed <strong className="num">{p.FINAL.toLocaleString()}</strong></span>
        <span><i style={{ background: "var(--series-2)" }} />Pending <strong className="num">{p.PENDING.toLocaleString()}</strong></span>
      </div>
    </div>
  );
}

/** Lines per minute, stacked by parse status (two series: legend + tooltip). */
export function MinuteChart({ data, height = 260 }: { data: Point[]; height?: number }) {
  if (!data.length) return <div className="muted small" style={{ height, display: "grid", placeItems: "center" }}>No lines in this range yet.</div>;
  return (
    <div>
      <div className="legend" style={{ marginBottom: 8 }}>
        <span><i style={{ background: "var(--series-1)" }} />Parsed</span>
        <span><i style={{ background: "var(--series-2)" }} />Pending</span>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -12 }} barCategoryGap={1}>
          <CartesianGrid vertical={false} stroke="var(--line)" />
          <XAxis dataKey="minute" tickFormatter={(m: string) => m.slice(11)} tick={{ fill: "var(--muted)", fontSize: 11 }}
            axisLine={{ stroke: "var(--line-strong)" }} tickLine={false} minTickGap={24} />
          <YAxis tick={{ fill: "var(--muted)", fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(n: number) => fmt(n)} allowDecimals={false} />
          <Tooltip content={<Tip />} cursor={{ fill: "var(--surface-2)" }} />
          <Bar dataKey="FINAL" stackId="s" fill="var(--series-1)" stroke="var(--surface)" strokeWidth={1} isAnimationActive={false} />
          <Bar dataKey="PENDING" stackId="s" fill="var(--series-2)" stroke="var(--surface)" strokeWidth={1} radius={[3, 3, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function Stats() {
  const [minutes, setMinutes] = useState(60);
  const { data, error } = usePoll<S>(`/api/stats?minutes=${minutes}`, 5000);
  return (
    <>
      <PageHead title="Statistics" sub="Volume over time and who sends the most."
        actions={
          <div className="row" role="group" aria-label="Time range">
            {RANGES.map(([label, m]) => (
              <button key={m} className={`btn small ${m === minutes ? "primary" : ""}`} onClick={() => setMinutes(m)}>{label}</button>
            ))}
          </div>
        } />
      {!data ? <Loading error={error} /> : (
        <div className="stack">
          <Card title="Lines per minute"><MinuteChart data={data.per_minute} /></Card>
          <div className="grid cols-3">
            <Card title="Top hosts"><RankBars rows={data.hosts} /></Card>
            <Card title="Top services"><RankBars rows={data.services.map(([s, n]) => [s || "(none)", n])} /></Card>
            <Card title="Top sources"><RankBars rows={data.sources.map(([s, n]) => [s || "(none)", n])} /></Card>
          </div>
          <Card title="Most frequent templates">
            <RankBars rows={data.templates.map(([id, n, text]) => [<span className="mono" title={text}>#{id} {text}</span>, n])} />
          </Card>
        </div>
      )}
    </>
  );
}
