// Explorer and dashboard widgets: time range, timeline (stacked bars over the whole range),
// toplist and log level badges.
import type { ReactNode } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useSearchParams } from "react-router-dom";
import type { Histogram, Level } from "./api";
import { fmt } from "./ui";

export const RANGES = [["15m", 15], ["1h", 60], ["4h", 240], ["1d", 1440], ["7d", 10080]] as const;
const RANGE_LABEL: Record<string, string> = { "15m": "Past 15 minutes", "1h": "Past 1 hour", "4h": "Past 4 hours", "1d": "Past 1 day", "7d": "Past 7 days" };

/** The page's time range, kept in `?range=` so it survives reloads and shared links. */
export function useRange(fallback = "1h"): [string, number, (r: string) => void] {
  const [params, setParams] = useSearchParams();
  const key = RANGES.some(([k]) => k === params.get("range")) ? params.get("range")! : fallback;
  const minutes = RANGES.find(([k]) => k === key)![1];
  const set = (r: string) => {
    const next = new URLSearchParams(params);
    next.set("range", r);
    setParams(next, { replace: true });
  };
  return [key, minutes, set];
}

export function TimeRange({ value, onChange }: { value: string; onChange: (r: string) => void }) {
  return (
    <div className="seg" role="group" aria-label="Time range">
      <span className="seg-label">{RANGE_LABEL[value]}</span>
      {RANGES.map(([k]) => (
        <button key={k} type="button" className={k === value ? "on" : ""} aria-pressed={k === value} onClick={() => onChange(k)}>{k}</button>
      ))}
    </div>
  );
}

export const LEVELS: { key: Level; label: string; color: string }[] = [
  { key: "error", label: "Error", color: "var(--lvl-error)" },
  { key: "warn", label: "Warn", color: "var(--lvl-warn)" },
  { key: "info", label: "Info", color: "var(--lvl-info)" },
  { key: "debug", label: "Debug", color: "var(--lvl-debug)" },
];
export const PARSE = [
  { key: "FINAL", label: "Parsed", color: "var(--series-1)" },
  { key: "PENDING", label: "Pending", color: "var(--series-2)" },
];

export function LevelBadge({ level }: { level: Level }) {
  return <span className={`lvl lvl-${level}`}>{level}</span>;
}

type Series = { key: string; label: string; color: string };

function clock(t: number, step: number) {
  const d = new Date(t);
  const hm = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: step < 60 ? "2-digit" : undefined });
  return step >= 1800 ? `${d.toLocaleDateString([], { month: "short", day: "numeric" })} ${hm}` : hm;
}

function Tip({ active, payload, series, step }: { active?: boolean; payload?: { payload: Record<string, number> }[]; series: Series[]; step: number }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  const total = series.reduce((a, s) => a + (p[s.key] ?? 0), 0);
  return (
    <div className="chart-tip">
      <div className="muted">{clock(p.t, step)} – {clock(p.t + step * 1000, step)}</div>
      <table>
        <tbody>
          {[...series].reverse().map((s) => (
            <tr key={s.key} className={p[s.key] ? "" : "muted"}>
              <td><i className="sw" style={{ background: s.color }} />{s.label}</td>
              <td className="right num">{(p[s.key] ?? 0).toLocaleString()}</td>
            </tr>
          ))}
          <tr className="total"><td>Total</td><td className="right num">{total.toLocaleString()}</td></tr>
        </tbody>
      </table>
    </div>
  );
}

/** Counts per bucket over the whole range, stacked (series listed bottom-up). Empty buckets
 * are real zeros from the server, so bars keep one width and quiet periods show as gaps. */
export function Timeline({ data, series, height = 160, legend = true }: { data?: Histogram; series: Series[]; height?: number; legend?: boolean }) {
  if (!data) return <div style={{ height }} className="chart-empty muted small">Loading…</div>;
  const empty = data.points.every((p) => series.every((s) => !p[s.key]));
  return (
    <div className="timeline">
      {legend && (
        <div className="legend">
          {series.map((s) => <span key={s.key}><i style={{ background: s.color }} />{s.label}</span>)}
          <span className="muted">· {data.step >= 60 ? `${data.step / 60} min` : `${data.step} s`} buckets</span>
        </div>
      )}
      <div style={{ position: "relative" }}>
        {empty && <div className="chart-empty muted small" style={{ position: "absolute", inset: 0 }}>No logs in this range</div>}
        <ResponsiveContainer width="100%" height={height}>
          <BarChart data={data.points} margin={{ top: 6, right: 0, bottom: 0, left: 0 }} barCategoryGap={1}>
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis dataKey="t" tickFormatter={(t: number) => clock(t, data.step)} tick={{ fill: "var(--muted)", fontSize: 11 }}
              axisLine={{ stroke: "var(--line)" }} tickLine={false} minTickGap={40} />
            <YAxis width={40} tick={{ fill: "var(--muted)", fontSize: 11 }} axisLine={false} tickLine={false}
              tickFormatter={(n: number) => fmt(n)} allowDecimals={false} />
            <Tooltip content={<Tip series={series} step={data.step} />} cursor={{ fill: "var(--hover)" }} isAnimationActive={false} />
            {series.map((s) => (
              <Bar key={s.key} dataKey={s.key} stackId="s" fill={s.color} isAnimationActive={false} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/** Ranked values with the bar drawn behind the label (Datadog's toplist). */
export function Toplist({ rows, onPick, unit = "lines", color }: { rows: [ReactNode, number, string?][]; onPick?: (key: string) => void; unit?: string; color?: string }) {
  const max = Math.max(1, ...rows.map((r) => r[1]));
  if (!rows.length) return <div className="muted small chart-empty" style={{ height: 80 }}>No data in this range</div>;
  return (
    <ol className="toplist">
      {rows.map(([label, n, key], i) => {
        const body = (
          <>
            <span className="tl-bar" style={{ width: `${(n / max) * 100}%`, background: color }} />
            <span className="tl-label">{label}</span>
            <span className="tl-value num">{fmt(n)}</span>
          </>
        );
        return (
          <li key={i} title={`${n.toLocaleString()} ${unit}`}>
            {onPick && key !== undefined ? <button type="button" onClick={() => onPick(key)}>{body}</button> : <div>{body}</div>}
          </li>
        );
      })}
    </ol>
  );
}
