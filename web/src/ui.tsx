import { useEffect, useState, type ReactNode } from "react";
import type { Job } from "./api";

export function PageHead({ title, sub, actions }: { title: string; sub?: ReactNode; actions?: ReactNode }) {
  useEffect(() => {
    document.title = `${title} · logpipe`;
  }, [title]);
  return (
    <div className="page-head">
      <div>
        <h1>{title}</h1>
        {sub && <div className="sub">{sub}</div>}
      </div>
      {actions && <div className="row">{actions}</div>}
    </div>
  );
}

export function Card({ title, hint, actions, children, className = "" }: {
  title?: ReactNode; hint?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <div className="card-head">
          <div>
            {title && <h2>{title}</h2>}
            {hint && <div className="hint">{hint}</div>}
          </div>
          {actions && <div className="row">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <h2>{title}</h2>
      {children && <div>{children}</div>}
    </div>
  );
}

export function Loading({ error }: { error?: Error | null }) {
  if (error) return <div className="callout bad"><div>Couldn't load: {error.message}</div></div>;
  return <div className="muted" role="status">Loading…</div>;
}

export function StatusBadge({ status }: { status: string }) {
  const kind = status === "FINAL" ? "good" : status === "PENDING" ? "warn" : "";
  return <span className={`badge ${kind}`}>{status === "FINAL" ? "parsed" : status === "PENDING" ? "pending" : status}</span>;
}

export function Progress({ value, state }: { value: number; state?: string }) {
  return (
    <div className={`progress ${state ?? ""}`} role="progressbar" aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={Math.round(value * 100)}>
      <div style={{ width: `${Math.max(2, value * 100)}%` }} />
    </div>
  );
}

export const JOB_NAMES: Record<Job["kind"], string> = {
  draft: "Draft templates",
  apply: "Apply templates",
  retrain: "Train LogBERT",
  calibrate: "Calibration",
  recalibrate: "Recalibration",
};

export function jobTarget(job: Job) {
  const p = job.params;
  if (p.streams?.length) return p.streams.join(", ");
  if (p.sources?.length) return p.sources.map(sourceLabel).join(", ");
  return job.kind === "draft" ? "all sources" : job.kind === "apply" ? "reviewed templates" : "";
}

export function JobRow({ job }: { job: Job }) {
  const state = job.status === "done" ? "done" : job.status === "failed" ? "failed" : "";
  const target = jobTarget(job);
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div className="spread">
        <strong>{JOB_NAMES[job.kind]}{target && <span className="muted" style={{ fontWeight: 400 }}> · {target}</span>}</strong>
        <JobStatus job={job} />
      </div>
      <Progress value={job.status === "queued" ? 0 : job.progress} state={state} />
      <div className={job.error ? "error-text" : "muted small"}>{job.error ?? job.message}</div>
    </div>
  );
}

export function JobStatus({ job }: { job: Job }) {
  const kind = { done: "good", failed: "bad", cancelled: "", queued: "", running: "info" }[job.status];
  return (
    <span className={`badge ${kind}`}>
      {job.status === "running" && <span className="spin" aria-hidden="true" />}
      {job.status === "running" ? `running ${Math.round(job.progress * 100)}%` : job.status}
    </span>
  );
}

/** How far something that starts on its own is from having enough data. */
export function Meter({ have, need, unit }: { have: number; need: number; unit: string }) {
  const ok = have >= need;
  return (
    <div className="meter" title={`${have.toLocaleString()} of ${need.toLocaleString()} ${unit}`}>
      <Progress value={Math.min(1, have / Math.max(need, 1))} state={ok ? "done" : ""} />
      <span className="small num">{fmt(have)} / {fmt(need)} {unit}</span>
    </div>
  );
}

export function sourceLabel(id: string) {
  const [host, service, source] = id.split("|");
  return `${service || source || "?"} on ${host}`;
}

export function TimeAgo({ iso }: { iso?: string | null }) {
  const [, tick] = useState(0);
  useEffect(() => {
    const t = setInterval(() => tick((n) => n + 1), 10_000);
    return () => clearInterval(t);
  }, []);
  if (!iso) return <span className="muted">never</span>;
  return <time dateTime={iso} title={new Date(iso).toLocaleString()}>{ago(iso)}</time>;
}

export function ago(iso: string) {
  const s = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function fmt(n: number | undefined | null) {
  if (n === undefined || n === null) return "–";
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (Math.abs(n) >= 1e4) return `${(n / 1e3).toFixed(1)}k`;
  return n.toLocaleString();
}

export function bytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1 << 20) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1 << 20)).toFixed(1)} MB`;
}

export function time(iso: string) {
  const d = new Date(iso);
  const today = new Date().toDateString() === d.toDateString();
  return today ? d.toLocaleTimeString() : d.toLocaleString();
}

export function CopyBlock({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="code-block">
      <pre>{text}</pre>
      <button className="btn small copy" onClick={() => {
        navigator.clipboard?.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        });
      }}>{copied ? "Copied" : "Copy"}</button>
    </div>
  );
}

/** Template text with its variable slots (<*>, <VAR:X>, also inside words like a=<*>)
 * highlighted and named by their keys, in order. */
export function TemplateText({ tokens, keys }: { tokens: string[]; keys: string[] }) {
  let k = 0;
  return (
    <span className="mono">
      {tokens.map((t, i) => (
        <span key={i}>
          {i > 0 && " "}
          {t.split(/(<\*>|<VAR:[A-Z0-9]+>)/).map((part, j) => j % 2
            ? <span key={j} className="slot" title={part}>{keys[k++] ?? part}</span>
            : part)}
        </span>
      ))}
    </span>
  );
}

export function Drawer({ onClose, children }: { onClose: () => void; children: ReactNode }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true">
        <div className="spread" style={{ marginBottom: 12 }}>
          <span />
          <button className="btn small ghost" onClick={onClose} aria-label="Close">Close ✕</button>
        </div>
        {children}
      </aside>
    </>
  );
}

/** Numbered steps across the top: finished ones show a tick and can be revisited. */
export function Stepper({ steps, current, onPick }: {
  steps: { label: string; done: boolean }[]; current: number; onPick: (i: number) => void;
}) {
  return (
    <ol className="stepper">
      {steps.map((s, i) => {
        const state = i === current ? "current" : s.done ? "done" : "todo";
        const reachable = s.done || i <= current || steps.slice(0, i).every((p) => p.done);
        return (
          <li key={s.label} className={state}>
            <button type="button" disabled={!reachable} onClick={() => onPick(i)} aria-current={i === current ? "step" : undefined}>
              <span className="step-dot" aria-hidden="true">{s.done && i !== current ? "✓" : i + 1}</span>
              <span className="step-label">{s.label}</span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}
