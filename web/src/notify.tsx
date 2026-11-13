// Toasts (and, opted in, desktop notifications) for a new anomaly incident, a firing metric alert,
// new maintenance to-dos and jobs that are queued, finish or fail; alerts and maintenance also
// chime unless muted. Plus the counts the sidebar shows. Only what changes after the page
// loaded is announced, never the backlog.
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { usePoll, type Incident, type Job } from "./api";
import type { MetricAlert } from "./pages/System";
import { JOB_NAMES, jobTarget, Switch } from "./ui";

interface Toast { id: string; title: string; body: string; to: string; tone: "bad" | "good" | "info" }

const DESKTOP = "desktop-notify";
const SOUND = "alert-sound";

function soundOn() {
  try { return localStorage.getItem(SOUND) !== "off"; } catch { return true; }
}

let audio: AudioContext | null = null;
// shortcut: browsers keep audio blocked until the page has had a click or key press, so a tab
// nobody has touched since loading stays silent; fine for an operator console.
function chime() {
  try {
    audio ??= new AudioContext();
    const t = audio.currentTime, osc = audio.createOscillator(), gain = audio.createGain();
    osc.frequency.setValueAtTime(880, t);
    osc.frequency.setValueAtTime(660, t + 0.12);
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(0.2, t + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.4);
    osc.connect(gain).connect(audio.destination);
    osc.start(t);
    osc.stop(t + 0.4);
  } catch { /* no Web Audio */ }
}

function desktopOn() {
  try { return typeof Notification !== "undefined" && Notification.permission === "granted" && localStorage.getItem(DESKTOP) === "on"; }
  catch { return false; }
}

/** `todo`: the Maintenance count, undefined until the overview has loaded. */
export function useNotifications(todo?: number) {
  const incidents = usePoll<{ incidents: Incident[] }>("/api/incidents", 5000).data?.incidents;
  const jobs = usePoll<{ jobs: Job[] }>("/api/jobs", 3000).data?.jobs;
  const metricAlerts = usePoll<{ alerts: MetricAlert[] }>("/api/system/alerts", 10000).data?.alerts;
  const seenMetric = useRef<Set<string> | null>(null);
  const seenIncidents = useRef<Set<string> | null>(null);
  const seenJobs = useRef<Map<string, Job["status"]> | null>(null);
  const lastTodo = useRef<number | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const go = useNavigate();

  const push = (t: Toast, desktop: boolean, sound = false) => {
    if (sound && soundOn()) chime();
    setToasts((all) => [...all.filter((x) => x.id !== t.id), t].slice(-4));
    setTimeout(() => setToasts((all) => all.filter((x) => x.id !== t.id)), t.tone === "bad" ? 15000 : 6000);
    if (desktop && desktopOn() && document.hidden) {
      const n = new Notification(t.title, { body: t.body, tag: t.id });
      n.onclick = () => { window.focus(); go(t.to); n.close(); };
    }
  };

  useEffect(() => {
    if (!incidents) return;
    if (seenIncidents.current === null) { seenIncidents.current = new Set(incidents.map((i) => i.id)); return; }
    for (const i of incidents) {
      if (seenIncidents.current.has(i.id)) continue;
      seenIncidents.current.add(i.id);
      push({ id: `inc-${i.id}`, title: `Anomaly on ${i.service || i.host}`, tone: "bad", to: `/alerts/${i.id}`,
        body: `${i.host}: ${Math.round(i.peak_miss * 100)}% of events mispredicted. Open to see the unusual lines.` }, true, true);
    }
  }, [incidents]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!metricAlerts) return;
    const firing = metricAlerts.filter((a) => a.status === "firing");
    if (seenMetric.current === null) { seenMetric.current = new Set(firing.map((a) => `${a.rule}@${a.started_at}`)); return; }
    for (const a of firing) {
      const key = `${a.rule}@${a.started_at}`;
      if (seenMetric.current.has(key)) continue;
      seenMetric.current.add(key);
      push({ id: `metric-${key}`, title: a.label, tone: "bad", to: "/system",
        body: `${a.metric} is ${a.value.toLocaleString()} (threshold ${a.op} ${a.threshold.toLocaleString()}).` }, true, true);
    }
  }, [metricAlerts]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (todo === undefined) return;
    const before = lastTodo.current;
    lastTodo.current = todo;
    if (before === null || todo <= before) return;
    push({ id: "maintenance", title: "Maintenance needs you", tone: "info", to: "/maintenance",
      body: `${todo} item${todo === 1 ? "" : "s"} to review: new templates or models that need retraining.` }, true, true);
  }, [todo]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!jobs) return;
    if (seenJobs.current === null) { seenJobs.current = new Map(jobs.map((j) => [j.id, j.status])); return; }
    for (const j of jobs) {
      const before = seenJobs.current.get(j.id);
      seenJobs.current.set(j.id, j.status);
      if (before === j.status) continue;
      const what = `${JOB_NAMES[j.kind]}${jobTarget(j) ? ` · ${jobTarget(j)}` : ""}`;
      if (before === undefined && (j.status === "queued" || j.status === "running"))
        push({ id: `job-${j.id}`, title: "Job queued", body: what, to: "/jobs", tone: "info" }, false);
      else if (j.status === "done")
        push({ id: `job-${j.id}`, title: "Job finished", body: `${what}: ${j.message}`, to: "/jobs", tone: "good" }, true);
      else if (j.status === "failed")
        push({ id: `job-${j.id}`, title: "Job failed", body: `${what}: ${j.error ?? ""}`, to: "/jobs", tone: "bad" }, true);
    }
  }, [jobs]); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    toasts, dismiss: (id: string) => setToasts((all) => all.filter((x) => x.id !== id)),
    openIncidents: incidents?.filter((i) => i.status === "open").length ?? 0,
    firingMetrics: metricAlerts?.filter((a) => a.status === "firing").length ?? 0,
    activeJobs: jobs?.filter((j) => j.status === "queued" || j.status === "running").length ?? 0,
  };
}

export function Toasts({ toasts, dismiss }: { toasts: Toast[]; dismiss: (id: string) => void }) {
  const go = useNavigate();
  return (
    <div className="toasts" role="status" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.tone}`}>
          <button type="button" className="toast-body" onClick={() => { dismiss(t.id); go(t.to); }}>
            <strong>{t.title}</strong>
            <span className="small">{t.body}</span>
          </button>
          <button type="button" className="toast-x" aria-label="Dismiss" onClick={() => dismiss(t.id)}>×</button>
        </div>
      ))}
    </div>
  );
}

/** Opt in to desktop notifications (the browser asks once). */
export function DesktopToggle() {
  const supported = typeof Notification !== "undefined";
  const [on, setOn] = useState(desktopOn);
  if (!supported) return null;
  const toggle = async () => {
    if (on) { try { localStorage.setItem(DESKTOP, "off"); } catch { /* private window */ } setOn(false); return; }
    const granted = Notification.permission === "granted" || (await Notification.requestPermission()) === "granted";
    try { localStorage.setItem(DESKTOP, granted ? "on" : "off"); } catch { /* private window */ }
    setOn(granted);
  };
  const denied = Notification.permission === "denied";
  return (
    <div className="stack" style={{ gap: 4 }}>
      <Switch checked={on} onChange={toggle} disabled={denied} label="Desktop notifications" />
      <span className="muted small">{denied ? "Blocked in this browser's site settings: allow notifications for this site there, then come back."
        : "Anomalies, metric alerts, maintenance and finished or failed jobs, while this tab is in the background."}</span>
    </div>
  );
}

/** Mute the chime that plays with anomalies, metric alerts and new maintenance. */
export function SoundToggle() {
  const [on, setOn] = useState(soundOn);
  const toggle = (next: boolean) => {
    try { localStorage.setItem(SOUND, next ? "on" : "off"); } catch { /* private window */ }
    setOn(next);
    if (next) chime();
  };
  return (
    <div className="stack" style={{ gap: 4 }}>
      <Switch checked={on} onChange={toggle} label="Alert sound" />
      <span className="muted small">A short chime with anomalies, metric alerts and new maintenance items.</span>
    </div>
  );
}
