// Toasts (and, opted in, desktop notifications) for a new anomaly incident and for jobs that are
// queued, finish or fail; plus the counts the sidebar shows. Only what changes after the page
// loaded is announced, never the backlog.
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { usePoll, type Incident, type Job } from "./api";
import { JOB_NAMES, jobTarget } from "./ui";

interface Toast { id: string; title: string; body: string; to: string; tone: "bad" | "good" | "info" }

const DESKTOP = "desktop-notify";

function desktopOn() {
  try { return typeof Notification !== "undefined" && Notification.permission === "granted" && localStorage.getItem(DESKTOP) === "on"; }
  catch { return false; }
}

export function useNotifications() {
  const incidents = usePoll<{ incidents: Incident[] }>("/api/incidents", 5000).data?.incidents;
  const jobs = usePoll<{ jobs: Job[] }>("/api/jobs", 3000).data?.jobs;
  const seenIncidents = useRef<Set<string> | null>(null);
  const seenJobs = useRef<Map<string, Job["status"]> | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const go = useNavigate();

  const push = (t: Toast, desktop: boolean) => {
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
        body: `${i.host}: ${Math.round(i.peak_miss * 100)}% of events mispredicted. Open to see the unusual lines.` }, true);
    }
  }, [incidents]); // eslint-disable-line react-hooks/exhaustive-deps

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
  return (
    <label className="row small" title={Notification.permission === "denied" ? "Blocked in the browser's site settings" : "Anomalies and finished or failed jobs, while this tab is in the background"}>
      <input type="checkbox" checked={on} onChange={toggle} disabled={Notification.permission === "denied"} />
      Desktop notifications
    </label>
  );
}
