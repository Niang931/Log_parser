import { useEffect, useState, type ReactNode } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { usePoll, useServerEvents, type Overview } from "./api";
import { LogoutButton } from "./login";
import { DesktopToggle, Toasts, useNotifications } from "./notify";
import AgentDetail from "./pages/AgentDetail";
import Agents from "./pages/Agents";
import Alerts, { IncidentPage } from "./pages/Alerts";
import Jobs from "./pages/Jobs";
import Keys from "./pages/Keys";
import Logs from "./pages/Logs";
import OverviewPage from "./pages/Overview";
import Query from "./pages/Query";
import Setup from "./pages/Setup";
import StatsPage from "./pages/Stats";
import Streams from "./pages/Streams";
import Templates from "./pages/Templates";
import Maintenance from "./pages/Maintenance";
import ReviewPage from "./pages/Review";

const paths: Record<string, string> = {
  setup: "M9 12l2 2 4-4M12 3a9 9 0 100 18 9 9 0 000-18z",
  overview: "M4 13h6V4H4v9zm0 7h6v-5H4v5zm10 0h6v-9h-6v9zm0-16v5h6V4h-6z",
  logs: "M4 6h16M4 12h16M4 18h10",
  templates: "M8 4H5a1 1 0 00-1 1v3m16 0V5a1 1 0 00-1-1h-3m0 16h3a1 1 0 001-1v-3M4 16v3a1 1 0 001 1h3M9 12h6",
  query: "M8 9l-3 3 3 3m8-6l3 3-3 3M13 6l-2 12",
  stats: "M5 20V10m7 10V4m7 16v-7",
  alerts: "M12 9v4m0 4h.01M10.3 3.9L2.4 17.5A2 2 0 004.1 20.5h15.8a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z",
  warnings: "M12 8v5m0 3h.01M12 3a9 9 0 100 18 9 9 0 000-18z",
  streams: "M4 7h10m-10 5h16M4 17h7",
  agents: "M5 6h14v8H5zM9 18h6m-3-4v4",
  keys: "M15 7a3 3 0 11-3 3m3-3l5-5m-2 2l2 2M12 10l-8 8v2h2l1-1v-2h2v-2h2l1-1",
  jobs: "M12 7v5l3 3M12 3a9 9 0 100 18 9 9 0 000-18z",
  demo: "M7 4l12 8-12 8V4z",
};

export function Icon({ name }: { name: keyof typeof paths }) {
  return (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={paths[name]} />
    </svg>
  );
}

function Item({ to, icon, children, count }: { to: string; icon: keyof typeof paths; children: ReactNode; count?: number }) {
  return (
    <NavLink to={to} end={to === "/"}>
      <Icon name={icon} />
      {children}
      {!!count && <span className="count" aria-label={`${count} open`}>{count}</span>}
    </NavLink>
  );
}

function ThemeToggle() {
  const [theme, setTheme] = useState<string>(() => {
    try { return localStorage.getItem("theme") ?? "auto"; } catch { return "auto"; }
  });
  useEffect(() => {
    if (theme === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("theme", theme); } catch { /* private window */ }
  }, [theme]);
  return (
    <label className="row small">
      Theme
      <select value={theme} onChange={(e) => setTheme(e.target.value)} style={{ minHeight: 26, padding: "2px 6px" }}>
        <option value="auto">System</option>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
    </label>
  );
}

export default function App() {
  const { data } = usePoll<Overview>("/api/overview", 3000);
  useServerEvents();
  const note = useNotifications();
  const [open, setOpen] = useState(false);
  const location = useLocation();
  useEffect(() => setOpen(false), [location.pathname]);
  const calibrated = data?.settings.calibrated;
  // What needs a person: templates to review and models with templates they haven't seen.
  const todo = (data?.review ?? 0) + (data?.warnings.novel_template ?? 0);

  return (
    <div className="shell">
      <div className="mobile-bar">
        <button className="btn small" onClick={() => setOpen(!open)} aria-expanded={open}>Menu</button>
        <strong>logpipe</strong>
      </div>
      <nav className={`nav ${open ? "open" : ""}`} aria-label="Main">
        <div className="brand">
          <span className="brand-mark"><svg width="16" height="16" viewBox="0 0 16 16"><path d="M3 4h10M3 8h7M3 12h9" stroke="#fff" strokeWidth="2" strokeLinecap="round" /></svg></span>
          logpipe
        </div>
        {calibrated === false && <Item to="/setup" icon="setup">Setup guide</Item>}
        <Item to="/" icon="overview">Overview</Item>
        <div className="nav-group">Explore</div>
        <Item to="/logs" icon="logs">Log Explorer</Item>
        <Item to="/templates" icon="templates">Templates</Item>
        <Item to="/query" icon="query">Query</Item>
        <Item to="/stats" icon="stats">Analytics</Item>
        <div className="nav-group">Detection</div>
        <Item to="/alerts" icon="alerts" count={note.openIncidents}>Anomaly alerts</Item>
        <Item to="/maintenance" icon="warnings" count={todo}>Maintenance</Item>
        <Item to="/streams" icon="streams">Streams &amp; models</Item>
        <Item to="/jobs" icon="jobs" count={note.activeJobs}>Jobs</Item>
        <div className="nav-group">Fleet</div>
        <Item to="/agents" icon="agents">Agents</Item>
        <Item to="/keys" icon="keys">API keys</Item>
        {calibrated && <Item to="/setup" icon="setup">Setup guide</Item>}
        <div className="nav-foot">
          <a href="/demo/" className="row"><Icon name="demo" />Developer demo</a>
          <a href="/api/docs">API docs</a>
          <DesktopToggle />
          <ThemeToggle />
          <LogoutButton />
        </div>
      </nav>
      <Toasts toasts={note.toasts} dismiss={note.dismiss} />
      <main className="main">
        <Routes>
          <Route path="/" element={calibrated === false && !data?.received ? <Navigate to="/setup" replace /> : <OverviewPage />} />
          <Route path="/setup" element={<Setup />} />
          <Route path="/logs" element={<Logs />} />
          <Route path="/templates" element={<Templates />} />
          <Route path="/query" element={<Query />} />
          <Route path="/stats" element={<StatsPage />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/alerts/:id" element={<IncidentPage />} />
          <Route path="/maintenance" element={<Maintenance />} />
          <Route path="/warnings" element={<Navigate to="/maintenance" replace />} />
          <Route path="/review/:id" element={<ReviewPage />} />
          <Route path="/streams" element={<Streams />} />
          <Route path="/jobs" element={<Jobs />} />
          <Route path="/agents" element={<Agents />} />
          <Route path="/agents/:host" element={<AgentDetail />} />
          <Route path="/keys" element={<Keys />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
