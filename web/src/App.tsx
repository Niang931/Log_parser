import { useEffect, useRef, useState, type ReactNode } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { usePoll, useServerEvents, type Overview } from "./api";
import { logout, usePasswordRequired } from "./login";
import { LogoMark } from "./logo";
import { Toasts, useNotifications } from "./notify";
import AgentDetail from "./pages/AgentDetail";
import Bench from "./pages/Bench";
import Agents from "./pages/Agents";
import Alerts, { IncidentPage } from "./pages/Alerts";
import Jobs from "./pages/Jobs";
import Logs from "./pages/Logs";
import OverviewPage from "./pages/Overview";
import Query from "./pages/Query";
import Settings from "./pages/Settings";
import Setup from "./pages/Setup";
import { Shortcuts } from "./shortcuts";
import StatsPage from "./pages/Stats";
import Streams from "./pages/Streams";
import SystemPage from "./pages/System";
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
  bench: "M12 14l4-4M4 18a8 8 0 1116 0",
  jobs: "M12 7v5l3 3M12 3a9 9 0 100 18 9 9 0 000-18z",
  demo: "M7 4l12 8-12 8V4z",
  system: "M3 12h4l3-8 4 16 3-8h4",
  settings: "M12 15a3 3 0 100-6 3 3 0 000 6zM19.4 15a1.7 1.7 0 00.3 1.8l.1.1a2 2 0 11-2.8 2.8l-.1-.1a1.7 1.7 0 00-1.8-.3 1.7 1.7 0 00-1 1.5V21a2 2 0 11-4 0v-.1a1.7 1.7 0 00-1.1-1.5 1.7 1.7 0 00-1.8.3l-.1.1a2 2 0 11-2.8-2.8l.1-.1a1.7 1.7 0 00.3-1.8 1.7 1.7 0 00-1.5-1H3a2 2 0 110-4h.1a1.7 1.7 0 001.5-1.1 1.7 1.7 0 00-.3-1.8l-.1-.1a2 2 0 112.8-2.8l.1.1a1.7 1.7 0 001.8.3H9a1.7 1.7 0 001-1.5V3a2 2 0 114 0v.1a1.7 1.7 0 001 1.5 1.7 1.7 0 001.8-.3l.1-.1a2 2 0 112.8 2.8l-.1.1a1.7 1.7 0 00-.3 1.8V9a1.7 1.7 0 001.5 1H21a2 2 0 110 4h-.1a1.7 1.7 0 00-1.5 1z",
  keyboard: "M3 7h18v10H3zM7 11h.01M11 11h.01M15 11h.01M8 14h8",
  docs: "M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h6",
  logout: "M15 17l5-5-5-5M20 12H9M12 21H5a1 1 0 01-1-1V4a1 1 0 011-1h7",
  more: "M7 15l5-5 5 5",
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

/** The sidebar's footer: one button opening Settings, docs, shortcuts and log out. */
function AccountMenu() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const required = usePasswordRequired();
  const location = useLocation();
  useEffect(() => setOpen(false), [location.pathname]);
  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => ref.current?.contains(e.target as Node) || setOpen(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("mousedown", away); document.removeEventListener("keydown", esc); };
  }, [open]);
  return (
    <div className="account" ref={ref}>
      {open && (
        <div className="account-menu" role="menu">
          <NavLink to="/settings" role="menuitem"><Icon name="settings" />Settings</NavLink>
          <button type="button" role="menuitem" onClick={() => { setOpen(false); window.dispatchEvent(new Event("logpipe:shortcuts")); }}>
            <Icon name="keyboard" />Keyboard shortcuts<kbd>?</kbd></button>
          <a href="/api/docs" role="menuitem"><Icon name="docs" />API docs</a>
          <a href="/demo/" role="menuitem"><Icon name="demo" />Developer demo</a>
          {required && <><hr /><button type="button" role="menuitem" onClick={logout}><Icon name="logout" />Log out</button></>}
        </div>
      )}
      <button type="button" className="account-btn" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span className="avatar" aria-hidden="true">A</span>
        <span className="grow">Admin</span>
        <Icon name="more" />
      </button>
    </div>
  );
}

export default function App() {
  const { data } = usePoll<Overview>("/api/overview", 3000);
  useServerEvents();
  // What needs a person: templates to review and models with templates they haven't seen.
  const todo = (data?.review ?? 0) + (data?.warnings.novel_template ?? 0);
  const note = useNotifications(data ? todo : undefined);
  const [open, setOpen] = useState(false);
  const location = useLocation();
  useEffect(() => setOpen(false), [location.pathname]);
  const calibrated = data?.settings.calibrated;

  return (
    <div className="shell">
      <div className="mobile-bar">
        <button className="btn small" onClick={() => setOpen(!open)} aria-expanded={open}>Menu</button>
        <strong>logpipe</strong>
      </div>
      <nav className={`nav ${open ? "open" : ""}`} aria-label="Main">
        <div className="brand">
          <span className="brand-mark"><LogoMark /></span>
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
        <Item to="/system" icon="system" count={note.firingMetrics}>System</Item>
        <div className="nav-group">Fleet</div>
        <Item to="/agents" icon="agents">Agents</Item>
        <Item to="/bench" icon="bench">Benchmark</Item>
        {calibrated && <Item to="/setup" icon="setup">Setup guide</Item>}
        <AccountMenu />
      </nav>
      <Toasts toasts={note.toasts} dismiss={note.dismiss} />
      <Shortcuts />
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
          <Route path="/system" element={<SystemPage />} />
          <Route path="/agents" element={<Agents />} />
          <Route path="/agents/:host" element={<AgentDetail />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/keys" element={<Navigate to="/settings#keys" replace />} />
          <Route path="/bench" element={<Bench />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
