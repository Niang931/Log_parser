// The admin password: a login screen in front of the whole UI, log out, and change password.
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState, type ReactNode } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { api, ApiError } from "./api";
import { Card } from "./ui";
import { LogoMark } from "./logo";

type AuthStatus = { required: boolean; authenticated: boolean };

/** Where to go after logging in: only a path on this site (no "//evil.example" open redirect). */
export function safeNext(raw: string | null) {
  return raw && raw.startsWith("/") && !raw.startsWith("//") && !raw.startsWith("/login") ? raw : "/";
}

/** Signed out: every page redirects to /login?next=<that page>; signing in goes back there. */
export function LoginGate({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const location = useLocation();
  const [status, setStatus] = useState<AuthStatus | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const check = () => api<AuthStatus>("/api/auth/status").then((s) => { setFailed(null); setStatus(s); }, (e: Error) => setFailed(e.message));
  useEffect(() => {
    check();
    const out = () => { qc.clear(); check(); };
    window.addEventListener("logpipe:logged-out", out);
    return () => window.removeEventListener("logpipe:logged-out", out);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  if (failed) return <div className="login"><div className="card login-card stack"><p>Can't reach the server: {failed}</p><button className="btn" onClick={check}>Try again</button></div></div>;
  if (!status) return null;
  const atLogin = location.pathname === "/login";
  const next = safeNext(new URLSearchParams(location.search).get("next"));
  if (status.authenticated) return atLogin ? <Navigate to={next} replace /> : <>{children}</>;
  if (!atLogin) {
    const here = location.pathname + location.search + location.hash;
    return <Navigate to={here === "/" ? "/login" : `/login?next=${encodeURIComponent(here)}`} replace />;
  }
  return <Login onDone={check} next={next} />;
}

function Login({ onDone, next }: { onDone: () => void; next: string }) {
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => { document.title = "Log in · logpipe"; }, []);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api("/api/auth/login", { method: "POST", json: { password } });
      onDone();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Login failed");
      setBusy(false);
    }
  };
  return (
    <main className="login">
      <form className="card login-card stack" onSubmit={submit}>
        <div className="brand login-brand"><span className="brand-mark"><LogoMark /></span>logpipe</div>
        <div>
          <h1>Log in</h1>
          <p className="muted small" style={{ marginTop: 4 }}>{next !== "/" ? <>Log in to continue to <span className="mono">{next.split(/[?#]/)[0]}</span>.</> : "Enter the admin password to open the console."}</p>
        </div>
        <label className="field">
          Admin password
          <span className="password">
            <input type={show ? "text" : "password"} autoComplete="current-password" autoFocus required value={password}
              onChange={(e) => setPassword(e.target.value)} />
            <button type="button" className="btn small ghost" onClick={() => setShow(!show)} aria-pressed={show}>{show ? "Hide" : "Show"}</button>
          </span>
        </label>
        {error && <p className="error-text" role="alert">{error}</p>}
        <button className="btn primary" disabled={busy || !password}>{busy ? "Checking…" : "Log in"}</button>
        <p className="muted small" style={{ margin: 0 }}>Forgot it? Set <code>LOGPIPE_ADMIN_PASSWORD</code> on the server and restart.</p>
      </form>
    </main>
  );
}

/** Whether a password is set (with none, there is nothing to log out of). */
export function usePasswordRequired() {
  const [required, setRequired] = useState(false);
  useEffect(() => { api<AuthStatus>("/api/auth/status").then((s) => setRequired(s.required), () => {}); }, []);
  return required;
}

// A full page load to /login: drops every cached query and doesn't bounce back via ?next=.
export function logout() {
  api("/api/auth/logout", { method: "POST" }).finally(() => window.location.assign("/login"));
}

export function ChangePassword() {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await api("/api/auth/password", { method: "POST", json: { current, new: next } });
      setMsg({ ok: true, text: "Password changed. Other browsers are signed out." });
      setCurrent("");
      setNext("");
    } catch (err) {
      setMsg({ ok: false, text: err instanceof Error ? err.message : "Failed" });
    }
  };
  return (
    <Card title="Admin password" hint="Guards this UI and the /api routes; at least 8 characters.">
      <form className="row" onSubmit={submit}>
        <input type="password" autoComplete="current-password" placeholder="Current password" aria-label="Current password"
          required value={current} onChange={(e) => setCurrent(e.target.value)} />
        <input type="password" autoComplete="new-password" placeholder="New password" aria-label="New password"
          required minLength={8} value={next} onChange={(e) => setNext(e.target.value)} />
        <button className="btn primary">Change</button>
      </form>
      {msg && <p className={msg.ok ? "small" : "error-text"} role="status" style={{ marginTop: 8 }}>{msg.text}</p>}
    </Card>
  );
}
