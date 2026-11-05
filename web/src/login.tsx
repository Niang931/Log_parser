// The admin password: a login screen in front of the whole UI, log out, and change password.
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState, type ReactNode } from "react";
import { api, ApiError } from "./api";
import { Card } from "./ui";

type AuthStatus = { required: boolean; authenticated: boolean };

export function LoginGate({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const [status, setStatus] = useState<AuthStatus | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const check = () => api<AuthStatus>("/api/auth/status").then(setStatus, (e: Error) => setFailed(e.message));
  useEffect(() => {
    check();
    const out = () => { qc.clear(); check(); };
    window.addEventListener("logpipe:logged-out", out);
    return () => window.removeEventListener("logpipe:logged-out", out);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  if (failed) return <div className="login"><p>Can't reach the server: {failed}</p></div>;
  if (!status) return null;
  return status.authenticated ? <>{children}</> : <Login onDone={check} />;
}

function Login({ onDone }: { onDone: () => void }) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
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
        <h1>logpipe</h1>
        <p className="muted small">Enter the admin password. The server printed it on first start; set LOGPIPE_ADMIN_PASSWORD and restart to reset it.</p>
        <label className="stack">
          Password
          <input type="password" autoComplete="current-password" autoFocus required value={password}
            onChange={(e) => setPassword(e.target.value)} />
        </label>
        {error && <p className="error-text" role="alert">{error}</p>}
        <button className="btn primary" disabled={busy || !password}>{busy ? "Checking…" : "Log in"}</button>
      </form>
    </main>
  );
}

export function LogoutButton() {
  const [required, setRequired] = useState(false);
  useEffect(() => { api<AuthStatus>("/api/auth/status").then((s) => setRequired(s.required), () => {}); }, []);
  if (!required) return null;
  return (
    <button className="btn small ghost" onClick={() => api("/api/auth/logout", { method: "POST" })
      .finally(() => window.dispatchEvent(new Event("logpipe:logged-out")))}>Log out</button>
  );
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
