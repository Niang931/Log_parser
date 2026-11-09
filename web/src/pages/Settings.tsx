// One place for everything you set rather than look at: this browser's preferences, access
// (password, agent keys), how the server learns, alert thresholds, and what it is running.
import { useEffect, useState, type ReactNode } from "react";
import { useLocation } from "react-router-dom";
import { api, useAction, usePoll, type Settings as ServerSettings } from "../api";
import { ChangePassword, logout, usePasswordRequired } from "../login";
import { DesktopToggle } from "../notify";
import { ShortcutList } from "../shortcuts";
import { getTheme, setTheme, type Theme } from "../theme";
import { Card, Loading, PageHead, Segmented, TimeAgo } from "../ui";
import Keys from "./Keys";
import { RulesEditor, type SystemData } from "./System";

const SECTIONS: [string, string][] = [
  ["appearance", "Appearance"], ["notifications", "Notifications"], ["security", "Password & session"],
  ["keys", "API keys"], ["learning", "Learning"], ["alerts", "Metric alerts"], ["drift", "Model drift"],
  ["shortcuts", "Keyboard shortcuts"], ["about", "About this server"],
];

function Section({ id, title, hint, actions, children }: { id: string; title: string; hint?: ReactNode; actions?: ReactNode; children: ReactNode }) {
  return <div id={id} className="settings-section"><Card title={title} hint={hint} actions={actions}>{children}</Card></div>;
}

function Appearance() {
  const [theme, set] = useState<Theme>(getTheme);
  return (
    <div className="row">
      <span className="small ink-2">Theme</span>
      <Segmented label="Theme" value={theme} onChange={(t) => { setTheme(t); set(t); }}
        options={[["auto", "Match system"], ["light", "Light"], ["dark", "Dark"]]} />
    </div>
  );
}

function Learning({ s }: { s: ServerSettings }) {
  const [v, setV] = useState({ sample_cap: s.sample_cap, retrain_days: s.retrain_days, draft_lines: s.draft_lines, draft_after_minutes: s.draft_after_minutes });
  const save = useAction(() => api("/api/settings", { method: "PUT", json: v }));
  const dirty = (Object.keys(v) as (keyof typeof v)[]).some((k) => v[k] !== s[k]);
  const field = (k: keyof typeof v, label: string, help: string, props: object) => (
    <label className="field">{label}
      <input type="number" value={v[k]} onChange={(e) => setV({ ...v, [k]: Number(e.target.value) })} {...props} />
      <span className="muted small" style={{ fontWeight: 400 }}>{help}</span>
    </label>
  );
  return (
    <form className="stack" onSubmit={(e) => { e.preventDefault(); save.mutate(); }}>
      <div className="grid cols-2">
        {field("sample_cap", "Sample cap (lines per source)", "Most lines read from each source when drafting templates or training.", { min: 100, max: 200000, step: 100 })}
        {field("retrain_days", "Default retrain period (days)", "A stream retrains once its model is this old and it has templates the model hasn't seen.", { min: 0.01, step: "any" })}
        {field("draft_lines", "Draft templates after (lines)", "A new source gets its templates drafted for review once it has sent this many lines…", { min: 1, max: 200000 })}
        {field("draft_after_minutes", "…or after (minutes)", "…or after this long, if it has at least the minimum to learn from.", { min: 1, step: "any" })}
      </div>
      <div className="row">
        <button className="btn primary" disabled={!dirty || save.isPending}>Save</button>
        {dirty && <button type="button" className="btn ghost" onClick={() => setV({ sample_cap: s.sample_cap, retrain_days: s.retrain_days, draft_lines: s.draft_lines, draft_after_minutes: s.draft_after_minutes })}>Discard</button>}
        {save.isSuccess && !dirty && <span className="small muted" role="status">Saved</span>}
        {save.error && <span className="error-text">{save.error.message}</span>}
      </div>
    </form>
  );
}

type About = { version: string; engine: string; store?: string; queue?: string; llm?: string; started_at: string; cpus: number; password_set: boolean; metrics_days: number };

function AboutServer() {
  const { data, error } = usePoll<About>("/api/about", 60000);
  if (!data) return <Loading error={error} />;
  return (
    <dl className="kv">
      <dt>Version</dt><dd className="mono">{data.version}</dd>
      <dt>Log store</dt><dd>{data.store ?? (data.engine === "sqlite" ? "memory" : data.engine)}</dd>
      {data.queue && <><dt>Queue</dt><dd>{data.queue}</dd></>}
      {data.llm && <><dt>LLM backend</dt><dd className="mono">{data.llm}</dd></>}
      <dt>Running since</dt><dd><TimeAgo iso={data.started_at} /></dd>
      <dt>CPU cores</dt><dd>{data.cpus}</dd>
      <dt>Metrics kept</dt><dd>{data.metrics_days} days</dd>
      <dt>Admin password</dt><dd>{data.password_set ? "set" : <span className="badge warn">not set: the UI is open to anyone who can reach it</span>}</dd>
      <dt>API</dt><dd><a href="/api/docs">Interactive API docs</a> · <a href="/demo/">Developer demo</a></dd>
    </dl>
  );
}

export default function Settings() {
  const { hash } = useLocation();
  const settings = usePoll<ServerSettings>("/api/settings", 30000).data;
  const metrics = usePoll<SystemData>("/api/system?minutes=1", 60000).data?.metrics;
  const required = usePasswordRequired();
  const [active, setActive] = useState(hash.slice(1) || SECTIONS[0][0]);
  // React Router doesn't scroll to #section; do it once the section exists.
  useEffect(() => {
    const id = hash.slice(1);
    if (!id) return;
    setActive(id);
    requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView({ block: "start" }));
  }, [hash, !!settings, !!metrics]); // eslint-disable-line react-hooks/exhaustive-deps
  // Highlight the section being read: the last one whose top has scrolled near the top.
  useEffect(() => {
    const onScroll = () => {
      const line = 140;  // a little below the sticky top
      const atBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4;
      const passed = SECTIONS.filter(([id]) => (document.getElementById(id)?.getBoundingClientRect().top ?? 1e9) <= line);
      setActive(atBottom ? SECTIONS[SECTIONS.length - 1][0] : (passed.at(-1) ?? SECTIONS[0])[0]);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  return (
    <>
      <PageHead title="Settings" sub="Appearance and notifications apply to this browser; everything else applies to the server and everyone using it." />
      <div className="settings">
        <nav className="settings-nav" aria-label="Settings sections">
          {SECTIONS.map(([id, label]) => <a key={id} href={`#${id}`} className={active === id ? "on" : ""} aria-current={active === id ? "true" : undefined}>{label}</a>)}
        </nav>
        <div className="stack">
          <Section id="appearance" title="Appearance"><Appearance /></Section>
          <Section id="notifications" title="Notifications" hint="Pop-up toasts in the corner are always on; desktop notifications also reach you in another tab.">
            <DesktopToggle />
          </Section>
          <div id="security" className="settings-section stack">
            <ChangePassword />
            {required && (
              <Card title="Session" hint="Signed in on this browser for up to 7 days. Changing the password signs out every other browser.">
                <button className="btn" onClick={logout}>Log out</button>
              </Card>
            )}
          </div>
          <Section id="keys" title="API keys" hint="Agents authenticate with these. They are stored hashed: copy a new one when it is shown.">
            <Keys />
          </Section>
          <Section id="learning" title="Learning" hint="How the server drafts templates for new sources and retrains anomaly models.">
            {settings ? <Learning key={JSON.stringify(settings)} s={settings} /> : <Loading />}
          </Section>
          <Section id="alerts" title="Metric alerts" hint="Fire when a server metric stays past its threshold for the set time; shown on System, as a toast and in the sidebar.">
            {metrics ? <RulesEditor metrics={metrics} /> : <Loading />}
          </Section>
          <Section id="drift" title="Model drift" hint="PSI compares a stream's recent anomaly scores with what its model saw in training.">
            <p className="small ink-2">Rule of thumb: under 0.1 is stable, 0.1 to 0.25 is a moderate shift, over 0.25 means the model no longer
              matches the traffic and should be retrained. The rule below sets where the System page flags a stream as drifted.
              Retraining is never automatic: use Retrain on the System page.</p>
            {metrics ? <RulesEditor metrics={metrics} only={(r) => r.metric === "score_psi_max"} /> : <Loading />}
          </Section>
          <Section id="shortcuts" title="Keyboard shortcuts" hint="Press ? anywhere to see these.">
            <ShortcutList />
          </Section>
          <Section id="about" title="About this server"><AboutServer /></Section>
        </div>
      </div>
    </>
  );
}
