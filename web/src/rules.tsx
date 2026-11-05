// Datadog-style parsing rule editor: write `user %{integer:user_id} logged in from %{ip:ip}`,
// see at once what it extracts from sample lines and which of the template's lines it misses.
import { Fragment, useEffect, useRef, useState, type ReactNode } from "react";
import { usePoll, type Template } from "./api";
import { TemplateText } from "./ui";

export interface RuleCheck {
  count: number;
  checked: number;
  keys: string[];
  missed: number;
  rows: { line: string; fields: Record<string, string> | null }[];
}

const MATCHERS: [string, string][] = [
  ["word", "letters, digits, _"], ["integer", "42, -7"], ["number", "3.14, 1e3"], ["notspace", "anything up to a space"],
  ["ip", "10.0.0.1, ::1"], ["hostname", "db-1.local"], ["path", "/api/items/12"], ["uuid", "123e4567-…"],
  ["quoted", "\"…\" or '…'"], ["timestamp", "2024-05-01 10:00:00"], ["data", "anything (lazy), even spaces"],
];

export function RuleEditor({ initial, check, onSave, onCancel, saving, error, saveLabel = "Save rule" }: {
  initial: string; saving: boolean; error?: Error | null; saveLabel?: string;
  check: (rule: string) => Promise<RuleCheck>;
  onSave: (rule: string) => void; onCancel: () => void;
}) {
  const [rule, setRule] = useState(initial);
  const [result, setResult] = useState<RuleCheck | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const box = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setBusy(true);
    const t = setTimeout(() => {
      check(rule).then((r) => { setResult(r); setProblem(null); }, (e: Error) => { setResult(null); setProblem(e.message); })
        .finally(() => setBusy(false));
    }, 300);
    return () => clearTimeout(t);
  }, [rule]); // eslint-disable-line react-hooks/exhaustive-deps

  const insert = (matcher: string) => {
    const el = box.current;
    const name = matcher === "data" ? "message" : matcher;
    const snippet = `%{${matcher}:${name}}`;
    const [a, b] = el ? [el.selectionStart, el.selectionEnd] : [rule.length, rule.length];
    setRule(rule.slice(0, a) + snippet + rule.slice(b));
    // select the field name so typing renames it
    const start = a + matcher.length + 3;
    requestAnimationFrame(() => { el?.focus(); el?.setSelectionRange(start, start + name.length); });
  };

  return (
    <form className="rule-editor stack" onSubmit={(e) => { e.preventDefault(); onSave(rule); }}>
      <label className="field">
        <span>Parsing rule <span className="muted small">the whole line must match; <code>%{"{matcher:field}"}</code> extracts a field, <code>%{"{matcher}"}</code> just matches</span></span>
        <textarea ref={box} className="mono" rows={2} value={rule} spellCheck={false} onChange={(e) => setRule(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); onSave(rule); } }} />
      </label>
      <div className="matchers" aria-label="Insert a matcher">
        {MATCHERS.map(([m, hint]) => <button key={m} type="button" className="chip-btn" title={hint} onClick={() => insert(m)}>%{"{"}{m}{"}"}</button>)}
        <button type="button" className="chip-btn" title="any regex; quotes inside are escaped as \&quot;" onClick={() => setRule(`${rule} %{regex("[a-z]+"):name}`)}>%{"{regex(\"…\")}"}</button>
      </div>

      <div className={`rule-status ${problem ? "bad" : result?.missed ? "warn" : result ? "good" : ""}`} role="status">
        {problem ? <>✗ {problem}</>
          : !result ? "Checking…"
          : <>
            {result.missed ? "!" : "✓"} Matches <strong>{result.count.toLocaleString()}</strong> of {result.checked.toLocaleString()} recent lines
            {result.keys.length > 0 && <> · extracts {result.keys.map((k) => <code key={k}>{k}</code>).reduce<ReactNode[]>((a, c, i) => (i ? [...a, ", ", c] : [c]), [])}</>}
            {result.missed > 0 && <> · <strong>misses {result.missed}{result.missed >= 10 ? "+" : ""}</strong> of this template's lines (shown last)</>}
          </>}
        {busy && <span className="spin" aria-hidden="true" style={{ marginLeft: 8 }} />}
      </div>

      {result && result.rows.length > 0 && (
        <div className="table-wrap extract-table">
          <table>
            <thead><tr><th style={{ width: 20 }} /><th>Sample line</th>{result.keys.map((k) => <th key={k} className="mono">{k}</th>)}</tr></thead>
            <tbody>
              {result.rows.map((r, i) => (
                <Fragment key={i}>
                  <tr className={r.fields ? "" : "miss"}>
                    <td aria-label={r.fields ? "matched" : "not matched"}>{r.fields ? <span className="good-text">✓</span> : <span className="bad-text">✗</span>}</td>
                    <td className="mono small line">{r.line}</td>
                    {result.keys.map((k) => <td key={k} className="mono small">{r.fields ? (r.fields[k] ?? <span className="muted">–</span>) : <span className="muted">–</span>}</td>)}
                  </tr>
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {error && <div className="error-text">{error.message}</div>}
      <div className="row">
        <button className="btn small primary" disabled={saving || !!problem || busy}>{saveLabel}</button>
        <button type="button" className="btn small ghost" onClick={onCancel}>Cancel</button>
        <span className="muted small">Ctrl/⌘ + Enter saves</span>
      </div>
    </form>
  );
}

/** Search every template by text or #id and pick one (to link a template to it). */
export function TemplatePicker({ exclude, onPick, onCancel, pending, error }: {
  exclude: number; onPick: (id: number) => void; onCancel: () => void; pending: boolean; error?: Error | null;
}) {
  const all = usePoll<{ templates: Template[] }>("/api/templates", 10000).data?.templates;
  const [q, setQ] = useState("");
  const [picked, setPicked] = useState<Template | null>(null);
  const hits = (all ?? [])
    .filter((t) => t.template_id !== exclude && t.alias_of === null)
    .filter((t) => !q || t.text.toLowerCase().includes(q.toLowerCase()) || `#${t.template_id}` === q || String(t.template_id) === q)
    .sort((a, b) => b.count - a.count).slice(0, 8);
  return (
    <div className="picker stack">
      <div className="muted small">Pick the template this one is the same event as. Its lines then count as that template for the anomaly models; both keep their own parsing rule.</div>
      <input type="search" autoFocus placeholder="Search template text or #id" value={q} onChange={(e) => { setQ(e.target.value); setPicked(null); }} aria-label="Search templates" />
      <ul className="picker-list" role="listbox">
        {!all ? <li className="muted small">Loading…</li> : hits.length === 0 ? <li className="muted small">No template matches.</li> : hits.map((t) => (
          <li key={t.template_id}>
            <button type="button" role="option" aria-selected={picked?.template_id === t.template_id} className={picked?.template_id === t.template_id ? "on" : ""} onClick={() => setPicked(t)}>
              <span className="muted num">#{t.template_id}</span>
              <span className="grow"><TemplateText tokens={t.tokens} keys={t.keys} /></span>
              <span className="muted small num">{t.count.toLocaleString()} lines</span>
            </button>
          </li>
        ))}
      </ul>
      <div className="row">
        <button className="btn small primary" disabled={!picked || pending} onClick={() => picked && onPick(picked.template_id)}>
          {picked ? `Link to #${picked.template_id}` : "Pick a template"}
        </button>
        <button className="btn small ghost" onClick={onCancel}>Cancel</button>
        {error && <span className="error-text">{error.message}</span>}
      </div>
    </div>
  );
}
