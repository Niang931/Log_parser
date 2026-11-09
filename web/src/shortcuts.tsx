// Keyboard shortcuts: "g" then a letter jumps to a page, "/" focuses the page's search box,
// "?" lists them. Ignored while typing in a field or with Ctrl/Cmd/Alt held.
import { Fragment, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

export const GO: [string, string, string][] = [
  ["o", "/", "Overview"], ["l", "/logs", "Log Explorer"], ["t", "/templates", "Templates"], ["q", "/query", "Query"],
  ["k", "/query?tab=ask", "Ask a question"],
  ["n", "/stats", "Analytics"], ["a", "/alerts", "Anomaly alerts"], ["m", "/maintenance", "Maintenance"],
  ["r", "/streams", "Streams & models"], ["j", "/jobs", "Jobs"], ["y", "/system", "System"],
  ["f", "/agents", "Agents"], ["b", "/bench", "Benchmark"], ["s", "/settings", "Settings"],
];
export const OTHER: [string, string][] = [["/", "Search on this page"], ["?", "Show these shortcuts"], ["Esc", "Close a dialog or menu"]];

const typing = (t: EventTarget | null) =>
  t instanceof HTMLElement && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName));

export function ShortcutList() {
  return (
    <div className="grid cols-2" style={{ gap: 8 }}>
      <dl className="kv">{GO.map(([k, , label]) => <Fragment key={k}><dt><kbd>g</kbd> <kbd>{k}</kbd></dt><dd>{label}</dd></Fragment>)}</dl>
      <dl className="kv" style={{ alignContent: "start" }}>{OTHER.map(([k, label]) => <Fragment key={k}><dt><kbd>{k}</kbd></dt><dd>{label}</dd></Fragment>)}</dl>
    </div>
  );
}

export function Shortcuts() {
  const navigate = useNavigate();
  const pending = useRef(0);  // when "g" was pressed; the next key within 1.5 s picks the page
  const dialog = useRef<HTMLDialogElement>(null);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const show = () => setOpen(true);
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey || typing(e.target)) return;
      if (Date.now() - pending.current < 1500) {
        pending.current = 0;
        const hit = GO.find(([k]) => k === e.key);
        if (hit) { e.preventDefault(); navigate(hit[1]); }
        return;
      }
      if (e.key === "g") pending.current = Date.now();
      else if (e.key === "?") { e.preventDefault(); setOpen((o) => !o); }
      else if (e.key === "/") {
        const box = document.querySelector<HTMLInputElement>("main input[type=search]");
        if (box) { e.preventDefault(); box.focus(); box.select(); }
      }
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("logpipe:shortcuts", show);
    return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("logpipe:shortcuts", show); };
  }, [navigate]);
  useEffect(() => {
    const d = dialog.current;
    if (open && d && !d.open) d.showModal();
    if (!open && d?.open) d.close();
  }, [open]);
  return (
    <dialog ref={dialog} className="dialog" onClose={() => setOpen(false)} onClick={(e) => e.target === dialog.current && setOpen(false)}>
      <div className="spread" style={{ marginBottom: 12 }}>
        <h2>Keyboard shortcuts</h2>
        <button className="btn small ghost" onClick={() => setOpen(false)} aria-label="Close">×</button>
      </div>
      <ShortcutList />
    </dialog>
  );
}
