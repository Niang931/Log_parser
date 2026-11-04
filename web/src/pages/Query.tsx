import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, useAction, usePoll, type QueryResult } from "../api";
import { Card, Empty, PageHead } from "../ui";

const EXAMPLES: Record<string, [string, string]> = {
  clickhouse: [
    "Errors per service, last hour",
    "SELECT service, count() AS lines\nFROM iot_logs FINAL\nWHERE timestamp > now() - INTERVAL 1 HOUR\n  AND positionCaseInsensitive(raw_message, 'error') > 0\nGROUP BY service\nORDER BY lines DESC",
  ],
  sqlite: [
    "Errors per service",
    "SELECT service, count(*) AS lines\nFROM logs\nWHERE raw_message LIKE '%error%'\nGROUP BY service\nORDER BY lines DESC",
  ],
};

export default function Query() {
  const [params] = useSearchParams();
  const engine = usePoll<{ engine: string }>("/api/templates", 60000).data?.engine ?? "clickhouse";
  const [sql, setSql] = useState(params.get("sql") ?? "");
  const run = useAction((q: string) => api<QueryResult>("/api/query", { method: "POST", json: { sql: q } }));
  const example = EXAMPLES[engine] ?? EXAMPLES.clickhouse;
  const table = engine === "sqlite" ? "logs" : "iot_logs";

  return (
    <>
      <PageHead title="Query" sub={<>Read-only SQL over <code>{table}</code> ({engine}). Columns: timestamp, device_id (host), service, source, tags, template_id, {engine === "sqlite" ? "status" : "log_status"}, payload, raw_message. At most 500 rows.</>} />
      <div className="stack">
        <Card>
          <form className="stack" style={{ gap: 8 }} onSubmit={(e) => { e.preventDefault(); run.mutate(sql); }}>
            <textarea rows={8} value={sql} spellCheck={false} aria-label="SQL"
              placeholder={example[1]}
              onChange={(e) => setSql(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run.mutate(sql); }} />
            <div className="spread">
              <button type="button" className="btn small ghost" onClick={() => setSql(example[1])}>Example: {example[0]}</button>
              <div className="row">
                <span className="muted small">Ctrl/⌘ + Enter</span>
                <button className="btn primary" disabled={!sql.trim() || run.isPending}>{run.isPending ? "Running…" : "Run"}</button>
              </div>
            </div>
          </form>
          {run.error && <div className="callout bad" style={{ marginTop: 12 }}>{run.error.message}</div>}
        </Card>
        {run.data && (
          <Card title={`${run.data.rows.length} rows`}>
            {run.data.rows.length === 0 ? <Empty title="No rows" /> : (
              <div className="table-wrap">
                <table>
                  <thead><tr>{run.data.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
                  <tbody>
                    {run.data.rows.map((r, i) => (
                      <tr key={i}>{r.map((v, j) => (
                        <td key={j} className="mono small">{typeof v === "object" && v !== null ? JSON.stringify(v) : String(v)}</td>
                      ))}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        )}
      </div>
    </>
  );
}
