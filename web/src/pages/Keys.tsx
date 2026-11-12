import { useState } from "react";
import { api, useAction, usePoll, type ApiKey, type KeyScope } from "../api";
import { CopyBlock, Empty, Loading, TimeAgo } from "../ui";

/** API keys for agents shipping logs and AI assistants using /mcp: create (shown once), list, revoke. Lives in Settings → API keys. */
export default function Keys() {
  const { data, error } = usePoll<{ keys: ApiKey[] }>("/api/keys", 10000);
  const [name, setName] = useState("");
  const [scope, setScope] = useState<KeyScope>("intake");
  const [token, setToken] = useState<string | null>(null);
  const create = useAction((n: string) => api<{ token: string }>("/api/keys", { method: "POST", json: { name: n, scope } }));
  const revoke = useAction((id: string) => api(`/api/keys/${id}`, { method: "DELETE" }));
  return (
    <div className="stack">
      <div>
          <form className="row" onSubmit={(e) => { e.preventDefault(); create.mutate(name, { onSuccess: (r) => { setToken(r.token); setName(""); } }); }}>
            <input placeholder="Name, e.g. prod gateways" value={name} onChange={(e) => setName(e.target.value)} aria-label="Key name" required />
            <select value={scope} onChange={(e) => setScope(e.target.value as KeyScope)} aria-label="Key use">
              <option value="intake">Agent: ships logs</option>
              <option value="mcp">AI assistant: reads via /mcp</option>
            </select>
            <button className="btn primary" disabled={create.isPending}>Create</button>
          </form>
          {token && <div style={{ marginTop: 12 }}><p><strong>Copy it now</strong>: it won't be shown again.</p><CopyBlock text={token} /></div>}
      </div>
      <div>
          {!data ? <Loading error={error} /> : data.keys.length === 0 ? <Empty title="No keys yet" /> : (
            <div className="table-wrap">
              <table>
                <thead><tr><th>Name</th><th>Use</th><th>Key</th><th>Created</th><th>Last used</th><th /></tr></thead>
                <tbody>
                  {data.keys.map((k) => (
                    <tr key={k.id}>
                      <td>{k.name} {k.revoked && <span className="badge bad">revoked</span>}</td>
                      <td><span className="badge">{k.scope === "mcp" ? "AI assistant" : "agent"}</span></td>
                      <td className="mono small">{k.prefix}…</td>
                      <td className="small"><TimeAgo iso={k.created_at} /></td>
                      <td className="small"><TimeAgo iso={k.last_used_at} /></td>
                      <td className="right">{!k.revoked && (
                        <button className="btn small ghost danger" onClick={() => confirm(`Revoke ${k.name}? Whatever uses it loses access.`) && revoke.mutate(k.id)}>Revoke</button>
                      )}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
      </div>
    </div>
  );
}
