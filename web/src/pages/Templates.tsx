import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { usePoll, type Template } from "../api";
import { Card, CopyBlock, Drawer, Empty, Loading, PageHead, TemplateText, TimeAgo, fmt } from "../ui";

interface Resp { version: number; templates: Template[]; pending: number; engine: string; masks: { label: string }[] }

export default function Templates() {
  const { data, error } = usePoll<Resp>("/api/templates", 5000);
  const [params, setParams] = useSearchParams();
  const [q, setQ] = useState("");
  const [only, setOnly] = useState<"all" | "unseen" | "aliases">("all");
  const openId = params.get("id");

  if (!data) return <><PageHead title="Templates" /><Loading error={error} /></>;
  const rows = data.templates
    .filter((t) => !q || t.text.toLowerCase().includes(q.toLowerCase()) || String(t.template_id) === q)
    .filter((t) => only === "all" || (only === "unseen" ? t.seen_by.length === 0 : t.alias_of !== null))
    .sort((a, b) => b.count - a.count);
  const open = data.templates.find((t) => String(t.template_id) === openId);

  return (
    <>
      <PageHead title="Templates"
        sub={<>Log shapes learned by the parser (registry v{data.version}, {data.masks.length} masks). Highlighted slots are the fields extracted from each line. {fmt(data.pending)} lines are still pending.</>} />
      <Card>
        <div className="filters">
          <input type="search" placeholder="Search template text or #id" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search templates" />
          <select aria-label="Show" value={only} onChange={(e) => setOnly(e.target.value as typeof only)}>
            <option value="all">All templates</option>
            <option value="unseen">Unseen by any model</option>
            <option value="aliases">Aliases</option>
          </select>
        </div>
        {rows.length === 0 ? (
          <Empty title={data.templates.length ? "No template matches" : "No templates yet"}>
            {!data.templates.length && <>Templates appear once you approve them in the <Link to="/setup">setup guide</Link>.</>}
          </Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>#</th><th style={{ width: "60%" }}>Template</th><th className="right">Lines</th><th>Models</th><th>Last seen</th></tr></thead>
              <tbody>
                {rows.map((t) => (
                  <tr key={t.template_id} className="clickable" tabIndex={0}
                    onClick={() => setParams({ id: String(t.template_id) })}
                    onKeyDown={(e) => e.key === "Enter" && setParams({ id: String(t.template_id) })}>
                    <td className="num muted">{t.template_id}</td>
                    <td>
                      <TemplateText tokens={t.tokens} keys={t.keys} />
                      {t.alias_of !== null && <span className="badge" style={{ marginLeft: 6 }}>alias of #{t.alias_of}</span>}
                    </td>
                    <td className="right num">{fmt(t.count)}</td>
                    <td>{t.seen_by.length ? <span className="badge good">{t.seen_by.length} stream{t.seen_by.length > 1 ? "s" : ""}</span>
                      : <span className="badge warn" title="No LogBERT model has seen this template; it is reported as novelty, not as an anomaly">unseen</span>}</td>
                    <td className="small nowrap"><TimeAgo iso={t.last_seen} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
      {open && (
        <Drawer onClose={() => setParams({})}>
          <div className="stack">
            <h2>Template #{open.template_id}</h2>
            <div className="code-block"><TemplateText tokens={open.tokens} keys={open.keys} /></div>
            <dl className="kv">
              <dt>Fields</dt><dd>{open.keys.length ? open.keys.join(", ") : "none"}</dd>
              <dt>Lines</dt><dd>{open.count.toLocaleString()}</dd>
              <dt>First seen</dt><dd><TimeAgo iso={open.first_seen} /></dd>
              <dt>Known to</dt><dd>{open.seen_by.join(", ") || "no model yet (novelty signal)"}</dd>
              {open.alias_of !== null && <><dt>Alias of</dt><dd>#{open.alias_of}</dd></>}
            </dl>
            <div className="row">
              <Link className="btn" to={`/logs?template_id=${open.template_id}`}>View its lines</Link>
              <Link className="btn" to={`/query?sql=${encodeURIComponent(open.sql)}`}>Open in Query</Link>
            </div>
            <div><h3>As a table ({data.engine})</h3><CopyBlock text={open.sql} /></div>
          </div>
        </Drawer>
      )}
    </>
  );
}
