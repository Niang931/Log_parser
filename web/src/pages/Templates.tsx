import { Fragment, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, useAction, usePoll, type Template } from "../api";
import { RuleEditor, TemplatePicker, type RuleCheck } from "../rules";
import { Card, CopyBlock, Drawer, Empty, Loading, PageHead, TemplateText, TimeAgo, fmt } from "../ui";

interface Resp { version: number; templates: Template[]; pending: number; engine: string; masks: { label: string }[] }

export function LinkBadge({ t }: { t: Template }) {
  if (t.alias_of === null) return null;
  return <span className={`badge ${t.linked_by === "you" ? "info" : ""}`}
    title={t.linked_by === "you" ? "Linked by a person" : t.linked_by === "pipeline" ? `Linked automatically: ${t.link_reason ?? "only variable parts differ"}` : `Linked by the LLM's same-event check${t.link_reason ? `: ${t.link_reason}` : ""}`}>
    {t.linked_by === "you" ? "linked by you" : t.linked_by === "pipeline" ? "linked automatically" : "linked by LLM"}
  </span>;
}

function Related({ t, byId }: { t: Template; byId: Map<number, Template> }) {
  const [picking, setPicking] = useState(false);
  const link = useAction((target: number) => api(`/api/templates/${t.template_id}/link`, { method: "POST", json: { target } }));
  const unlink = useAction((id: number) => api(`/api/templates/${id}/unlink`, { method: "POST" }));
  const parent = t.alias_of !== null ? byId.get(t.alias_of) : undefined;
  const children = t.linked.map((id) => byId.get(id)).filter((x): x is Template => !!x);
  return (
    <div className="stack" style={{ gap: 8 }}>
      <h3>Related templates</h3>
      <p className="muted small" style={{ margin: 0 }}>Linked templates are the same event written differently. Anomaly models count them as one; each keeps its own rule and fields.</p>
      {parent && (
        <div className="related">
          <span className="muted small">Counts as</span>
          <Link to={`/templates?id=${parent.template_id}`} className="grow"><TemplateText tokens={parent.tokens} keys={parent.keys} /></Link>
          <LinkBadge t={t} />
          <button className="btn small ghost" disabled={unlink.isPending} onClick={() => unlink.mutate(t.template_id)}>Unlink</button>
        </div>
      )}
      {children.map((c) => (
        <div key={c.template_id} className="related">
          <span className="muted small num">#{c.template_id}</span>
          <Link to={`/templates?id=${c.template_id}`} className="grow"><TemplateText tokens={c.tokens} keys={c.keys} /></Link>
          <LinkBadge t={c} />
          <button className="btn small ghost" disabled={unlink.isPending} onClick={() => unlink.mutate(c.template_id)}>Unlink</button>
        </div>
      ))}
      {!parent && !children.length && !picking && <div className="muted small">None.</div>}
      {picking ? (
        <TemplatePicker exclude={t.template_id} pending={link.isPending} error={link.error} onCancel={() => setPicking(false)}
          onPick={(target) => link.mutate(target, { onSuccess: () => setPicking(false) })} />
      ) : !parent && (
        <button className="btn small" style={{ alignSelf: "flex-start" }} onClick={() => setPicking(true)}>Link to the same event…</button>
      )}
      {unlink.error && <div className="error-text">{unlink.error.message}</div>}
    </div>
  );
}

function RuleSection({ t }: { t: Template }) {
  const [editing, setEditing] = useState(false);
  const save = useAction((rule: string) => api<{ job?: { id: string } }>(`/api/templates/${t.template_id}/review`, { method: "POST", json: { rule } }));
  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="spread">
        <h3>Parsing rule</h3>
        {!editing && <button className="btn small" onClick={() => setEditing(true)}>Edit rule</button>}
      </div>
      {editing ? (
        <RuleEditor initial={t.rule} saving={save.isPending} error={save.error} saveLabel="Save and re-extract stored lines"
          check={(rule) => api<RuleCheck>(`/api/templates/${t.template_id}/check`, { method: "POST", json: { rule } })}
          onCancel={() => setEditing(false)} onSave={(rule) => save.mutate(rule, { onSuccess: () => setEditing(false) })} />
      ) : (
        <>
          <code className="code-block">{t.rule}</code>
          <span className="muted small">{t.rule_edited ? "Written by hand; tried before the learned templates." : "Generated from the learned template. Edit it to choose matchers and field names."}</span>
        </>
      )}
      {save.data?.job && <div className="small">Re-extracting stored lines… <Link to="/jobs">follow the job</Link></div>}
    </div>
  );
}

export default function Templates() {
  const { data, error } = usePoll<Resp>("/api/templates", 5000);
  const [params, setParams] = useSearchParams();
  const [q, setQ] = useState("");
  const [only, setOnly] = useState<"all" | "unseen" | "linked" | "rules">("all");
  const openId = params.get("id");

  if (!data) return <><PageHead title="Templates" /><Loading error={error} /></>;
  const byId = new Map(data.templates.map((t) => [t.template_id, t]));
  const match = (t: Template) => (!q || t.text.toLowerCase().includes(q.toLowerCase()) || String(t.template_id) === q.replace("#", ""))
    && (only === "all" || (only === "unseen" ? t.seen_by.length === 0 : only === "rules" ? t.rule_edited : t.alias_of !== null || t.linked.length > 0));
  // Linked templates sit under the one they count as.
  const roots = data.templates.filter((t) => t.alias_of === null && (match(t) || t.linked.some((id) => match(byId.get(id)!))))
    .sort((a, b) => b.count - a.count);
  const open = openId ? byId.get(Number(openId)) : undefined;
  const go = (id: number) => setParams({ id: String(id) });

  const row = (t: Template, child = false) => (
    <tr key={t.template_id} className={`clickable ${child ? "linked-row" : ""}`} tabIndex={0}
      onClick={() => go(t.template_id)} onKeyDown={(e) => e.key === "Enter" && go(t.template_id)}>
      <td className="num muted">{t.template_id}</td>
      <td>
        {child && <span className="link-elbow" aria-hidden="true">↳</span>}
        <TemplateText tokens={t.tokens} keys={t.keys} />
        {child && <> <LinkBadge t={t} /></>}
        {t.rule_edited && <span className="badge" style={{ marginLeft: 6 }} title={t.rule}>rule</span>}
        {!child && t.linked.length > 0 && <span className="badge info" style={{ marginLeft: 6 }}>+{t.linked.length} linked</span>}
      </td>
      <td className="right num">{fmt(t.count)}</td>
      <td>{child ? <span className="muted small">as #{t.alias_of}</span> : t.seen_by.length ? <span className="badge good">{t.seen_by.length} stream{t.seen_by.length > 1 ? "s" : ""}</span>
        : <span className="badge warn" title="No LogBERT model has seen this template; it is reported as novelty, not as an anomaly">unseen</span>}</td>
      <td className="small nowrap"><TimeAgo iso={t.last_seen} /></td>
    </tr>
  );

  return (
    <>
      <PageHead title="Templates"
        sub={<>Log shapes the parser knows (registry v{data.version}). Highlighted parts are the fields extracted from each line. Linked templates are listed under the one they count as. {fmt(data.pending)} lines are still pending.</>} />
      <Card>
        <div className="filters">
          <input type="search" placeholder="Search template text or #id" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search templates" />
          <select aria-label="Show" value={only} onChange={(e) => setOnly(e.target.value as typeof only)}>
            <option value="all">All templates</option>
            <option value="linked">Linked to another</option>
            <option value="rules">With a hand-written rule</option>
            <option value="unseen">Unseen by any model</option>
          </select>
        </div>
        {roots.length === 0 ? (
          <Empty title={data.templates.length ? "No template matches" : "No templates yet"}>
            {!data.templates.length && <>Templates appear once you approve them in the <Link to="/setup">setup guide</Link>.</>}
          </Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>#</th><th style={{ width: "60%" }}>Template</th><th className="right">Lines</th><th>Models</th><th>Last seen</th></tr></thead>
              <tbody>
                {roots.map((t) => (
                  <Fragment key={t.template_id}>
                    {row(t)}
                    {t.linked.map((id) => byId.get(id)).filter((c): c is Template => !!c).map((c) => row(c, true))}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
      {open && (
        <Drawer onClose={() => setParams({})}>
          <div className="stack">
            <h2>Template #{open.template_id} <LinkBadge t={open} /></h2>
            <div className="code-block"><TemplateText tokens={open.tokens} keys={open.keys} /></div>
            <dl className="kv">
              <dt>Fields</dt><dd>{open.keys.length ? open.keys.map((k) => <code key={k} style={{ marginRight: 6 }}>{k}</code>) : "none"}</dd>
              <dt>Lines</dt><dd>{open.count.toLocaleString()}</dd>
              <dt>First seen</dt><dd><TimeAgo iso={open.first_seen} /></dd>
              <dt>Known to</dt><dd>{open.seen_by.join(", ") || "no model yet (novelty signal)"}</dd>
            </dl>
            <div className="row">
              <Link className="btn" to={`/logs?template_id=${open.canonical}`}>View its lines</Link>
              <Link className="btn" to={`/query?sql=${encodeURIComponent(open.sql)}`}>Open in Query</Link>
            </div>
            <RuleSection key={open.template_id} t={open} />
            <Related key={`r${open.template_id}`} t={open} byId={byId} />
            <div><h3>As a table ({data.engine})</h3><CopyBlock text={open.sql} /></div>
          </div>
        </Drawer>
      )}
    </>
  );
}
