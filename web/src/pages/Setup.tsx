// The setup guide, one step per screen: key -> agents -> sources -> sample -> streams ->
// review templates -> train. The stepper shows where you are; finished steps can be revisited.
import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { api, useAction, usePoll, type Batch, type Job, type Settings, type Source, type Stream } from "../api";
import { Card, JobRow, Loading, Meter, PageHead, Stepper, TimeAgo, fmt, sourceLabel } from "../ui";
import { InstallStep, KeyStep } from "./AddAgent";
import { ReviewBatch } from "./Review";
import { GroupsCard, StreamsEditor } from "./Streams";

interface SetupState {
  keys: number;
  agents: { hostname: string; online: boolean; sources: number }[];
  sources: Source[];
  settings: Settings;
  streams: Stream[];
  jobs: Job[];
  min_lines: number;
  review: Batch | null;
}

const STEPS = ["API key", "Agents", "Sources", "Sample", "Streams & groups", "Review", "Train"];

function Screen({ n, title, intro, children }: { n: number; title: string; intro: ReactNode; children: ReactNode }) {
  return (
    <>
      <div className="wizard-head">
        <span className="muted small">Step {n + 1} of {STEPS.length}</span>
        <h2>{title}</h2>
        <p className="ink-2">{intro}</p>
      </div>
      <div className="wizard-body">{children}</div>
    </>
  );
}

const active = (j?: Job) => !!j && (j.status === "queued" || j.status === "running");

export default function Setup() {
  const { data, error } = usePoll<SetupState>("/api/setup", 3000);
  const [picked, setPicked] = useState<number | null>(null);
  const [token, setToken] = useState("");
  const [cap, setCap] = useState<number | null>(null);
  const [streamsSeen, setStreamsSeen] = useState(false);
  const generate = useAction((sample_cap: number) => api("/api/review/generate", { method: "POST", json: { sample_cap } }));

  if (!data) return <><PageHead title="Setup guide" /><Loading error={error} /></>;
  const calibrated = data.settings.calibrated;
  const total = data.sources.reduce((a, s) => a + s.count, 0);
  const last = (kind: Job["kind"]) => [...data.jobs].reverse().find((j) => j.kind === kind);
  const draftJob = last("draft");
  const applyJob = last("apply");
  const done = [
    data.keys > 0,
    data.agents.length > 0,
    total >= data.min_lines,
    !!data.review || calibrated,
    streamsSeen || calibrated,
    calibrated,
    calibrated && !active(applyJob) && data.streams.every((s) => s.model_version || s.waiting),
  ];
  const first = done.findIndex((d) => !d);
  // Opens at the first unfinished step, then only moves when you do: a step finishing in the
  // background (a key created, an agent connecting) never yanks the screen away.
  const step = picked ?? (first === -1 ? STEPS.length - 1 : first);
  if (picked === null) setPicked(step);
  const go = (i: number) => setPicked(Math.max(0, Math.min(STEPS.length - 1, i)));
  const sampleCap = cap ?? data.settings.sample_cap;
  const canNext = done[step] || step === 4;

  return (
    <>
      <PageHead title="Setup guide"
        sub="Connect your hosts, let them collect logs, review the templates drafted from them, then train the anomaly models." />
      {first === -1 && (
        <div className="callout good" style={{ marginBottom: 16 }}>
          <div className="grow"><strong>Setup complete</strong> <TimeAgo iso={data.settings.calibrated_at} />. New sources are drafted for review by themselves under <Link to="/maintenance">Maintenance</Link>.</div>
          <Link className="btn small primary" to="/">Go to overview</Link>
        </div>
      )}
      <Card className="wizard">
        <Stepper steps={STEPS.map((label, i) => ({ label, done: done[i] }))} current={step} onPick={go} />

        {step === 0 && (
          <Screen n={0} title="Create an API key for your agents"
            intro={data.keys > 0 ? `You already have ${data.keys} active key${data.keys > 1 ? "s" : ""}. Create another, or continue if you have one at hand.` : "Agents send this key with every request so only your hosts can ship logs."}>
            <KeyStep token={token} onToken={setToken} />
          </Screen>
        )}

        {step === 1 && (
          <Screen n={1} title="Install the agent on each host" intro="Run the command on every host whose logs you want. Each one shows up below as soon as it connects.">
            <InstallStep token={token} />
            <h3 style={{ marginTop: 16 }}>Connected agents</h3>
            {data.agents.length === 0 ? (
              <div className="waiting" role="status"><span className="spin" aria-hidden="true" /><div><strong>Waiting for the first agent…</strong></div></div>
            ) : (
              <div className="row">
                {data.agents.map((a) => (
                  <Link key={a.hostname} to={`/agents/${encodeURIComponent(a.hostname)}`} className="badge">
                    <span className={`dot ${a.online ? "good" : "off"}`} />{a.hostname} · {a.sources} sources
                  </Link>
                ))}
              </div>
            )}
          </Screen>
        )}

        {step === 2 && (
          <Screen n={2} title="Track the log files" intro={<>Pick the files on each host: open an agent under <Link to="/agents">Agents</Link> and push sources to it, or run <code>logpipe-agent track '/var/log/app/*.log' --service app</code> on the host. Let them collect lines for a while; more lines give better templates.</>}>
            <Meter have={total} need={data.min_lines} unit="lines (minimum to continue)" />
            {data.sources.length > 0 && (
              <div className="table-wrap" style={{ marginTop: 12 }}>
                <table>
                  <thead><tr><th>Source</th><th>Type</th><th className="right">Lines so far</th><th>First seen</th></tr></thead>
                  <tbody>
                    {data.sources.map((s) => (
                      <tr key={s.id}>
                        <td>{sourceLabel(s.id)}</td>
                        <td className="muted">{s.source || "–"}</td>
                        <td className="right num">{fmt(s.count)}</td>
                        <td><TimeAgo iso={s.first_seen} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Screen>
        )}

        {step === 3 && (
          <Screen n={3} title="Draft templates from a sample"
            intro="Each source contributes up to this many recent lines; priority sampling picks the most varied ones, so one chatty source can't drown out the rest. Nothing reaches the parser yet: you review the drafts next.">
            {calibrated ? <div className="callout good"><div>Templates are already applied.</div></div> : (
              <form className="row" onSubmit={(e) => { e.preventDefault(); generate.mutate(sampleCap); }}>
                <label className="field">
                  Lines per source
                  <input type="number" min={100} max={200000} step={100} value={sampleCap}
                    onChange={(e) => setCap(Number(e.target.value))} style={{ width: 140 }} />
                </label>
                <button className="btn primary" style={{ alignSelf: "flex-end" }}
                  disabled={total < data.min_lines || active(draftJob) || generate.isPending}>
                  {data.review ? "Draft again" : "Draft templates"}
                </button>
                {total < data.min_lines && <span className="muted small" style={{ alignSelf: "flex-end" }}>Needs {data.min_lines} lines; {total} so far.</span>}
              </form>
            )}
            {generate.error && <div className="error-text">{generate.error.message}</div>}
            {draftJob && <div style={{ marginTop: 12 }}><JobRow job={draftJob} /></div>}
            {data.review && !active(draftJob) && <div className="callout good" style={{ marginTop: 12 }}><div className="grow">{data.review.drafts as number} templates drafted.</div><button className="btn small primary" onClick={() => go(4)}>Continue</button></div>}
          </Screen>
        )}

        {step === 4 && (
          <Screen n={4} title="Streams and groups (optional)"
            intro={<>Each stream trains its own anomaly model on the sources in it. Every source starts with a stream of its own, and a new one joins the stream of a source like it (same service). Tick sources that behave alike, such as IoT gateways, into one stream. You can change both any time under Streams.</>}>
            <h3 style={{ margin: "0 0 8px" }}>Streams</h3>
            <StreamsEditor compact />
            <p className="muted small" style={{ margin: "20px 0 8px", maxWidth: "75ch" }}>A group puts the streams of one system together, e.g. <em>mail</em> = the ldap and proxy streams, <em>iot</em> = cameras and sensors.
              Ask then covers all of a group's services when you name it ("why is mail down?"), and the LLM is told which services depend on each other.</p>
            <GroupsCard />
          </Screen>
        )}

        {step === 5 && (
          <Screen n={5} title="Review the drafted templates"
            intro="Approve the ones that describe your logs, fix wrong variables or field names, merge duplicates and reject noise. Only approved templates go into the parser.">
            {calibrated ? <div className="callout good"><div>Templates applied <TimeAgo iso={data.settings.calibrated_at} />. Review later additions under <Link to="/maintenance">Maintenance</Link>.</div></div>
              : data.review ? <ReviewBatch batchId={data.review.id} onApplied={() => go(6)} />
              : active(draftJob) ? <JobRow job={draftJob!} />
              : <div className="muted">Draft templates first (step 4).</div>}
          </Screen>
        )}

        {step === 6 && (
          <Screen n={6} title="Train the anomaly models"
            intro="Stored lines are back-filled with the new templates, then each stream's model trains. A stream without enough parsed lines waits and trains by itself once it has them.">
            {applyJob ? <JobRow job={applyJob} /> : <div className="muted">Apply the reviewed templates first (step 6).</div>}
            {calibrated && (
              <ul className="todos" style={{ marginTop: 12 }}>
                {data.streams.map((s) => (
                  <li key={s.id} className="todo">
                    <span className={`badge ${s.model_version ? "good" : s.waiting ? "warn" : ""}`}>{s.model_version ? "Trained" : s.waiting ? "Waiting for data" : "Queued"}</span>
                    <div className="todo-body">
                      <div className="todo-title">{s.name}</div>
                      {s.waiting && s.data && <Meter have={s.data.have} need={s.data.need} unit="windows" />}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </Screen>
        )}

        <div className="wizard-foot">
          {step > 0 ? <button className="btn" onClick={() => go(step - 1)}>Back</button> : <span />}
          {step < STEPS.length - 1 && (
            <button className="btn primary" disabled={!canNext} onClick={() => { if (step === 4) setStreamsSeen(true); go(step + 1); }}
              title={canNext ? "" : "Finish this step first"}>
              Next
            </button>
          )}
        </div>
      </Card>
    </>
  );
}
