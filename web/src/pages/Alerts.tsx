import { Link } from "react-router-dom";
import { usePoll, type Alert } from "../api";
import { Card, Empty, Loading, PageHead, time } from "../ui";

interface Resp { alerts: Alert[]; streams: Record<string, { miss_threshold: number; version: string }> }

const REASONS: Record<string, string> = {
  sequence: "the order of events is unlike anything the model learned",
  hypersphere: "the window sits far from normal behaviour",
};

export default function Alerts() {
  const { data, error } = usePoll<Resp>("/api/alerts", 5000);
  return (
    <>
      <PageHead title="Anomaly alerts"
        sub={<>Windows of consecutive events a stream's LogBERT model finds unusual. Templates a model has never seen are not alerts: they are listed under <Link to="/warnings">Warnings</Link>.</>} />
      {!data ? <Loading error={error} /> : (
        <div className="stack">
          {Object.keys(data.streams).length === 0 && (
            <div className="callout info"><div>No stream has a model yet. Models are trained by calibration, or from <Link to="/streams">Streams</Link>.</div></div>
          )}
          <Card>
            {data.alerts.length === 0 ? <Empty title="No alerts">Nothing unusual so far.</Empty> : (
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Window</th><th>Host / service</th><th>Why</th><th className="right">Missed</th><th>Model</th><th>Templates</th></tr></thead>
                  <tbody>
                    {data.alerts.map((a) => (
                      <tr key={a.alert_id}>
                        <td className="small nowrap num">{time(a.window_start)}<br /><span className="muted">to {time(a.window_end)}</span></td>
                        <td><Link to={`/logs?host=${encodeURIComponent(a.key.split("/")[0])}`}>{a.key}</Link></td>
                        <td>{a.reasons.map((r) => <div key={r}><span className="badge bad">{r}</span> <span className="small ink-2">{REASONS[r] ?? ""}</span></div>)}</td>
                        <td className="right num">{Math.round(a.miss_ratio * 100)}%</td>
                        <td className="small mono">{a.model_version}</td>
                        <td className="small mono">{a.template_ids.join(" ")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      )}
    </>
  );
}
