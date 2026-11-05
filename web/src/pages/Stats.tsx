import { useNavigate } from "react-router-dom";
import { usePoll, type Facets, type Histogram, type Stats as S } from "../api";
import { LEVELS, PARSE, Timeline, TimeRange, Toplist, useRange } from "../charts";
import { Card, Loading, PageHead } from "../ui";

export default function Stats() {
  const [range, minutes, setRange] = useRange("1h");
  const nav = useNavigate();
  const { data, error } = usePoll<S>(`/api/stats?minutes=${minutes}`, 5000);
  const byLevel = usePoll<Histogram>(`/api/logs/histogram?minutes=${minutes}&by=level`, 5000).data;
  const byParse = usePoll<Histogram>(`/api/logs/histogram?minutes=${minutes}&by=status`, 5000).data;
  const errors = usePoll<Facets>(`/api/logs/facets?minutes=${minutes}&level=error`, 5000).data;
  const explore = (k: string) => (v: string) => nav(`/logs?range=${range}&${k}=${encodeURIComponent(v)}`);

  return (
    <>
      <PageHead title="Analytics" sub="Volume over time and who sends the most. Click a row to open those logs."
        actions={<TimeRange value={range} onChange={setRange} />} />
      <div className="dash">
        <Card title="Log volume by status" className="w-8"><Timeline data={byLevel} series={[...LEVELS].reverse()} height={220} /></Card>
        <Card title="Errors by service" className="w-4">
          {errors ? <Toplist rows={errors.service.map(([s, n]) => [s || "(none)", n, s])} onPick={(s) => nav(`/logs?range=${range}&level=error&service=${encodeURIComponent(s)}`)} color="var(--lvl-error-soft)" unit="errors" /> : <Loading />}
        </Card>
        <Card title="Parsing coverage" hint="lines matched to a template vs still waiting to be learned" className="w-12">
          <Timeline data={byParse} series={PARSE} height={160} />
        </Card>
        {!data ? <div className="w-12"><Loading error={error} /></div> : (
          <>
            <Card title="Top hosts" className="w-4"><Toplist rows={data.hosts.map(([h, n]) => [h, n, h])} onPick={explore("host")} /></Card>
            <Card title="Top services" className="w-4"><Toplist rows={data.services.map(([s, n]) => [s || "(none)", n, s])} onPick={explore("service")} /></Card>
            <Card title="Top sources" className="w-4"><Toplist rows={data.sources.map(([s, n]) => [s || "(none)", n, s])} onPick={explore("source")} /></Card>
            <Card title="Most frequent templates" className="w-12">
              <Toplist rows={data.templates.map(([id, n, text]) => [<span className="mono" title={text}><span className="muted">#{id}</span> {text}</span>, n, String(id)])} onPick={explore("template_id")} />
            </Card>
          </>
        )}
      </div>
    </>
  );
}
