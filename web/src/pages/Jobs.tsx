import { usePoll, type Job } from "../api";
import { Card, Empty, JobRow, Loading, PageHead, TimeAgo } from "../ui";

export default function Jobs() {
  const { data, error } = usePoll<{ jobs: Job[] }>("/api/jobs", 1500);
  return (
    <>
      <PageHead title="Jobs" sub="Calibration, recalibration and LogBERT training run one at a time in the background." />
      {!data ? <Loading error={error} /> : data.jobs.length === 0 ? <Card><Empty title="No jobs yet" /></Card> : (
        <div className="stack">
          {data.jobs.map((j) => (
            <Card key={j.id}>
              <JobRow job={j} />
              <div className="muted small" style={{ marginTop: 6 }}>
                queued <TimeAgo iso={j.created_at} />
                {j.finished_at && <> · finished <TimeAgo iso={j.finished_at} /></>}
              </div>
            </Card>
          ))}
        </div>
      )}
    </>
  );
}
