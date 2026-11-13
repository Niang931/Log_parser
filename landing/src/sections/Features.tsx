import { Feature } from '../components/Feature'
import { KeyValues, Panel } from '../components/Panel'
import { Points, Section } from '../components/Section'
import { SHOTS } from '../shots'

const NGINX_LINE = '10.0.1.191 - - [12/Nov/2026:21:56:08 +0800] "POST /api/checkout HTTP/1.1" 200 8978'
const NGINX_FIELDS: [string, string][] = [
  ['client_ip', '10.0.1.191'],
  ['timestamp', '12/Nov/2026:21:56:08 +0800'],
  ['url_path', '/api/checkout'],
  ['status_code', '200'],
  ['response_size_bytes', '8978'],
]

export function Parse() {
  return (
    <Section id="parse" route="hot" rail="first">
      <Feature
        headingId="parse-h"
        title="Turns every line into fields"
        lead="Each line comes out as a template with named fields. Lines it can’t read yet are kept, never dropped."
        shot={SHOTS.templates}
      >
        <Panel className="grid gap-3 p-5" role="img" aria-label="An nginx line and the fields logpipe extracts from it: client_ip, timestamp, url_path, status_code and response_size_bytes.">
          <code className="break-all text-muted">{NGINX_LINE}</code>
          <span className="font-sans text-xs font-medium text-route-learn">↓ template #4</span>
          <KeyValues rows={NGINX_FIELDS} keyClass="text-field" />
        </Panel>
      </Feature>
    </Section>
  )
}

const INCIDENT: [string, string][] = [
  ['Service', 'shop-api on shop-web-1'],
  ['Window', '9:56:36 to 9:57:35 PM'],
  ['Alerts', '90'],
  ['Unusual lines', '279 of 400'],
]

export function Detect() {
  return (
    <Section id="detect" route="detect" band>
      <Feature
        headingId="detect-h"
        title="Spots the lines that don’t belong"
        lead="LogBERT learns each service’s normal flow and flags what breaks it."
        shot={SHOTS.incident}
        reverse
      >
        <dl className="divide-y divide-line border-y border-line text-[15px]">
          {INCIDENT.map(([k, v]) => (
            <div key={k} className="flex justify-between gap-6 py-3">
              <dt className="text-muted">{k}</dt>
              <dd className="text-right tabular-nums">{v}</dd>
            </div>
          ))}
        </dl>
      </Feature>
    </Section>
  )
}

export function Explain() {
  return (
    <Section id="explain" route="store">
      <div className="grid gap-20 md:gap-24">
        <Feature
          headingId="explain-h"
          title="Ask why it broke. Get the lines that prove it."
          lead="Plain-language answers that cite the exact log lines. Every query is read-only."
          shot={SHOTS.rootCause}
        >
          <Points items={['The likely cause, rated by confidence', 'A timeline of what happened, line by line', 'Next steps to check']} />
        </Feature>
        <Feature
          level={3}
          title="Shows its working"
          lead="Every template the answer looked at, ranked. Each row says which factor set its score."
          shot={SHOTS.ranking}
          reverse
        >
          <Points items={[<>Score = anomaly × rarity × proximity</>, 'The earliest line in the chain is marked', 'Evidence lists everything the LLM saw, oldest first']} />
        </Feature>
      </div>
    </Section>
  )
}

export function Ask() {
  return (
    <Section id="ask" route="store">
      <Feature
        headingId="ask-h"
        title="Or just ask"
        lead="Type a question on the Query page. logpipe plans the queries and answers the same way."
        shot={SHOTS.ask}
      >
        <Points items={['Approve each query before it runs, if you want to', 'Answers link to the lines and templates they cite']} />
      </Feature>
    </Section>
  )
}
