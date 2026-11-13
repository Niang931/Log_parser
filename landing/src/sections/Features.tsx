import { KeyValues, Panel } from '../components/Panel'
import { Screenshot } from '../components/Screenshot'
import { Section } from '../components/Section'
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
    <Section
      id="parse"
      route="hot"
      rail="first"
      title="Turns every line into fields"
      lead="Each line comes out as a template with named fields. Lines it can’t read yet are kept, never dropped."
      aside={
        <Panel className="grid gap-3 p-5" role="img" aria-label="An nginx line and the fields logpipe extracts from it: client_ip, timestamp, url_path, status_code and response_size_bytes.">
          <code className="break-all text-muted">{NGINX_LINE}</code>
          <span className="font-sans text-xs font-medium text-route-learn">↓ template #4</span>
          <KeyValues rows={NGINX_FIELDS} keyClass="text-field" />
        </Panel>
      }
    >
      <Screenshot {...SHOTS.templates} caption="The Templates page: every shape the parser knows, fields highlighted, new shapes marked unseen until they show up." />
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
    <Section
      id="detect"
      route="detect"
      band
      title="Spots the lines that don’t belong"
      lead="LogBERT learns each service’s normal flow and flags what breaks it."
      aside={
        <dl className="divide-y divide-line border-y border-line text-[15px]">
          {INCIDENT.map(([k, v]) => (
            <div key={k} className="flex justify-between gap-6 py-3">
              <dt className="text-muted">{k}</dt>
              <dd className="text-right tabular-nums">{v}</dd>
            </div>
          ))}
        </dl>
      }
    >
      <Screenshot {...SHOTS.incident} caption="An open incident: each line masked in turn, the ones the model never learned marked in amber." />
    </Section>
  )
}

export function Explain() {
  return (
    <Section
      id="explain"
      route="store"
      title="Ask why it broke. Get the lines that prove it."
      lead="Plain-language answers that cite the exact log lines. Every query is read-only."
    >
      <div className="grid gap-16 md:gap-20">
        <Screenshot {...SHOTS.rootCause} caption="Investigate an incident: the likely cause, a timeline and next steps, each claim linked to the lines it came from." />
        <Screenshot {...SHOTS.ranking} caption="The working behind it: every template scored as anomaly × rarity × proximity, and which factor set the score." />
      </div>
    </Section>
  )
}

export function Ask() {
  return (
    <Section
      id="ask"
      route="store"
      title="Or just ask"
      lead="Type a question on the Query page. logpipe plans the queries, waits for your approval if you want it, and answers the same way."
    >
      <Screenshot {...SHOTS.ask} />
    </Section>
  )
}
