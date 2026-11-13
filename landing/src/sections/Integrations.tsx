import { Section } from '../components/Section'

type Tool = { name: string; icon?: string; note?: string; invert?: boolean }

// Each role and what can fill it, from the README. Logos and credits: public/assets/icons.
const ROLES: { role: string; tools: Tool[] }[] = [
  { role: 'Ship', tools: [{ name: 'logpipe-agent', icon: 'go', note: 'Go agent on each host' }] },
  { role: 'Queue', tools: [{ name: 'In-memory' }, { name: 'Kafka', icon: 'kafka', invert: true }] },
  { role: 'Store', tools: [{ name: 'Memory' }, { name: 'ClickHouse', icon: 'clickhouse' }, { name: 'MongoDB', icon: 'mongodb' }] },
  { role: 'Name fields', tools: [{ name: 'Offline rules', note: 'default, no network' }, { name: 'Gemini', icon: 'gemini', note: 'or any web chat or hosted API' }] },
  { role: 'Ask from', tools: [{ name: 'Claude Code', icon: 'claude-code', note: 'over MCP' }] },
  { role: 'Run with', tools: [{ name: 'Docker Compose', icon: 'docker' }] },
]

export function Integrations() {
  return (
    <Section
      id="run"
      route="ink"
      rail="last"
      title="Works with what you already run"
      lead="Kafka or an in-memory queue, ClickHouse or MongoDB, any LLM or none at all."
    >
      <dl className="grid border-t border-line sm:grid-cols-2 lg:grid-cols-3">
        {ROLES.map(({ role, tools }) => (
          <div key={role} className="border-b border-line py-6 sm:pr-8">
            <dt className="text-sm text-muted">{role}</dt>
            {tools.map(t => (
              <dd key={t.name} className="mt-3 flex items-center gap-3">
                {t.icon
                  ? <img src={`/assets/icons/${t.icon}.svg`} alt="" width={22} height={22} className={`size-[22px] object-contain ${t.invert ? 'invert' : ''}`} />
                  : <span aria-hidden className="size-[22px] rounded-full border border-dashed border-line-strong" />}
                <span className="text-[17px] font-medium">{t.name}</span>
                {t.note && <span className="text-sm text-muted">{t.note}</span>}
              </dd>
            ))}
          </div>
        ))}
      </dl>
    </Section>
  )
}
