import type { CSSProperties } from 'react'
import { Heading, Lead, Section } from '../components/Section'

type Tool = { name: string; icon?: string; note?: string; invert?: boolean }
type Stop = { role: string; tools: Tool[] }

// Each stage a line goes through and what can fill it, from the README.
// Logos and their credits: public/assets/icons.
const PIPELINE: Stop[] = [
  { role: 'Ship', tools: [{ name: 'logpipe-agent', icon: 'go', note: 'Go, one per host' }] },
  { role: 'Queue', tools: [{ name: 'In-memory' }, { name: 'Kafka', icon: 'kafka', invert: true }] },
  { role: 'Store', tools: [{ name: 'Memory' }, { name: 'ClickHouse', icon: 'clickhouse' }, { name: 'MongoDB', icon: 'mongodb' }] },
  { role: 'Name fields', tools: [{ name: 'Offline rules', note: 'default' }, { name: 'Gemini', icon: 'gemini', note: 'or any web chat or API' }] },
]
// one colour per stretch, borrowed from the page's own rail
const SEGMENTS = ['var(--color-route-hot)', 'var(--color-route-learn)', 'var(--color-route-detect)']

const AROUND: Stop[] = [
  { role: 'Ask from', tools: [{ name: 'Claude Code', icon: 'claude-code', note: 'over MCP' }] },
  { role: 'Run with', tools: [{ name: 'Docker Compose', icon: 'docker' }] },
]

function ToolRow({ name, icon, note, invert }: Tool) {
  return (
    <li className="flex flex-wrap items-center gap-x-3 gap-y-0.5">
      {icon
        ? <img src={`/assets/icons/${icon}.svg`} alt="" width={22} height={22} className={`size-[22px] object-contain ${invert ? 'invert' : ''}`} />
        : <span aria-hidden className="size-[22px] rounded-full border border-dashed border-line-strong" />}
      <span className="text-[17px] font-medium">{name}</span>
      {note && <span className="text-sm text-muted">{note}</span>}
    </li>
  )
}

/** A line's path as a route with one stop per stage (the page's rail, turned sideways on desktop). */
export function Integrations() {
  return (
    <Section id="run" route="ink" rail="last">
      <header className="grid gap-x-8 gap-y-5 lg:grid-cols-12">
        <div className="lg:col-span-6"><Heading id="run-h">Works with what you already run</Heading></div>
        <div className="lg:col-span-5 lg:col-start-8"><Lead>Kafka or an in-memory queue, ClickHouse or MongoDB, any LLM or none at all.</Lead></div>
      </header>

      {/* drawn like the page's rail: 6px line, ringed stations, one colour per stretch */}
      <ol className="mt-14 grid gap-10 md:mt-20 lg:grid-cols-4 lg:gap-8">
        {PIPELINE.map(({ role, tools }, i) => (
          <li
            key={role}
            style={{ '--seg': SEGMENTS[i] } as CSSProperties}
            className={`relative pl-12 lg:pt-14 lg:pl-0 ${i < PIPELINE.length - 1
              ? 'before:absolute before:top-3 before:-bottom-10 before:left-3 before:-ml-[3px] before:w-1.5 before:bg-(--seg) lg:before:top-3 lg:before:-right-8 lg:before:bottom-auto lg:before:left-3 lg:before:-mt-[3px] lg:before:ml-0 lg:before:h-1.5 lg:before:w-auto'
              : ''}`}
          >
            <span aria-hidden className="absolute top-0 left-0 z-10 size-6 rounded-full border-[5px] border-ink bg-bg" />
            <p className="text-lg font-semibold">{role}</p>
            <ul className="mt-4 grid gap-3">{tools.map(t => <ToolRow key={t.name} {...t} />)}</ul>
          </li>
        ))}
      </ol>

      <div className="mt-14 grid gap-8 border-t border-line pt-8 sm:grid-cols-2 lg:grid-cols-4 lg:gap-8">
        {AROUND.map(({ role, tools }) => (
          <div key={role}>
            <p className="text-lg font-semibold">{role}</p>
            <ul className="mt-3 grid gap-3">{tools.map(t => <ToolRow key={t.name} {...t} />)}</ul>
          </div>
        ))}
      </div>
    </Section>
  )
}
