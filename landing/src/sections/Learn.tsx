import { useEffect, useState, type CSSProperties, type ReactNode } from 'react'
import { Field, KeyValues, Panel } from '../components/Panel'
import { PauseIcon, PlayIcon } from '../components/icons'
import { Heading, Lead, Section } from '../components/Section'

const STEP_MS = 4000

const STEPS: [string, string][] = [
  ['Collect', 'Lines no template matches are kept as PENDING and grouped by shape.'],
  ['Name', 'An LLM, or offline rules, names each field in the shape.'],
  ['Publish', 'The parser gets the new template and old rows are backfilled.'],
]

// One real PENDING shape from the showcase, followed through the three steps.
const RAW = [
  '21:56:36.384 WARN [http-nio-8080-exec-2] c.s.payment.PaymentClient - payment pay_3632506 slow response 3358 ms, retrying',
  '21:56:36.384 WARN [http-nio-8080-exec-2] c.s.payment.PaymentClient - payment pay_3632506 slow response 5196 ms, retrying',
  '21:56:36.718 WARN [http-nio-8080-exec-10] c.s.payment.PaymentClient - payment pay_1280876 slow response 4788 ms, retrying',
]
const MASK = '<VAR:TIMESTAMP> WARN [http-nio-<VAR:NUMBER>-exec-<VAR:NUMBER>] c.s.payment.PaymentClient - payment <*> slow response <VAR:NUMBER> ms, retrying'
const KEYS: [string, string][] = [['<VAR:TIMESTAMP>', '→ timestamp'], ['<*>', '→ payment_id'], ['<VAR:NUMBER> ms', '→ duration_ms']]

const VALUES: [string, string][] = [['timestamp', '21:56:36.384'], ['port', '8080'], ['thread_id', '2'], ['payment_id', 'pay_3632506'], ['duration_ms', '3358']]

/** One stored row, before (outlined) or after (lit green) the backfill. */
const Stored = ({ final, children }: { final?: boolean; children: ReactNode }) => (
  <div className={`grid justify-items-start gap-2.5 rounded-lg border p-4 ${final ? 'border-route-learn/45 bg-route-learn/6' : 'border-line'}`}>{children}</div>
)
const Label = ({ children }: { children: string }) => <p className="font-sans text-sm font-medium text-route-learn">{children}</p>
const Code = ({ children, muted }: { children: ReactNode; muted?: boolean }) => (
  <code className={`block break-all ${muted ? 'text-muted' : 'text-ink'}`}>{children}</code>
)
const Mask = ({ children }: { children: ReactNode }) => (
  <code className="block break-all rounded-md border border-line bg-white/4 px-3.5 py-2.5 text-ink">{children}</code>
)
const Tag = ({ final }: { final?: boolean }) => (
  <span className={`inline-flex h-6 flex-none items-center rounded-[3px] px-2 font-sans text-xs font-semibold ${final ? 'bg-route-learn text-[#07170f]' : 'text-ink ring-[1.5px] ring-route-learn ring-inset'}`}>
    {final ? 'FINAL' : 'PENDING'}
  </span>
)

const VIEWS: ReactNode[] = [
  <>
    <Label>3 lines no template matched</Label>
    {RAW.map(l => <Code key={l} muted>{l}</Code>)}
    <Label>↓ grouped into one mask</Label>
    <Mask>{MASK}</Mask>
  </>,
  <>
    <Label>The mask</Label>
    <Mask>{MASK}</Mask>
    <Label>↓ each value gets a name</Label>
    <KeyValues rows={KEYS} valueClass="text-field" />
  </>,
  <>
    <Label>Rows stored before the template existed</Label>
    <Stored>
      <Tag />
      <Code muted>{RAW[0]}</Code>
      <p className="font-sans text-sm text-muted">No fields yet</p>
    </Stored>
    <Label>↓ backfilled with template #9</Label>
    <Stored final>
      <Tag final />
      <Code>
        <Field>21:56:36.384</Field> WARN [http-nio-<Field>8080</Field>-exec-<Field>2</Field>] c.s.payment.PaymentClient - payment <Field>pay_3632506</Field> slow response <Field>3358</Field> ms, retrying
      </Code>
      <ul className="flex flex-wrap gap-2">
        {VALUES.map(([k, v]) => (
          <li key={k} className="rounded-md border border-line bg-white/4 px-2.5 py-1"><span className="text-field">{k}</span> <span className="text-ink">{v}</span></li>
        ))}
      </ul>
    </Stored>
  </>,
]

/** Steps on the left, the line they act on at reading size on the right. Plays on its own;
    clicking a step holds it there. */
export function Learn() {
  const [step, setStep] = useState(0)
  const [paused, setPaused] = useState(false)
  const [reduced] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches)
  const playing = !paused && !reduced

  useEffect(() => {
    if (!playing) return
    const t = setTimeout(() => setStep(s => (s + 1) % STEPS.length), STEP_MS)
    return () => clearTimeout(t)
  }, [step, playing])

  return (
    <Section id="learn" route="learn">
      <header className="grid gap-x-8 gap-y-5 lg:grid-cols-12">
        <div className="lg:col-span-6"><Heading id="learn-h">Learns new formats on its own</Heading></div>
        <div className="lg:col-span-5 lg:col-start-8"><Lead>No regex to write. Follow one unknown line through the three steps.</Lead></div>
      </header>

      <div className="mt-12 grid items-start gap-8 md:mt-16 lg:grid-cols-12 lg:gap-12">
        <div className="lg:col-span-4">
          <ol className="-ml-4 grid gap-1">
            {STEPS.map(([title, desc], i) => {
              const on = i === step
              return (
                <li key={title} className="relative">
                  {i < STEPS.length - 1 && (
                    <span aria-hidden className={`absolute top-14 bottom-0 left-[35px] w-0.5 -translate-x-1/2 ${i < step ? 'bg-route-learn' : 'bg-route-learn/25'}`} />
                  )}
                  <button
                    type="button"
                    aria-pressed={on}
                    onClick={() => { setStep(i); setPaused(true) }}
                    className={`flex w-full cursor-pointer gap-4 rounded-lg p-4 text-left transition-colors duration-200 ${on ? 'bg-white/5' : 'hover:bg-white/3'}`}
                  >
                    <span className={`relative z-10 grid size-10 flex-none place-items-center rounded-full border-2 border-route-learn text-base font-semibold transition-colors duration-200 ${i <= step ? 'bg-route-learn text-[#07170f]' : 'bg-bg text-ink'}`}>{i + 1}</span>
                    <span className="grid gap-1 pt-1.5">
                      <span className={`text-xl font-medium transition-colors ${on ? 'text-ink' : 'text-muted'}`}>{title}</span>
                      <span className="text-[15px] leading-snug text-muted">{desc}</span>
                      {/* time left on this step while it plays */}
                      <span aria-hidden className={`mt-2 h-0.5 overflow-hidden rounded-full bg-route-learn/20 ${on && playing ? '' : 'invisible'}`}>
                        <span key={`${step}-${playing}`} className="block h-full origin-left animate-[lp-progress_var(--step-ms)_linear_both] bg-route-learn" style={{ '--step-ms': `${STEP_MS}ms` } as CSSProperties} />
                      </span>
                    </span>
                  </button>
                </li>
              )
            })}
          </ol>
        </div>

        <Panel className="overflow-hidden lg:col-span-8">
          <div className="flex items-center justify-between gap-4 border-b border-line px-5 py-3 font-sans text-sm">
            <span className="text-muted">Step {step + 1} of 3 · <span className="text-ink">{STEPS[step][0]}</span></span>
            {!reduced && (
              <button
                type="button"
                onClick={() => setPaused(p => !p)}
                className="inline-flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1 text-muted transition-colors hover:text-ink"
              >
                {paused ? <PlayIcon className="size-3.5" /> : <PauseIcon className="size-3.5" />}
                {paused ? 'Play' : 'Pause'}
              </button>
            )}
          </div>
          {/* on desktop all three views share one cell so the panel is as tall as the tallest and never jumps; on phones it fits the current step */}
          <div className="grid p-5 text-[13px] leading-relaxed sm:p-7 sm:text-[15px]" aria-live="polite">
            {VIEWS.map((view, i) => (
              <div
                key={i}
                aria-hidden={i !== step}
                className={`col-start-1 row-start-1 content-start gap-3.5 ${i === step ? 'grid *:animate-[lp-in_450ms_var(--ease-out-expo)_both] motion-reduce:*:animate-none' : 'hidden lg:invisible lg:grid'}`}
              >
                {view}
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </Section>
  )
}
