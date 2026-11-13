import { useEffect, useState, type CSSProperties, type ReactNode } from 'react'
import { Field, KeyValues, Panel } from '../components/Panel'
import { PauseIcon, PlayIcon } from '../components/icons'
import { Section } from '../components/Section'

const STEPS: [string, string][] = [
  ['Collect', 'Unknown lines are grouped by shape.'],
  ['Name', 'An LLM names each field.'],
  ['Publish', 'The parser updates and old rows are backfilled.'],
]

// One real PENDING shape from the showcase, followed through the three steps.
const RAW = [
  '21:56:36.384 WARN [http-nio-8080-exec-2] c.s.payment.PaymentClient - payment pay_3632506 slow response 3358 ms, retrying',
  '21:56:36.384 WARN [http-nio-8080-exec-2] c.s.payment.PaymentClient - payment pay_3632506 slow response 5196 ms, retrying',
  '21:56:36.718 WARN [http-nio-8080-exec-10] c.s.payment.PaymentClient - payment pay_1280876 slow response 4788 ms, retrying',
]
const MASK = '<VAR:TIMESTAMP> WARN [http-nio-<VAR:NUMBER>-exec-<VAR:NUMBER>] c.s.payment.PaymentClient - payment <*> slow response <VAR:NUMBER> ms, retrying'
const KEYS: [string, string][] = [['<VAR:TIMESTAMP>', '→ timestamp'], ['<*>', '→ payment_id'], ['<VAR:NUMBER> ms', '→ duration_ms']]

const Heading = ({ children }: { children: string }) => <p className="font-sans text-[13px] font-medium text-muted">{children}</p>
const To = ({ children }: { children: string }) => <p className="font-sans text-xs font-medium text-route-learn">{children}</p>
const Mask = ({ children }: { children: ReactNode }) => <code className="block break-all rounded bg-white/5 px-2.5 py-2 text-ink">{children}</code>
const Tag = ({ final }: { final?: boolean }) => (
  <span className={`inline-flex h-5 flex-none items-center rounded-[3px] px-2 font-sans text-xs font-medium ${final ? 'bg-route-learn text-[#07170f]' : 'text-ink ring-[1.5px] ring-route-learn ring-inset'}`}>
    {final ? 'FINAL' : 'PENDING'}
  </span>
)

function StepView({ step }: { step: number }) {
  if (step === 0) return (
    <>
      <Heading>3 lines no template matched · stored as PENDING</Heading>
      {RAW.map(l => <code key={l} className="block break-all text-muted">{l}</code>)}
      <To>↓ grouped into one mask</To>
      <Mask>{MASK}</Mask>
    </>
  )
  if (step === 1) return (
    <>
      <Heading>The LLM names each value in the mask</Heading>
      <Mask>{MASK}</Mask>
      <KeyValues rows={KEYS} valueClass="text-field" />
    </>
  )
  return (
    <>
      <Heading>Template #9 joins registry v4 · the stored rows are rewritten</Heading>
      <Mask>
        <Field>timestamp</Field> WARN [http-nio-<Field>port</Field>-exec-<Field>thread_id</Field>] c.s.payment.PaymentClient - payment <Field>payment_id</Field> slow response <Field>duration_ms</Field> ms, retrying
      </Mask>
      <div className="flex items-baseline gap-3"><Tag /><code className="break-all text-ink">template UNK · registry v3 · {'{}'}</code></div>
      <To>↓ backfilled</To>
      <div className="flex items-baseline gap-3"><Tag final /><code className="break-all text-ink">template 9 · registry v4 · {'{"payment_id": "pay_3632506", "duration_ms": 3358}'}</code></div>
    </>
  )
}

/** Three steps left to right; the dot rides to the active one and the panel shows what it does to one real line. */
export function Learn() {
  const [step, setStep] = useState(0)
  const [paused, setPaused] = useState(false)
  useEffect(() => {
    if (paused || matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const t = setTimeout(() => setStep(s => (s + 1) % STEPS.length), 3200)
    return () => clearTimeout(t)
  }, [step, paused])

  return (
    <Section id="learn" route="learn" title="Learns new formats on its own" lead="No regex to write. Follow one unknown line through the three steps.">
      <div className="relative mb-8" style={{ '--step': step } as CSSProperties}>
        {/* line from the first number's centre to the last one's; 50cqw of it is one step */}
        <span aria-hidden className="@container absolute top-[19px] right-[calc(33.333%-36px)] left-5 hidden h-0.5 bg-route-learn/45 sm:block">
          <span className="absolute -top-1.5 -left-[7px] size-3.5 translate-x-[calc(var(--step)*50cqw)] rounded-full bg-route-learn shadow-[0_0_0_5px_color-mix(in_oklab,var(--color-route-learn)_25%,transparent)] transition-transform duration-700 ease-in-out motion-reduce:transition-none" />
        </span>
        <ol className="grid grid-cols-3 gap-3 sm:gap-6">
          {STEPS.map(([title, desc], i) => (
            <li key={title}>
              <button
                type="button"
                aria-pressed={i === step}
                onClick={() => { setStep(i); setPaused(true) }}
                className="group grid w-full cursor-pointer justify-items-start gap-2 rounded-sm text-left"
              >
                <span className={`relative z-10 grid size-10 place-items-center rounded-full border-2 border-route-learn font-medium transition-colors duration-200 motion-reduce:transition-none ${i <= step ? 'bg-route-learn text-[#07170f]' : 'bg-bg text-ink'}`}>{i + 1}</span>
                <span className={`mt-3 text-lg font-medium transition-colors group-hover:text-ink ${i === step ? 'text-ink' : 'text-muted'}`}>{title}</span>
                <span className="hidden text-muted sm:block">{desc}</span>
              </button>
            </li>
          ))}
        </ol>
      </div>
      <Panel className="relative min-h-[300px] py-5 pr-16 pl-6 sm:h-[340px] sm:overflow-hidden">
        <div aria-live="polite" key={step} className="grid content-start gap-2.5 *:animate-[lp-in_400ms_var(--ease-out-expo)_both] motion-reduce:*:animate-none">
          <StepView step={step} />
        </div>
        <button
          type="button"
          onClick={() => setPaused(p => !p)}
          aria-label={paused ? 'Play animation' : 'Pause animation'}
          className="absolute top-4 right-4 grid size-8 cursor-pointer place-items-center rounded-full border border-line-strong text-muted transition-colors hover:text-ink"
        >
          {paused ? <PlayIcon className="size-3.5" /> : <PauseIcon className="size-3.5" />}
        </button>
      </Panel>
    </Section>
  )
}
