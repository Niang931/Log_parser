import { useEffect, useState, type CSSProperties } from 'react'

const REPO = 'https://github.com/Niang931/Log_parser'
const DOCS = 'https://niang931.github.io/Log_parser/getting-started/'
const DEMO = 'uv run logpipe showcase'

const Arrow = () => (
  <svg className="m-arrow" viewBox="0 0 12 12" aria-hidden="true"><path className="m-arrow__stem" d="M1 6h8" /><path className="m-arrow__head" d="M4.5 2.5 8 6l-3.5 3.5" /></svg>
)

// Square brackets in real log lines get their own span, so the kit's placeholder guard (any
// [text] in one text node) does not outline them as leftovers.
const B = (line: string) => line.split(/([[\]])/).map((p, i) => (p === '[' || p === ']' ? <span key={i}>{p}</span> : p))

const Logo = () => (
  <a className="nav__logo" href="#top"><span className="mark" data-mark="arc" aria-hidden="true"></span>logpipe</a>
)

const PauseIcon = ({ paused }: { paused: boolean }) => paused
  ? <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5.25 3.7v8.6c0 .62.68 1 1.2.66l6.5-4.3c.47-.31.47-1 0-1.32l-6.5-4.3c-.52-.34-1.2.04-1.2.66Z" /></svg>
  : <svg viewBox="0 0 16 16" aria-hidden="true"><rect x="4.25" y="3.25" width="2.75" height="9.5" rx="0.9" /><rect x="9" y="3.25" width="2.75" height="9.5" rx="0.9" /></svg>

// A screenshot of logpipe's own UI, cropped from docs/apps_ref_for_ads_web/screenshots.
const Shot = ({ src, w, h, alt, cls = '' }: { src: string; w: number; h: number; alt: string; cls?: string }) => (
  <div className={`frame frame--photo lp-shot ${cls}`} style={{ '--ratio': cls.includes('frame--fill') ? '0px' : `${(h / w) * 100}%` } as CSSProperties} data-visual>
    <img className="frame__img" src={`/img/${src}`} width={w} height={h} alt={alt} loading="lazy" data-source="client" data-subject="logpipe's UI" />
  </div>
)

function CopyDemo() {
  const [done, setDone] = useState(false)
  const copy = async () => {
    try { await navigator.clipboard.writeText(DEMO) } catch { return }
    setDone(true)
    setTimeout(() => setDone(false), 1800)
  }
  return (
    <button className="btn btn--secondary lp-copy" type="button" onClick={copy} aria-label={done ? 'Copied' : `Copy the command ${DEMO}`}>
      <code>$ {DEMO}</code>
      {done
        ? <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m3.5 8.4 2.8 2.8 6.2-6.4" /></svg>
        : <svg viewBox="0 0 16 16" aria-hidden="true"><rect x="5.5" y="5.5" width="8" height="8" rx="1.5" /><path d="M10.5 5.5v-2a1 1 0 0 0-1-1h-6a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2" /></svg>}
    </button>
  )
}

function Nav() {
  return (
    <header className="nav">
      <div className="wrap nav__bar">
        <a className="skip-link" href="#top">Skip to content</a>
        <Logo />
        <nav className="nav__links" aria-label="Main">
          <div className="nav__item"><a className="nav__link" href="#parse">Parsing</a></div>
          <div className="nav__item"><a className="nav__link" href="#detect">Detection</a></div>
          <div className="nav__item"><a className="nav__link" href="#explain">Root cause</a></div>
          <div className="nav__item"><a className="nav__link" href={DOCS}>Docs</a></div>
        </nav>
        <div className="nav__actions">
          <a className="btn btn--primary" href={REPO}>View on GitHub</a>
        </div>
        <button className="nav__toggle" type="button" aria-expanded="false" aria-label="Menu">
          <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><path d="M2 5h14M2 13h14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>
        </button>
      </div>
    </header>
  )
}

function Hero() {
  return (
    <section className="hero hero--centred" data-enter aria-labelledby="hero-h">
      <div className="wrap">
        <div className="hero-text">
          <div className="hero__c">
            <h1 id="hero-h">Self-learning log parser for SRE teams</h1>
            <p className="hero__lead">Parses known lines in under a millisecond, learns new formats on its own and tells you why things broke.</p>
            <div className="btn-row">
              <a className="btn btn--primary" href={REPO}>View on GitHub<Arrow /></a>
              <CopyDemo />
            </div>
          </div>
          <div className="hero__window lp-herowin" data-enter-media>
            <img className="frame__img" src="/img/log-explorer.webp" width={1668} height={640} alt="logpipe's Log Explorer: a live chart of log volume over a table of parsed nginx, postgres and shop-api lines." fetchPriority="high" data-source="client" data-subject="logpipe's UI" />
          </div>
        </div>
      </div>
    </section>
  )
}

function Parse() {
  return (
    <section className="section rail rail--hot rail--first" id="parse" aria-labelledby="parse-h">
      <div className="wrap">
        <div className="grid fill">
          <div className="span-5 stack-4">
            <h2 id="parse-h">Turns every line into fields</h2>
            <p className="t-lead">Each line comes out as a template with named fields. Lines it can’t read yet are kept, never dropped.</p>
            <div className="ui ui-panel lp-example" role="img" aria-label="An nginx line and the fields logpipe extracts from it: client_ip, timestamp, url_path, status_code and response_size_bytes.">
              <code className="lp-example__raw">{B('10.0.1.191 - - [12/Nov/2026:21:56:08 +0800] "POST /api/checkout HTTP/1.1" 200 8978')}</code>
              <span className="lp-example__arrow" aria-hidden="true">↓ template #4</span>
              <dl className="lp-example__fields">
                {[['client_ip', '10.0.1.191'], ['timestamp', '12/Nov/2026:21:56:08 +0800'], ['url_path', '/api/checkout'], ['status_code', '200'], ['response_size_bytes', '8978']].map(([k, v]) => (
                  <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
                ))}
              </dl>
            </div>
          </div>
          <Shot cls="span-6 start-7 frame--fill" src="templates.webp" w={840} h={560} alt="logpipe's Templates page: log shapes with their fields highlighted, such as client_ip, timestamp and duration_ms." />
        </div>
      </div>
    </section>
  )
}

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

function StepView({ step }: { step: number }) {
  if (step === 0) return (
    <>
      <p className="lp-demo__h">3 lines no template matched · stored as PENDING</p>
      {RAW.map(l => <code key={l} className="lp-demo__raw">{B(l)}</code>)}
      <p className="lp-demo__to">↓ grouped into one mask</p>
      <code className="lp-demo__mask">{B(MASK)}</code>
    </>
  )
  if (step === 1) return (
    <>
      <p className="lp-demo__h">The LLM names each value in the mask</p>
      <code className="lp-demo__mask">{B(MASK)}</code>
      <dl className="lp-demo__keys">
        {[['<VAR:TIMESTAMP>', 'timestamp'], ['<*>', 'payment_id'], ['<VAR:NUMBER> ms', 'duration_ms']].map(([m, k]) => (
          <div key={k}><dt>{m}</dt><dd>→ <span className="lp-field">{k}</span></dd></div>
        ))}
      </dl>
    </>
  )
  return (
    <>
      <p className="lp-demo__h">Template #9 joins registry v4 · the stored rows are rewritten</p>
      <code className="lp-demo__mask"><span className="lp-field">timestamp</span> WARN {B('[http-nio-')}<span className="lp-field">port</span>-exec-<span className="lp-field">thread_id</span>{B(']')} c.s.payment.PaymentClient - payment <span className="lp-field">payment_id</span> slow response <span className="lp-field">duration_ms</span> ms, retrying</code>
      <div className="lp-demo__row"><span className="lp-tag">PENDING</span><code>template UNK · registry v3 · {'{}'}</code></div>
      <p className="lp-demo__to">↓ backfilled</p>
      <div className="lp-demo__row"><span className="lp-tag lp-tag--final">FINAL</span><code>template 9 · registry v4 · {'{"payment_id": "pay_3632506", "duration_ms": 3358}'}</code></div>
    </>
  )
}

function Learn() {
  const [step, setStep] = useState(0)
  const [paused, setPaused] = useState(false)
  useEffect(() => {
    if (paused || matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const t = setTimeout(() => setStep(s => (s + 1) % STEPS.length), 3200)
    return () => clearTimeout(t)
  }, [step, paused])
  return (
    <section className="section rail rail--learn" id="learn" aria-labelledby="learn-h">
      <div className="wrap">
        <div className="section-head">
          <h2 id="learn-h">Learns new formats on its own</h2>
          <p>No regex to write. Follow one unknown line through the 3 steps.</p>
        </div>
        <div className="lp-flow" style={{ '--step': step } as CSSProperties} data-visual>
          <span className="lp-flow__line" aria-hidden="true"><span className="lp-flow__dot"></span></span>
          <ol className="lp-flow__steps">
            {STEPS.map(([t, d], i) => (
              <li key={t} className={`lp-step${i === step ? ' is-on' : ''}${i < step ? ' is-done' : ''}`}>
                <button type="button" className="lp-step__btn" aria-pressed={i === step} onClick={() => { setStep(i); setPaused(true) }}>
                  <span className="lp-step__n" aria-hidden="true">{i + 1}</span>
                  <span className="lp-step__t">{t}</span>
                  <span className="lp-step__d">{d}</span>
                </button>
              </li>
            ))}
          </ol>
        </div>
        <div className="ui ui-panel lp-demo" data-visual>
          <div className="lp-demo__body" aria-live="polite"><StepView key={step} step={step} /></div>
          <button className="loop-pause" type="button" aria-label={paused ? 'Play animation' : 'Pause animation'} onClick={() => setPaused(p => !p)}><PauseIcon paused={paused} /></button>
        </div>
      </div>
    </section>
  )
}

function Detect() {
  return (
    <section className="section ground-surface rail rail--detect" id="detect" aria-labelledby="detect-h">
      <div className="wrap">
        <div className="grid fill">
          <div className="span-5 stack-4">
            <h2 id="detect-h">Spots the lines that don’t belong</h2>
            <p className="t-lead">LogBERT learns each service’s normal flow and flags what breaks it.</p>
            <ul className="rows lp-incident">
              <li><span className="rows__name">Service</span><span className="rows__value">shop-api on shop-web-1</span></li>
              <li><span className="rows__name">Window</span><span className="rows__value">9:56:36 to 9:57:35 PM</span></li>
              <li><span className="rows__name">Alerts</span><span className="rows__value">90</span></li>
              <li><span className="rows__name">Unusual lines</span><span className="rows__value">279 of 400</span></li>
            </ul>
          </div>
          <Shot cls="span-6 start-7 frame--fill" src="detect.webp" w={855} h={660} alt="An incident in logpipe: payment warnings and errors in shop-api marked as lines the model never learned." />
        </div>
      </div>
    </section>
  )
}

function Explain() {
  return (
    <section className="section rail rail--store" id="explain" aria-labelledby="explain-h">
      <div className="wrap">
        <div className="section-head">
          <h2 id="explain-h">Ask why it broke. Get the lines that prove it.</h2>
          <p>Plain-language answers that cite the exact log lines. Every query is read-only.</p>
        </div>
        <Shot cls="lp-shot--phone" src="root-cause.webp" w={1640} h={620} alt="logpipe's root cause answer: PostgreSQL ran out of connection slots, citing log lines 18 and 19, with a timeline and next steps." />
      </div>
    </section>
  )
}

const HUB = {
  layout: 'hub',
  alt: 'logpipe connects to its Go agent, Kafka, ClickHouse, MongoDB, Docker, Claude Code over MCP and Gemini.',
  center: { label: 'logpipe' },
  spokes: [
    { label: 'logpipe-agent', icon: 'si:go' },
    { label: 'Kafka', icon: 'kafka' },
    { label: 'ClickHouse', icon: 'si:clickhouse' },
    { label: 'MongoDB', icon: 'mongodb' },
    { label: 'Docker', icon: 'docker' },
    { label: 'Claude Code', icon: 'claude-code' },
    { label: 'Gemini', icon: 'gemini' },
  ],
}

function Run() {
  return (
    <section className="section rail rail--ink rail--last" id="run" aria-labelledby="run-h">
      <div className="wrap">
        <div className="grid">
          <div className="span-5 stack-4">
            <h2 id="run-h">Works with what you already run</h2>
            <p className="t-lead">Kafka or an in-memory queue, ClickHouse or MongoDB, any LLM or none at all.</p>
          </div>
          <div className="frame frame--split span-6 start-7" data-visual>
            <div className="dg" data-diagram data-dg-still><script type="application/json" dangerouslySetInnerHTML={{ __html: JSON.stringify(HUB) }} /></div>
          </div>
        </div>
      </div>
    </section>
  )
}

function Footer() {
  return (
    <footer className="footer closing footer--dark" id="contact" aria-labelledby="close-h">
      <div className="wrap">
        <div className="closing__cta">
          <div className="grid lp-cta">
          <div className="lp-cta__text span-6">
            <h2 id="close-h">Try logpipe on your own logs</h2>
            <p>Open source and offline by default. Set up in four commands.</p>
            <div className="btn-row">
              <a className="btn btn--primary" href={REPO}>View on GitHub<Arrow /></a>
              <a className="btn btn--secondary" href={DOCS}>Read the docs</a>
            </div>
          </div>
          <div className="ui ui-panel lp-term span-6 start-7" data-visual role="img" aria-label="The four commands that clone logpipe, install it and start the showcase.">
            <code><span className="lp-term__p">$</span> git clone {REPO}.git</code>
            <code><span className="lp-term__p">$</span> cd Log_parser</code>
            <code><span className="lp-term__p">$</span> uv sync --group dev</code>
            <code><span className="lp-term__p">$</span> {DEMO}</code>
          </div>
          </div>
        </div>
        <div className="closing__links">
          <div className="grid">
            <div className="span-4 stack-4">
              <Logo />
              <p className="footer__lede">Log parsing and anomaly detection for SRE teams.</p>
            </div>
            <div className="span-2 start-9 footer__col"><h3>Product</h3><ul><li><a href="#parse">Parsing</a></li><li><a href="#learn">Learning</a></li><li><a href="#detect">Detection</a></li><li><a href="#explain">Root cause</a></li><li><a href="#run">Backends</a></li></ul></div>
            <div className="span-2 footer__col"><h3>Resources</h3><ul><li><a href={DOCS}>Docs</a></li><li><a href={REPO}>GitHub</a></li><li><a href={`${REPO}/issues`}>Issues</a></li></ul></div>
          </div>
          <div className="footer__legal">
            <span>&copy; 2026 logpipe</span>
            <div className="socials"></div>
            <span className="credits">Includes code from DeepParse (Apache-2.0) and LogBERT (MIT). Product names and logos are trademarks of their owners.</span>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default function App() {
  return (
    <>
      <Nav />
      <main id="top">
        <Hero />
        <Parse />
        <Learn />
        <Detect />
        <Explain />
        <Run />
      </main>
      <Footer />
    </>
  )
}
