import { ButtonLink } from '../components/Button'
import { Container } from '../components/Container'
import { Logo } from '../components/Logo'
import { Panel } from '../components/Panel'
import { DEMO, DOCS, REPO } from '../links'

const SETUP = [`git clone ${REPO}.git`, 'cd Log_parser', 'uv sync --group dev', DEMO]

const COLUMNS = [
  { title: 'Product', links: [['Parsing', '#parse'], ['Learning', '#learn'], ['Detection', '#detect'], ['Root cause', '#explain'], ['Backends', '#run']] },
  { title: 'Resources', links: [['Docs', DOCS], ['GitHub', REPO], ['Issues', `${REPO}/issues`]] },
]

export function Footer() {
  return (
    <footer aria-labelledby="close-h" className="relative isolate border-t border-line bg-[#0e0e12]">
      <div aria-hidden className="absolute inset-x-0 top-0 -z-10 h-[70%] bg-[radial-gradient(60%_80%_at_15%_0%,color-mix(in_oklab,var(--color-accent)_18%,transparent),transparent_70%)]" />
      <Container className="py-20 md:py-28">
        <div className="grid items-start gap-10 lg:grid-cols-12">
          <div className="lg:col-span-6">
            <h2 id="close-h" className="text-h2 tracking-[-0.035em]">Try logpipe on your own logs</h2>
            <p className="mt-5 text-lg text-muted">Open source and offline by default. Set up in four commands.</p>
            <div className="mt-8 flex flex-wrap gap-3">
              <ButtonLink href={REPO} arrow>View on GitHub</ButtonLink>
              <ButtonLink href={DOCS} variant="secondary">Read the docs</ButtonLink>
            </div>
          </div>
          <Panel className="grid gap-1.5 px-5 py-4 text-sm lg:col-span-6 lg:col-start-7" role="img" aria-label="The four commands that clone logpipe, install it and start the showcase.">
            {SETUP.map(c => (
              <code key={c} className="break-all text-ink"><span className="mr-2 text-route-learn select-none">$</span>{c}</code>
            ))}
          </Panel>
        </div>

        <div className="mt-20 grid gap-10 border-t border-line pt-12 sm:grid-cols-2 lg:grid-cols-12">
          <div className="space-y-4 lg:col-span-4">
            <Logo />
            <p className="text-muted">Log parsing and anomaly detection for SRE teams.</p>
          </div>
          <div className="grid grid-cols-2 gap-8 lg:col-span-4 lg:col-start-9">
            {COLUMNS.map(col => (
              <div key={col.title}>
                <h3 className="text-sm text-muted">{col.title}</h3>
                <ul className="mt-4 space-y-2.5">
                  {col.links.map(([label, href]) => (
                    <li key={label}><a href={href} className="transition-colors hover:text-accent">{label}</a></li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-14 flex flex-col gap-3 text-sm text-muted md:flex-row md:justify-between">
          <span>&copy; 2026 logpipe</span>
          <span>Includes code from DeepParse (Apache-2.0) and LogBERT (MIT). Product names and logos are trademarks of their owners.</span>
        </div>
      </Container>
    </footer>
  )
}
