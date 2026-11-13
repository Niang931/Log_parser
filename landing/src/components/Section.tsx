import type { CSSProperties, ReactNode } from 'react'
import { Container } from './Container'

export type Route = 'hot' | 'learn' | 'detect' | 'store' | 'ink'

type Props = {
  id: string
  title: ReactNode
  lead: ReactNode
  /** Sits under the lead in the right column: an example, a fact list. */
  aside?: ReactNode
  /** Colour of this section's stretch of the rail. */
  route: Route
  /** Indigo band (the product's sidebar colour) instead of the page ground. */
  band?: boolean
  /** The rail starts at the first station and stops at the last one. */
  rail?: 'first' | 'last'
  children?: ReactNode
}

const routeColor: Record<Route, string> = {
  hot: 'var(--color-route-hot)',
  learn: 'var(--color-route-learn)',
  detect: 'var(--color-route-detect)',
  store: 'var(--color-route-store)',
  ink: 'color-mix(in oklab, var(--color-ink) 30%, transparent)',
}

// The rail: a trunk line left of the content column with one station per section (xl and up only,
// where the 80px gutter has room for it). Station centre = section top padding + 24px.
const railLine = 'xl:before:absolute xl:before:inset-y-0 xl:before:left-[calc(max(0px,(100%_-_1440px)/2)_+_40px)] xl:before:-ml-[3px] xl:before:w-1.5 xl:before:bg-(--route)'
const railStation = 'xl:after:absolute xl:after:top-[140px] xl:after:left-[calc(max(0px,(100%_-_1440px)/2)_+_40px)] xl:after:-ml-3 xl:after:size-6 xl:after:rounded-full xl:after:border-[5px] xl:after:border-ink xl:after:bg-bg'
const railEnds = { first: 'xl:before:top-[152px]', last: 'xl:before:bottom-auto xl:before:h-[152px]' }

export function Section({ id, title, lead, aside, route, band, rail, children }: Props) {
  return (
    <section
      id={id}
      aria-labelledby={`${id}-h`}
      style={{ '--route': routeColor[route] } as CSSProperties}
      className={`relative py-20 md:py-28 xl:pt-32 ${band ? 'bg-surface' : ''} ${railLine} ${railStation} ${rail ? railEnds[rail] : ''}`}
    >
      <Container>
        <header className="grid gap-x-8 gap-y-5 lg:grid-cols-12">
          <h2 id={`${id}-h`} className="text-h2 tracking-[-0.035em] lg:col-span-6">{title}</h2>
          <div className="space-y-6 lg:col-span-5 lg:col-start-8">
            <p className="max-w-[52ch] text-lg leading-relaxed text-muted">{lead}</p>
            {aside}
          </div>
        </header>
        {children && <div className="mt-12 md:mt-16">{children}</div>}
      </Container>
    </section>
  )
}
