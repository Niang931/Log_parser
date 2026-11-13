import type { CSSProperties, ReactNode } from 'react'
import { Container } from './Container'

export type Route = 'hot' | 'learn' | 'detect' | 'store' | 'ink'

type Props = {
  id: string
  /** Colour of this section's stretch of the rail; also tints its point markers. */
  route: Route
  /** Indigo band (the product's sidebar colour) instead of the page ground. */
  band?: boolean
  /** The rail starts at the first station and stops at the last one. */
  rail?: 'first' | 'last'
  children: ReactNode
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
const railStation = 'xl:after:absolute xl:after:top-[124px] xl:after:left-[calc(max(0px,(100%_-_1440px)/2)_+_40px)] xl:after:-ml-3 xl:after:size-6 xl:after:rounded-full xl:after:border-[5px] xl:after:border-ink xl:after:bg-bg'
const railEnds = { first: 'xl:before:top-[136px]', last: 'xl:before:bottom-auto xl:before:h-[136px]' }

/** A page section on the rail. Its heading must carry id `${id}-h`. */
export function Section({ id, route, band, rail, children }: Props) {
  return (
    <section
      id={id}
      aria-labelledby={`${id}-h`}
      style={{ '--route': routeColor[route] } as CSSProperties}
      className={`relative py-20 md:py-24 xl:pt-28 ${band ? 'bg-surface' : ''} ${railLine} ${railStation} ${rail ? railEnds[rail] : ''}`}
    >
      <Container>{children}</Container>
    </section>
  )
}

export const Heading = ({ id, level = 2, children }: { id?: string; level?: 2 | 3; children: ReactNode }) => {
  const H = `h${level}` as const
  return <H id={id} className={`${level === 2 ? 'text-h2' : 'text-[clamp(1.5rem,1vw+1.2rem,2.125rem)] leading-tight'} tracking-[-0.035em]`}>{children}</H>
}

export const Lead = ({ children }: { children: ReactNode }) => (
  <p className="max-w-[52ch] text-[19px] leading-[1.6] text-ink/80">{children}</p>
)

/** Short facts beside a screenshot, each ticked in the section's route colour. */
export function Points({ items }: { items: ReactNode[] }) {
  return (
    <ul className="grid gap-3.5 border-t border-line pt-6 text-[17px] leading-snug font-medium">
      {items.map((item, i) => (
        <li key={i} className="flex items-start gap-3">
          <span aria-hidden className="mt-px grid size-5 flex-none place-items-center rounded-full bg-[color-mix(in_oklab,var(--route)_22%,transparent)] text-(--route)">
            <svg viewBox="0 0 12 12" className="size-3" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="m2.5 6.2 2.3 2.3 4.7-4.9" /></svg>
          </span>
          <span>{item}</span>
        </li>
      ))}
    </ul>
  )
}
