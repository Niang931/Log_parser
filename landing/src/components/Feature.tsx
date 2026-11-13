import type { ReactNode } from 'react'
import { Screenshot, type Shot } from './Screenshot'
import { Heading, Lead } from './Section'

type Props = {
  headingId?: string
  level?: 2 | 3
  title: ReactNode
  lead: ReactNode
  /** Under the lead: points, an example panel, a fact list. */
  children?: ReactNode
  shot: Shot
  /** Screenshot on the left, words on the right. */
  reverse?: boolean
}

/** Words in four columns beside a screenshot in eight; the words stay in view while the
    screenshot scrolls past. Stacked, words first, below lg. */
export function Feature({ headingId, level = 2, title, lead, children, shot, reverse }: Props) {
  return (
    <div className="grid items-start gap-10 lg:grid-cols-12 lg:gap-12">
      <div className={`grid gap-6 lg:sticky lg:top-28 lg:col-span-4 lg:row-start-1 ${reverse ? 'lg:col-start-9' : ''}`}>
        <Heading id={headingId} level={level}>{title}</Heading>
        <Lead>{lead}</Lead>
        {children}
      </div>
      <Screenshot {...shot} className={`lg:col-span-8 lg:row-start-1 ${reverse ? 'lg:col-start-1' : 'lg:col-start-5'}`} />
    </div>
  )
}
