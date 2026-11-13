import { useState } from 'react'
import { ButtonLink } from '../components/Button'
import { Container } from '../components/Container'
import { MenuIcon } from '../components/icons'
import { Logo } from '../components/Logo'
import { DOCS, REPO } from '../links'

const LINKS = [
  ['Parsing', '#parse'],
  ['Detection', '#detect'],
  ['Root cause', '#explain'],
  ['Docs', DOCS],
] as const

export function Nav() {
  const [open, setOpen] = useState(false)
  return (
    <header className="sticky top-0 z-50 border-b border-line bg-bg/95">
      <a href="#top" className="sr-only focus:not-sr-only focus:absolute focus:top-3 focus:left-3 focus:rounded-sm focus:bg-accent focus:px-3 focus:py-2 focus:text-accent-ink">Skip to content</a>
      <Container className="flex h-16 items-center gap-8">
        <Logo />
        <nav aria-label="Main" className="hidden md:block">
          <ul className="flex gap-7 text-[15px] text-muted">
            {LINKS.map(([label, href]) => (
              <li key={label}><a href={href} className="transition-colors hover:text-ink">{label}</a></li>
            ))}
          </ul>
        </nav>
        <ButtonLink href={REPO} className="ml-auto hidden h-9! px-4! text-sm! md:inline-flex">View on GitHub</ButtonLink>
        <button
          type="button"
          className="-mr-2 ml-auto grid size-10 cursor-pointer place-items-center rounded-sm text-ink md:hidden"
          aria-expanded={open}
          aria-controls="mobile-nav"
          aria-label={open ? 'Close menu' : 'Open menu'}
          onClick={() => setOpen(o => !o)}
        >
          <MenuIcon open={open} className="size-[18px]" />
        </button>
      </Container>
      {open && (
        <nav id="mobile-nav" aria-label="Main" className="border-t border-line md:hidden">
          <Container className="flex flex-col py-3">
            {LINKS.map(([label, href]) => (
              <a key={label} href={href} onClick={() => setOpen(false)} className="py-3 text-lg">{label}</a>
            ))}
            <ButtonLink href={REPO} className="mt-3 mb-2">View on GitHub</ButtonLink>
          </Container>
        </nav>
      )}
    </header>
  )
}
