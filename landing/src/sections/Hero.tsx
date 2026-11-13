import { ButtonLink } from '../components/Button'
import { Container } from '../components/Container'
import { Screenshot } from '../components/Screenshot'
import { DOCS, REPO } from '../links'
import { SHOTS } from '../shots'

export function Hero() {
  return (
    <section aria-labelledby="hero-h" className="relative isolate overflow-hidden pt-20 md:pt-28">
      {/* the violet light the product's action colour casts behind the window */}
      <div aria-hidden className="absolute inset-x-0 top-0 -z-10 h-[720px] bg-[radial-gradient(50%_60%_at_50%_0%,color-mix(in_oklab,var(--color-accent)_20%,transparent),transparent_75%)]" />
      <Container>
        <div className="mx-auto max-w-[52rem] text-center">
          <h1 id="hero-h" className="text-display tracking-[-0.04em]">Self-learning log parser for SRE teams</h1>
          <p className="mx-auto mt-6 max-w-[38rem] text-lg leading-relaxed text-muted md:text-xl">
            Parses known lines in under a millisecond, learns new formats on its own and tells you why things broke.
          </p>
          <div className="mt-10 flex flex-wrap justify-center gap-3">
            <ButtonLink href={REPO} arrow>View on GitHub</ButtonLink>
            <ButtonLink href={DOCS} variant="secondary">Read the docs</ButtonLink>
          </div>
        </div>
        <Screenshot {...SHOTS.explorer} priority className="mt-16 md:mt-20" />
      </Container>
    </section>
  )
}
