/** The logpipe word with its arc mark (the favicon's shape, in the accent colour). */
export function Logo() {
  return (
    <a href="#top" className="inline-flex items-center gap-2.5 text-lg font-semibold tracking-[-0.02em]">
      <svg viewBox="0 0 24 24" className="size-5 text-accent" fill="currentColor" aria-hidden>
        <path d="M2.5 21.5V12a9.5 9.5 0 0 1 19 0v9.5h-6V12a3.5 3.5 0 0 0-7 0v9.5z" />
      </svg>
      logpipe
    </a>
  )
}
