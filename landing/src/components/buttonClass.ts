const variants = {
  primary: 'bg-accent text-accent-ink hover:bg-[color-mix(in_oklab,var(--color-accent)_86%,white)]',
  secondary: 'border border-line-strong text-ink hover:border-[color-mix(in_oklab,var(--color-ink)_40%,transparent)] hover:bg-white/4',
}

export type ButtonVariant = keyof typeof variants

/** Shared by link buttons and real <button>s so both look the same. */
export const buttonClass = (variant: ButtonVariant = 'primary') =>
  `group inline-flex h-11 items-center justify-center gap-2.5 rounded-sm px-5 text-[15px] font-medium transition-colors duration-200 ${variants[variant]}`
