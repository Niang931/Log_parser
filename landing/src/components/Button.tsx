import type { AnchorHTMLAttributes } from 'react'
import { ArrowIcon } from './icons'

const variants = {
  primary: 'bg-accent text-accent-ink hover:bg-[color-mix(in_oklab,var(--color-accent)_86%,white)]',
  secondary: 'border border-line-strong text-ink hover:border-[color-mix(in_oklab,var(--color-ink)_40%,transparent)] hover:bg-white/4',
}

type Props = AnchorHTMLAttributes<HTMLAnchorElement> & {
  variant?: keyof typeof variants
  arrow?: boolean
}

export function ButtonLink({ variant = 'primary', arrow, className = '', children, ...rest }: Props) {
  return (
    <a
      className={`group inline-flex h-11 items-center justify-center gap-2.5 rounded-sm px-5 text-[15px] font-medium transition-colors duration-200 ${variants[variant]} ${className}`}
      {...rest}
    >
      {children}
      {arrow && <ArrowIcon className="size-3 transition-transform duration-300 ease-out-expo group-hover:translate-x-0.5" />}
    </a>
  )
}
