import type { AnchorHTMLAttributes } from 'react'
import { buttonClass, type ButtonVariant } from './buttonClass'
import { ArrowIcon } from './icons'

type Props = AnchorHTMLAttributes<HTMLAnchorElement> & {
  variant?: ButtonVariant
  arrow?: boolean
}

export function ButtonLink({ variant = 'primary', arrow, className = '', children, ...rest }: Props) {
  return (
    <a className={`${buttonClass(variant)} ${className}`} {...rest}>
      {children}
      {arrow && <ArrowIcon className="size-3 transition-transform duration-300 ease-out-expo group-hover:translate-x-0.5" />}
    </a>
  )
}
