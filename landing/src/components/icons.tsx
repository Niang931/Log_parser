import type { SVGProps } from 'react'

type IconProps = SVGProps<SVGSVGElement>

const stroke = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.5,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  'aria-hidden': true,
} as const

export const ArrowIcon = (p: IconProps) => (
  <svg viewBox="0 0 12 12" {...stroke} {...p}><path d="M1 6h8M4.5 2.5 8 6l-3.5 3.5" /></svg>
)

export const ExpandIcon = (p: IconProps) => (
  <svg viewBox="0 0 16 16" {...stroke} {...p}><path d="M9.5 2.5h4v4M13.5 2.5 8.5 7.5M6.5 13.5h-4v-4M2.5 13.5l5-5" /></svg>
)

export const MenuIcon = ({ open, ...p }: IconProps & { open: boolean }) => (
  <svg viewBox="0 0 18 18" {...stroke} strokeWidth={1.6} {...p}>
    <path d={open ? 'M4 4l10 10M14 4 4 14' : 'M2 5h14M2 13h14'} />
  </svg>
)

export const PlayIcon = (p: IconProps) => (
  <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden {...p}>
    <path d="M5.25 3.7v8.6c0 .62.68 1 1.2.66l6.5-4.3c.47-.31.47-1 0-1.32l-6.5-4.3c-.52-.34-1.2.04-1.2.66Z" />
  </svg>
)

export const PauseIcon = (p: IconProps) => (
  <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden {...p}>
    <rect x="4.25" y="3.25" width="2.75" height="9.5" rx="0.9" />
    <rect x="9" y="3.25" width="2.75" height="9.5" rx="0.9" />
  </svg>
)
