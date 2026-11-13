import { useState } from 'react'
import { ExpandIcon } from './icons'
import { Lightbox } from './Lightbox'

export type Shot = { src: string; width: number; height: number; alt: string; title: string }

type Props = Shot & {
  caption?: string
  priority?: boolean
  className?: string
}

/** A screenshot of logpipe's UI in a window frame titled with the page it shows. Never cropped:
    the image keeps its own aspect ratio. Clicking it opens it full size. */
export function Screenshot({ caption, priority, className = '', ...shot }: Props) {
  const [open, setOpen] = useState(false)
  const { src, width, height, alt, title } = shot
  return (
    <figure className={className}>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-label={`View full size: ${alt}`}
        className="group relative block w-full cursor-zoom-in overflow-hidden rounded-lg border border-line-strong bg-sheet text-left shadow-[0_30px_80px_-30px_rgb(0_0_0/0.8)] focus-visible:outline-offset-2"
      >
        <div className="flex h-8 items-center gap-2 border-b border-line bg-[#0f0f14] px-3 text-xs text-muted">
          <svg viewBox="0 0 24 24" className="size-3 flex-none text-accent" fill="currentColor" aria-hidden>
            <path d="M2.5 21.5V12a9.5 9.5 0 0 1 19 0v9.5h-6V12a3.5 3.5 0 0 0-7 0v9.5z" />
          </svg>
          <span className="truncate">logpipe · {title}</span>
          <span className="ml-auto inline-flex flex-none items-center gap-1.5 text-ink opacity-100 transition-opacity duration-200 md:opacity-0 md:group-hover:opacity-100 md:group-focus-visible:opacity-100">
            <ExpandIcon className="size-3.5" />
            <span className="max-sm:sr-only">Full size</span>
          </span>
        </div>
        <img
          src={src}
          width={width}
          height={height}
          alt=""
          loading={priority ? 'eager' : 'lazy'}
          fetchPriority={priority ? 'high' : undefined}
          decoding="async"
          className="block h-auto w-full"
        />
      </button>
      {caption && <figcaption className="mt-4 max-w-[70ch] text-sm leading-relaxed text-muted">{caption}</figcaption>}
      <Lightbox {...shot} open={open} onClose={() => setOpen(false)} />
    </figure>
  )
}
