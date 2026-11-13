import { ExpandIcon } from './icons'

export type Shot = { src: string; width: number; height: number; alt: string }

type Props = Shot & {
  caption?: string
  priority?: boolean
  className?: string
}

/** A screenshot of logpipe's UI at its own aspect ratio: never cropped, links to the full-size file. */
export function Screenshot({ src, width, height, alt, caption, priority, className = '' }: Props) {
  return (
    <figure className={className}>
      <a
        href={src}
        target="_blank"
        rel="noopener"
        className="group relative block overflow-hidden rounded-lg border border-line bg-sheet shadow-[0_30px_80px_-30px_rgb(0_0_0/0.75)] focus-visible:outline-offset-2"
      >
        <img
          src={src}
          width={width}
          height={height}
          alt={alt}
          loading={priority ? 'eager' : 'lazy'}
          fetchPriority={priority ? 'high' : undefined}
          decoding="async"
          className="block h-auto w-full"
        />
        <span className="absolute right-2 bottom-2 inline-flex sm:right-3 sm:bottom-3 items-center gap-1.5 rounded-sm border border-line-strong bg-bg/90 px-2.5 py-1.5 text-xs font-medium text-ink opacity-100 transition-opacity duration-200 md:opacity-0 md:group-hover:opacity-100 md:group-focus-visible:opacity-100">
          <ExpandIcon className="size-3.5" />
          <span className="max-sm:sr-only">Full size</span>
        </span>
      </a>
      {caption && <figcaption className="mt-4 max-w-[70ch] text-sm leading-relaxed text-muted">{caption}</figcaption>}
    </figure>
  )
}
