import { useEffect, useRef } from 'react'
import type { Shot } from './Screenshot'

type Props = Shot & { open: boolean; onClose: () => void }

/** A screenshot at full size over the page. Native <dialog>: Escape closes it and focus stays
    inside; the X and a click on the dark backdrop close it too. */
export function Lightbox({ src, width, height, alt, title, open, onClose }: Props) {
  const ref = useRef<HTMLDialogElement>(null)

  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) d.showModal()
    if (!open && d.open) d.close()
  }, [open])

  return (
    <dialog
      ref={ref}
      aria-label={`${title}, full size`}
      onClose={onClose}
      onClick={e => { if (e.target === e.currentTarget) onClose() }}
      className="m-0 h-dvh max-h-none w-dvw max-w-none bg-transparent p-0 text-ink opacity-0 transition-[opacity,display,overlay] transition-discrete duration-200 backdrop:bg-black/85 open:opacity-100 starting:open:opacity-0"
    >
      <div onClick={e => { if (e.target === e.currentTarget) onClose() }} className="grid h-full place-items-center overflow-auto p-4 pt-16 sm:p-10 sm:pt-16">
        <div className="fixed inset-x-0 top-0 flex items-center gap-4 px-4 py-3 sm:px-6">
          <p className="truncate text-sm text-ink/80">logpipe · {title}</p>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            autoFocus
            className="ml-auto grid size-10 flex-none cursor-pointer place-items-center rounded-full border border-line-strong bg-bg/80 text-ink transition-colors hover:bg-white/10"
          >
            <svg viewBox="0 0 16 16" className="size-4" fill="none" stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" aria-hidden><path d="m4 4 8 8M12 4l-8 8" /></svg>
          </button>
        </div>
        {/* the whole image, scaled down only if the screen is smaller than it */}
        <img
          src={src}
          width={width}
          height={height}
          alt={alt}
          className="h-auto max-h-[calc(100dvh-6rem)] w-auto max-w-full rounded-lg border border-line-strong object-contain"
        />
      </div>
    </dialog>
  )
}
