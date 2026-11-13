import type { HTMLAttributes } from 'react'

/** A dark code panel in the product's own surface: example lines, the terminal, the learning demo. */
export function Panel({ className = '', ...rest }: HTMLAttributes<HTMLDivElement>) {
  return <div className={`rounded-lg border border-line bg-sheet font-mono text-[13px] leading-[1.6] ${className}`} {...rest} />
}

/** An extracted field name, tinted like the Templates page highlights them. */
export const Field = ({ children }: { children: string }) => (
  <span className="rounded-[3px] bg-route-store/20 px-1 text-field">{children}</span>
)

/** Key/value rows in two columns (field → value, mask → key). */
export function KeyValues({ rows, keyClass = 'text-muted', valueClass = 'text-ink' }: { rows: [string, string][]; keyClass?: string; valueClass?: string }) {
  return (
    <dl className="grid gap-1.5">
      {rows.map(([k, v]) => (
        <div key={k} className="grid grid-cols-[minmax(0,11rem)_minmax(0,1fr)] gap-3">
          <dt className={`break-words ${keyClass}`}>{k}</dt>
          <dd className={`break-words ${valueClass}`}>{v}</dd>
        </div>
      ))}
    </dl>
  )
}
