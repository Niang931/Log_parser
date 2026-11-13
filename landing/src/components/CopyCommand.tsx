import { useState } from 'react'
import { buttonClass } from './buttonClass'
import { CheckIcon, CopyIcon } from './icons'

export function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try { await navigator.clipboard.writeText(command) } catch { return }
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }
  const Icon = copied ? CheckIcon : CopyIcon
  return (
    <button type="button" onClick={copy} className={`${buttonClass('secondary')} cursor-pointer gap-3`}>
      <code className="font-mono text-sm font-normal">
        <span className="text-muted select-none">$ </span>{command}
      </code>
      <Icon className={`size-4 ${copied ? 'text-route-learn' : 'text-muted'}`} />
      <span className="sr-only" aria-live="polite">{copied ? 'Copied' : 'Copy command'}</span>
    </button>
  )
}
