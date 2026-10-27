import type { Check } from '../api'

const ICON = { ok: '✓', warn: '!', fail: '✕' } as const

export function Checks({ checks }: { checks: Check[] }) {
  return (
    <ul className="checks">
      {checks.map((c) => (
        <li key={c.check} className={`check ${c.status}`}>
          <span className="check-icon" aria-label={c.status}>{ICON[c.status]}</span>
          <span>{c.message}</span>
        </li>
      ))}
    </ul>
  )
}
