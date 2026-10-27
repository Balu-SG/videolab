import { useEffect, useRef, useState, type ReactNode } from 'react'
import { api, type ChatReply } from '../api'

interface Message {
  role: 'user' | 'agent'
  text: string
  reply?: ChatReply
}

const REF = /\[((?:r\d+|compare)\.\w+)\]/g

/** Render the agent's text: bullets, **bold**, and [r1.g] refs as clickable chips that open the evidence. */
function renderAnswer(text: string, onRef: (ref: string) => void): ReactNode {
  const inline = (line: string, key: string) => {
    const parts: ReactNode[] = []
    let last = 0
    for (const m of line.matchAll(REF)) {
      parts.push(bold(line.slice(last, m.index), `${key}t${last}`))
      parts.push(
        <button key={`${key}r${m.index}`} className="ref" onClick={() => onRef(m[1])} title="Show where this number comes from">
          {m[1].split('.')[1].replace(/_/g, ' ')}
        </button>,
      )
      last = (m.index ?? 0) + m[0].length
    }
    parts.push(bold(line.slice(last), `${key}t${last}`))
    return parts
  }
  const lines = text.split('\n').filter((l) => l.trim())
  const out: ReactNode[] = []
  let bullets: ReactNode[] = []
  const flush = () => {
    if (bullets.length) out.push(<ul key={`u${out.length}`}>{bullets}</ul>)
    bullets = []
  }
  lines.forEach((l, i) => {
    const b = l.match(/^\s*[*-]\s+(.*)/)
    if (b) bullets.push(<li key={i}>{inline(b[1], `l${i}`)}</li>)
    else {
      flush()
      out.push(<p key={i}>{inline(l, `l${i}`)}</p>)
    }
  })
  flush()
  return out
}

function bold(s: string, key: string): ReactNode {
  return s.split(/\*\*(.+?)\*\*/g).map((part, i) => (i % 2 ? <strong key={`${key}b${i}`}>{part}</strong> : part))
}

export function Chat({ opener, onRef }: { opener: string | null; onRef: (ref: string) => void }) {
  const [messages, setMessages] = useState<Message[]>([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const session = useRef<string | undefined>(undefined)
  const sentOpener = useRef<string | null>(null)

  async function send(text: string) {
    if (!text.trim() || busy) return
    setMessages((m) => [...m, { role: 'user', text }])
    setDraft('')
    setBusy(true)
    try {
      const reply = await api.chat(text, session.current)
      session.current = reply.session_id
      setMessages((m) => [...m, { role: 'agent', text: reply.answer, reply }])
    } catch (e) {
      setMessages((m) => [...m, { role: 'agent', text: `Something went wrong: ${(e as Error).message}` }])
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    if (opener && sentOpener.current !== opener) {
      sentOpener.current = opener
      send(opener)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [opener])

  return (
    <section className="card chat">
      <h2>Lab partner</h2>
      {messages.map((m, i) => (
        <div key={i} className={`msg ${m.role}`}>
          {m.role === 'agent' ? renderAnswer(m.text, onRef) : <p>{m.text}</p>}
          {m.reply && (
            <div className="msg-meta">
              <span>{m.reply.tools.map((t) => t.tool).join(' → ')}</span>
              {m.reply.grounding.grounded ? (
                <span className="badge ok">every number from measured data</span>
              ) : (
                <span className="badge warn">unverified: {m.reply.grounding.ungrounded_numbers.join(', ')}</span>
              )}
            </div>
          )}
        </div>
      ))}
      {busy && <div className="msg agent thinking">Checking the data…</div>}
      <form
        className="chat-input"
        onSubmit={(e) => {
          e.preventDefault()
          send(draft)
        }}
      >
        <input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="Ask about your run…" />
        <button className="button" disabled={busy || !draft.trim()}>Ask</button>
      </form>
    </section>
  )
}
