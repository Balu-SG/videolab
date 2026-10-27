import { useEffect, useRef, useState } from 'react'
import { api, type PreflightResult } from '../api'
import { Checks } from './Checks'

/** Live camera view, checked once a second for the A4 sheet, lighting, angle, sharpness and clutter. */
export function Preflight({ onDone }: { onDone: () => void }) {
  const video = useRef<HTMLVideoElement>(null)
  const overlay = useRef<HTMLCanvasElement>(null)
  const [result, setResult] = useState<PreflightResult | null>(null)
  const [cameraError, setCameraError] = useState<string | null>(null)

  useEffect(() => {
    let stream: MediaStream | null = null
    let timer: number | undefined
    let busy = false
    const grab = document.createElement('canvas')

    async function tick() {
      const v = video.current
      if (busy || !v || v.readyState < 2) return
      busy = true
      grab.width = v.videoWidth
      grab.height = v.videoHeight
      grab.getContext('2d')!.drawImage(v, 0, 0)
      const blob = await new Promise<Blob | null>((r) => grab.toBlob(r, 'image/jpeg', 0.85))
      try {
        if (blob) setResult(await api.preflight(blob))
      } catch {
        /* keep the last result; the next tick retries */
      } finally {
        busy = false
      }
    }

    navigator.mediaDevices
      ?.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false })
      .then((s) => {
        stream = s
        if (video.current) video.current.srcObject = s
        timer = window.setInterval(tick, 1000)
      })
      .catch((e: Error) => setCameraError(e.message || 'camera unavailable'))
    if (!navigator.mediaDevices) setCameraError('camera needs HTTPS or localhost')

    return () => {
      window.clearInterval(timer)
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  useEffect(() => {
    const c = overlay.current
    const v = video.current
    if (!c || !v) return
    c.width = v.clientWidth
    c.height = v.clientHeight
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, c.width, c.height)
    const corners = result?.sheet_corners_px
    if (!corners || !result) return
    const s = c.width / result.size[0]
    ctx.strokeStyle = '#38e07b'
    ctx.lineWidth = 3
    ctx.beginPath()
    corners.forEach(([x, y], i) => (i ? ctx.lineTo(x * s, y * s) : ctx.moveTo(x * s, y * s)))
    ctx.closePath()
    ctx.stroke()
  }, [result])

  async function checkPhoto(file: File) {
    setResult(await api.preflight(file))
  }

  return (
    <section className="card">
      <h2>Pre-flight check</h2>
      <p className="muted">Point the camera at your setup. VideoLab checks it before you record.</p>
      {cameraError ? (
        <label className="button secondary">
          Check a photo of the setup instead
          <input type="file" accept="image/*" capture="environment" hidden onChange={(e) => e.target.files?.[0] && checkPhoto(e.target.files[0])} />
        </label>
      ) : (
        <div className="stage">
          <video ref={video} autoPlay playsInline muted />
          <canvas ref={overlay} />
        </div>
      )}
      {result ? <Checks checks={result.checks} /> : <p className="muted">Checking…</p>}
      <button className={result?.overall === 'ok' ? 'button' : 'button secondary'} onClick={onDone}>
        {result?.overall === 'ok' ? 'All good: next' : 'Continue anyway'}
      </button>
    </section>
  )
}
