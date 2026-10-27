import { useEffect, useRef, useState } from 'react'
import { api, type Run } from '../api'
import { Chart } from './Chart'
import { Checks } from './Checks'

const RATES = [1, 0.25, 0.1]

export function formatMeasurement(value: number, sigma: number) {
  if (!sigma) return value.toPrecision(4)
  const digits = Math.max(0, 1 - Math.floor(Math.log10(Math.abs(sigma))))
  return `${value.toFixed(digits)} ± ${sigma.toFixed(digits)}`
}

/** Measurement key a ref points at: "r1.g" -> "g", "r1.g_vs_standard" -> "g", "r1.prediction_gap" -> "prediction". */
export function refTarget(ref: string): string {
  const key = ref.split('.')[1] ?? ''
  if (key === 'prediction_gap') return 'prediction'
  return key.replace(/_vs_standard$/, '')
}

export function RunView({ run, selectedRef }: { run: Run; selectedRef: string | null }) {
  const video = useRef<HTMLVideoElement>(null)
  const overlay = useRef<HTMLCanvasElement>(null)
  const [rate, setRate] = useState(0.25)
  const [showGhost, setShowGhost] = useState(true)
  const { analysis } = run
  const fps = analysis.video.fps
  const selected = selectedRef?.startsWith(run.run_id + '.') ? refTarget(selectedRef) : null

  useEffect(() => {
    if (video.current) video.current.playbackRate = rate
  }, [rate])

  // Clicking a grounded number jumps the video to the frames that produced it.
  useEffect(() => {
    if (selected && video.current) {
      video.current.currentTime = analysis.fit.window[0] / fps
      video.current.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }, [selected, analysis.fit.window, fps])

  useEffect(() => {
    const v = video.current
    const c = overlay.current
    if (!v || !c) return
    const ghostStart = analysis.ghost.frames[0]
    const trackIndex = new Map(run.track_px?.frames.map((f, i) => [f, i]))
    let handle = 0

    function draw() {
      if (!v || !c) return
      c.width = v.clientWidth
      c.height = v.clientHeight
      const s = c.width / analysis.video.width
      const ctx = c.getContext('2d')!
      ctx.clearRect(0, 0, c.width, c.height)
      const frame = Math.round(v.currentTime * fps)

      ctx.strokeStyle = 'rgba(255,255,255,0.75)'
      ctx.lineWidth = 2
      ctx.beginPath()
      analysis.sheet.corners_px.forEach(([x, y], i) => (i ? ctx.lineTo(x * s, y * s) : ctx.moveTo(x * s, y * s)))
      ctx.closePath()
      ctx.stroke()

      const ti = trackIndex.get(frame)
      if (ti !== undefined && run.track_px) {
        const [x, y] = run.track_px.xy[ti]
        ctx.fillStyle = '#ffb020'
        ctx.beginPath()
        ctx.arc(x * s, y * s, 4, 0, Math.PI * 2)
        ctx.fill()
      }
      const gi = frame - ghostStart
      if (showGhost && gi >= 0 && gi < analysis.ghost.xy_px.length) {
        // The fitted ideal motion: if the physics model is right, this ring sits on the real object.
        const [x, y] = analysis.ghost.xy_px[gi]
        ctx.strokeStyle = '#38e07b'
        ctx.lineWidth = 3
        ctx.beginPath()
        ctx.arc(x * s, y * s, 14, 0, Math.PI * 2)
        ctx.stroke()
      }
      handle = requestAnimationFrame(draw)
    }
    handle = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(handle)
  }, [analysis, run.track_px, fps, showGhost])

  const pred = run.prediction
  const predMeasured = pred ? analysis.fit.measurements[pred.quantity] : undefined

  return (
    <div className="run">
      <div className="stage">
        <video ref={video} src={api.videoUrl(run.run_id)} controls playsInline muted loop preload="auto" />
        <canvas ref={overlay} />
      </div>
      <div className="row">
        <span className="muted">Speed</span>
        {RATES.map((r) => (
          <button key={r} className={`pill ${r === rate ? 'on' : ''}`} onClick={() => setRate(r)}>{r}×</button>
        ))}
        <button className={`pill ${showGhost ? 'on' : ''}`} onClick={() => setShowGhost(!showGhost)}>Ghost</button>
      </div>
      <p className="legend"><span className="dot amber" /> tracked by SAM 3.1 <span className="dot green ring" /> fitted ideal motion</p>

      {pred && predMeasured && (
        <div className={`measure prediction ${selected === 'prediction' ? 'selected' : ''}`}>
          <div className="measure-name">Your prediction vs measured {pred.quantity}</div>
          <div className="measure-value">
            {pred.value} {pred.unit} → {formatMeasurement(predMeasured.value, predMeasured.sigma)} {predMeasured.unit}
          </div>
        </div>
      )}
      <div className="measures">
        {Object.entries(analysis.fit.measurements).map(([key, m]) => (
          <div key={key} className={`measure ${selected === key ? 'selected' : ''} ${['g', 'k'].includes(key) ? 'primary' : ''}`}>
            <div className="measure-name">{m.name}</div>
            <div className="measure-value">{formatMeasurement(m.value, m.sigma)} <span className="unit">{m.unit}</span></div>
            {selected === key && (
              <div className="measure-src">
                From frames {analysis.fit.window[0]}–{analysis.fit.window[1]} ({(analysis.fit.window[0] / fps).toFixed(2)}–{(analysis.fit.window[1] / fps).toFixed(2)} s), fit residual {(1000 * analysis.fit.residual_rms_m).toFixed(2)} mm
              </div>
            )}
          </div>
        ))}
      </div>
      <Chart run={run} highlight={selected ? analysis.fit.window : null} />
      <h3>Run quality</h3>
      <Checks checks={run.quality.checks} />
    </div>
  )
}
