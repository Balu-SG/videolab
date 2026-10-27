import type { Run } from '../api'

/** Measured position (dots) against the fitted model (line) for the coordinate that moves most. */
export function Chart({ run, highlight }: { run: Run; highlight: [number, number] | null }) {
  const { trajectory, fit, video } = run.analysis
  const axis = run.experiment === 'pendulum' ? 'x' : 'y'
  const W = 640
  const H = 240
  const pad = 34
  const top = 30

  const t = trajectory.t
  const v = axis === 'x' ? trajectory.x_m : trajectory.y_m
  const modelT = fit.model.frames.map((f) => f / video.fps)
  const modelV = fit.model.xy_m.map((p) => (axis === 'x' ? p[0] : p[1]))
  const all = [...v, ...modelV]
  const tMin = Math.min(...t)
  const tMax = Math.max(...t)
  const vMin = Math.min(...all)
  const vMax = Math.max(...all)
  const sx = (x: number) => pad + ((x - tMin) / (tMax - tMin || 1)) * (W - pad - 8)
  const sy = (y: number) => H - pad - ((y - vMin) / (vMax - vMin || 1)) * (H - pad - top)
  const step = Math.max(1, Math.floor(t.length / 400))
  const line = modelT.map((x, i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(modelV[i]).toFixed(1)}`).join('')

  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${axis} position over time, measured and fitted`}>
      {highlight && (
        <rect x={sx(highlight[0] / video.fps)} y={top - 6} width={Math.max(2, sx(highlight[1] / video.fps) - sx(highlight[0] / video.fps))} height={H - pad - top + 6} className="chart-window" />
      )}
      <line x1={pad} y1={H - pad} x2={W - 8} y2={H - pad} className="chart-axis" />
      <line x1={pad} y1={top - 6} x2={pad} y2={H - pad} className="chart-axis" />
      {t.map((x, i) => (i % step === 0 ? <circle key={i} cx={sx(x)} cy={sy(v[i])} r={1.6} className="chart-dot" /> : null))}
      <path d={line} className="chart-line" />
      <text x={pad} y={H - 10} className="chart-label">0 s</text>
      <text x={W - 8} y={H - 10} textAnchor="end" className="chart-label">{tMax.toFixed(2)} s</text>
      <text x={pad} y={14} className="chart-label">{axis} (m) · dots measured · line fitted</text>
    </svg>
  )
}
