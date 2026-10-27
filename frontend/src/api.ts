export type Experiment = 'pendulum' | 'free_fall' | 'spring'

export interface Check {
  check: string
  status: 'ok' | 'warn' | 'fail'
  value: number | null
  message: string
}

export interface Measurement {
  name: string
  value: number
  sigma: number
  unit: string
}

export interface Prediction {
  quantity: string
  value: number
  unit: string
}

export interface Run {
  run_id: string
  experiment: Experiment
  concept: string
  prediction?: Prediction
  mass_kg?: number | null
  length_m?: number | null
  analysis: {
    video: { fps: number; width: number; height: number; frames: number }
    sheet: { corners_px: number[][]; scale_sigma: number }
    trajectory: { t: number[]; frames: number[]; x_m: number[]; y_m: number[] }
    fit: {
      measurements: Record<string, Measurement>
      window: [number, number]
      residual_rms_m: number
      model: { frames: number[]; xy_m: number[][] }
      notes: string[]
    }
    ghost: { frames: number[]; xy_px: number[][] }
  }
  track_px?: { frames: number[]; xy: number[][] }
  quality: { overall: string; checks: Check[] }
}

export interface Plan {
  experiment: Experiment
  you_need: string[]
  steps: string[]
  measure: string
}

export interface PreflightResult {
  overall: 'ok' | 'warn' | 'fail'
  checks: Check[]
  sheet_corners_px: number[][] | null
  size: [number, number]
}

export interface ChatReply {
  session_id: string
  answer: string
  tools: { tool: string; arguments: string }[]
  grounding: { grounded: boolean; ungrounded_numbers: string[]; refs_cited: string[] }
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* not JSON */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  runs: () => fetch('/api/runs').then((r) => json<Run[]>(r)),
  run: (id: string) => fetch(`/api/runs/${id}`).then((r) => json<Run>(r)),
  videoUrl: (id: string) => `/api/runs/${id}/video`,
  plan: (e: Experiment) => fetch(`/api/plan/${e}`).then((r) => json<Plan>(r)),
  preflight: (image: Blob) => {
    const body = new FormData()
    body.append('image', image, 'frame.jpg')
    return fetch('/api/preflight', { method: 'POST', body }).then((r) => json<PreflightResult>(r))
  },
  createRun: (form: FormData) => fetch('/api/runs', { method: 'POST', body: form }).then((r) => json<Run>(r)),
  chat: (message: string, sessionId?: string) =>
    fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId }),
    }).then((r) => json<ChatReply>(r)),
}

export const EXPERIMENTS: Record<Experiment, { title: string; blurb: string; predict: { quantity: string; unit: string; label: string }; concept: string }> = {
  pendulum: {
    title: 'Pendulum',
    blurb: 'Measure g from the swing of a weight on a string.',
    predict: { quantity: 'period', unit: 's', label: 'How long will one full swing take?' },
    concept: 'orange ball',
  },
  free_fall: {
    title: 'Free fall',
    blurb: 'Measure g from a dropped ball.',
    predict: { quantity: 'g', unit: 'm/s²', label: 'What value of g will you measure?' },
    concept: 'red ball',
  },
  spring: {
    title: 'Spring',
    blurb: 'Measure a spring constant from a bouncing mass.',
    predict: { quantity: 'k', unit: 'N/m', label: 'How stiff is the spring (k)?' },
    concept: 'green block',
  },
}
