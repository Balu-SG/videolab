import { useEffect, useState } from 'react'
import { api, EXPERIMENTS, type Experiment, type Plan, type Run } from './api'
import { Chat } from './components/Chat'
import { Preflight } from './components/Preflight'
import { RunView } from './components/RunView'
import { TapPicker, type Tap } from './components/TapPicker'

type Step = 'home' | 'plan' | 'preflight' | 'predict' | 'record' | 'result'

const STEPS: { id: Step; label: string }[] = [
  { id: 'plan', label: 'Plan' },
  { id: 'preflight', label: 'Check' },
  { id: 'predict', label: 'Predict' },
  { id: 'record', label: 'Record' },
  { id: 'result', label: 'Explain' },
]

export default function App() {
  const [step, setStep] = useState<Step>('home')
  const [experiment, setExperiment] = useState<Experiment>('pendulum')
  const [plan, setPlan] = useState<Plan | null>(null)
  const [prediction, setPrediction] = useState('')
  const [mass, setMass] = useState('')
  const [length, setLength] = useState('')
  const [concept, setConcept] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [tap, setTap] = useState<Tap | null>(null)
  const [uploading, setUploading] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [run, setRun] = useState<Run | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const [selectedRef, setSelectedRef] = useState<string | null>(null)
  const [opener, setOpener] = useState<string | null>(null)
  const info = EXPERIMENTS[experiment]

  useEffect(() => {
    if (step === 'home') api.runs().then(setRuns).catch(() => setRuns([]))
  }, [step])

  useEffect(() => {
    if (!uploading) return
    const start = Date.now()
    const t = window.setInterval(() => setElapsed(Math.round((Date.now() - start) / 1000)), 500)
    return () => window.clearInterval(t)
  }, [uploading])

  function start(e: Experiment) {
    setExperiment(e)
    setPlan(null)
    setPrediction('')
    setConcept(EXPERIMENTS[e].concept)
    setFile(null)
    setTap(null)
    setError(null)
    api.plan(e).then(setPlan).catch(() => setPlan(null))
    setStep('plan')
  }

  async function openRun(id: string, withOpener: boolean) {
    const r = await api.run(id)
    setRun(r)
    setSelectedRef(null)
    setOpener(
      withOpener && r.prediction
        ? `I filmed run ${r.run_id} and predicted the ${r.prediction.quantity} would be ${r.prediction.value} ${r.prediction.unit}. What did I actually measure, is the run trustworthy, and why was my prediction off?`
        : `Explain what run ${r.run_id} measured and whether the run is trustworthy.`,
    )
    setStep('result')
  }

  async function upload() {
    if (!file) return
    const form = new FormData()
    form.append('video', file)
    form.append('experiment', experiment)
    form.append('concept', concept || info.concept)
    if (tap) {
      form.append('tap_time', String(tap.time))
      form.append('tap_x', String(tap.x))
      form.append('tap_y', String(tap.y))
    }
    if (mass) form.append('mass_kg', mass)
    if (length) form.append('length_m', length)
    if (prediction) {
      form.append('predict_quantity', info.predict.quantity)
      form.append('predict_value', prediction)
      form.append('predict_unit', info.predict.unit)
    }
    setUploading(true)
    setError(null)
    try {
      const { job_id } = await api.createRun(form)
      let job = await api.job(job_id)
      while (job.status === 'running') {
        await new Promise((resolve) => window.setTimeout(resolve, 2000))
        job = await api.job(job_id)
      }
      if (job.status === 'failed' || !job.run_id) throw new Error(job.error ?? 'measurement failed')
      await openRun(job.run_id, true)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setUploading(false)
    }
  }

  const stage = elapsed < 8 ? 'Uploading the clip' : elapsed < 60 ? 'SAM 3.1 is finding and tracking the object' : 'Refining the track and fitting the physics'

  return (
    <main>
      <header>
        <button className="brand" onClick={() => setStep('home')}>VideoLab</button>
        <span className="muted">a physics lab in your phone</span>
      </header>

      {step !== 'home' && (
        <ol className="steps">
          {STEPS.map((s) => (
            <li key={s.id} className={s.id === step ? 'on' : ''}>{s.label}</li>
          ))}
        </ol>
      )}

      {step === 'home' && (
        <>
          <section className="hero">
            <h1>Film it in slow motion. Measure real physics.</h1>
            <p className="muted">SAM 3.1 tracks the motion, an A4 sheet calibrates distance, and every number comes from a physics fit you can check.</p>
          </section>
          <div className="experiments">
            {(Object.keys(EXPERIMENTS) as Experiment[]).map((e) => (
              <button key={e} className="experiment" onClick={() => start(e)}>
                <strong>{EXPERIMENTS[e].title}</strong>
                <span className="muted">{EXPERIMENTS[e].blurb}</span>
              </button>
            ))}
          </div>
          {runs.length > 0 && (
            <section className="card">
              <h2>Your runs</h2>
              <ul className="run-list">
                {runs.slice().reverse().map((r) => (
                  <li key={r.run_id}>
                    <button onClick={() => openRun(r.run_id, true)}>
                      <span>{r.run_id} · {EXPERIMENTS[r.experiment].title}</span>
                      {r.prediction && <span className="muted">predicted {r.prediction.value} {r.prediction.unit}</span>}
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}

      {step === 'plan' && (
        <section className="card">
          <h2>{info.title}: plan</h2>
          {plan ? (
            <>
              <p>{plan.measure}.</p>
              <h3>You need</h3>
              <ul>{plan.you_need.map((x) => <li key={x}>{x}</li>)}</ul>
              <h3>Set up</h3>
              <ol>{plan.steps.map((x) => <li key={x}>{x}</li>)}</ol>
            </>
          ) : (
            <p className="muted">Loading…</p>
          )}
          <button className="button" onClick={() => setStep('preflight')}>Check my setup</button>
        </section>
      )}

      {step === 'preflight' && <Preflight onDone={() => setStep('predict')} />}

      {step === 'predict' && (
        <section className="card">
          <h2>Predict first</h2>
          <p className="muted">Commit to a guess before you film. VideoLab will explain the gap.</p>
          <label className="field">
            {info.predict.label}
            <span className="with-unit">
              <input inputMode="decimal" value={prediction} onChange={(e) => setPrediction(e.target.value)} placeholder="your guess" />
              <span>{info.predict.unit}</span>
            </span>
          </label>
          {experiment === 'spring' && (
            <label className="field">
              Hanging mass
              <span className="with-unit"><input inputMode="decimal" value={mass} onChange={(e) => setMass(e.target.value)} placeholder="0.2" /><span>kg</span></span>
            </label>
          )}
          {experiment === 'pendulum' && (
            <label className="field">
              String length, pivot to the bob's centre (recommended)
              <span className="with-unit"><input inputMode="decimal" value={length} onChange={(e) => setLength(e.target.value)} placeholder="0.50" /><span>m</span></span>
              <span className="muted small">The period is always precise. Without the length, VideoLab estimates it from the curve of the swing, which is much less certain for short or narrow swings.</span>
            </label>
          )}
          <button className="button" disabled={!prediction || (experiment === 'spring' && !mass)} onClick={() => setStep('record')}>Lock in my prediction</button>
        </section>
      )}

      {step === 'record' && (
        <section className="card">
          <h2>Record</h2>
          <p className="muted">Film in slow motion (240 fps) with the phone held sideways and steady. On iPhone, share the clip with "All Photos Data" so the slow-motion frames are kept.</p>
          <label className="field">
            Slow-motion video
            <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>
          {file && <TapPicker file={file} tap={tap} onTap={setTap} />}
          {!tap && (
            <label className="field">
              Or describe what is moving (colour + shape)
              <input value={concept} onChange={(e) => setConcept(e.target.value)} placeholder={info.concept} />
            </label>
          )}
          {error && <p className="error">{error}</p>}
          <button className="button" disabled={!file || uploading} onClick={upload}>
            {uploading ? `${stage}… ${elapsed}s` : 'Measure'}
          </button>
        </section>
      )}

      {step === 'result' && run && (
        <>
          <section className="card">
            <h2>{EXPERIMENTS[run.experiment].title} · {run.run_id}</h2>
            <RunView run={run} selectedRef={selectedRef} />
          </section>
          <Chat opener={opener} onRef={setSelectedRef} />
        </>
      )}
    </main>
  )
}
