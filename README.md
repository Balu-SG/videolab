# VideoLab

A phone-based physics measurement instrument. Film an experiment in slow motion with an A4 sheet in frame. SAM 3.1 tracks the object, deterministic code turns pixels into metres and fits the physics, and a Muse Spark agent plans, quality-checks and explains the run.

Built for the Meta Global AI Developer Hackathon (Push the Frontier). **All code in this repo was written during the hackathon, starting after the official start on 2026-10-26 (12:00 PM ET).** The commit history shows how it was built.

## What it does

- SAM 3.1 video tracking, scored against synthetic clips with known ground truth.
- A4 sheet detection (threshold sweep, midpoint re-cut, sub-pixel edge-line corners) and the pixel -> metre homography. `scripts/check_geometry.py` reports corner, offset and scale error.
- Physics fits (pendulum: circle-fit length + damped, amplitude-corrected period; free fall: auto release/impact + parabola; spring: damped oscillation -> k) with calibration-aware uncertainties, plus `backend/pipeline.py` (track -> sheet -> metres -> fit -> ghost overlay in pixels). On the synthetic clips with SAM 3.1 tracks: pendulum g 9.76 ± 0.07, free-fall g 9.71 ± 0.06, spring k 20.00 ± 0.02 N/m (true 9.81 and 20.0).
- Muse Spark lab partner (`backend/agent.py`, Responses API tool loop) with five deterministic tools (`plan_experiment`, `check_quality`, `analyze_run`, `compare_runs`, `explain_physics`). Every number carries a ref like `[r1.g]`, and `backend/grounding.py` flags any number in the answer that no tool returned. FastAPI: `/runs`, `/runs/{id}`, `/runs/{id}/video`, `POST /runs` (upload -> SAM -> analysis), `POST /chat`.
- Tap to track: on the Record step the user taps the moving object on any frame and SAM 3.1 tracks from that point prompt. Small or plain objects that a text concept misses (a marble seen from 2 m) are still found; the text concept remains as the fallback.
- Accuracy benchmark (`scripts/benchmark.py`, clips from `scripts/synth.py`): 20 synthetic runs with exact ground truth across camera angle, distance, dim light, noise and motion blur. All 20 measured, all within 2σ; spring k within 0.03%, free-fall g within 1.3%, pendulum g within 0.02% with the string length entered. Results in [docs/ACCURACY.md](docs/ACCURACY.md).
- Phone-first React app (`frontend/`): plan → live pre-flight camera check (A4 sheet, lighting, sharpness from the sheet's edge width, camera angle, clutter) → predict → record → results with the ghost overlay, measurement cards, fit chart, run quality, and the lab-partner chat. Every cited number is a chip that jumps the video to its frames and opens its evidence.

## Setup

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
# MODEL_API_KEY goes in ../.env (outside the repo) or in the environment
```

## Track a clip and score it

```bash
.venv/bin/python scripts/track_clip.py ../clips/synthetic_pendulum.mp4 --concept "pendulum bob"
.venv/bin/python scripts/score_track.py output/synthetic_pendulum.track.json ../clips/synthetic_pendulum.json
.venv/bin/python scripts/check_geometry.py ../clips/synthetic_pendulum.mp4 ../clips/synthetic_pendulum.json
.venv/bin/python scripts/check_physics.py ../clips/synthetic_pendulum.mp4 ../clips/synthetic_pendulum.json --track output/synthetic_pendulum.track.json
```

## Run the API

```bash
.venv/bin/uvicorn backend.main:app --port 8010 --reload
# front end dev server (proxies /api to 8010): cd frontend && npm run dev
# production: cd frontend && npm run build, then uvicorn serves frontend/dist at /
```

## Deploy (Modal)

The phone camera needs HTTPS, so the hosted app runs on Modal with a volume for runs and uploads. Measuring a clip is a background job (`POST /api/runs` returns a `job_id`, then poll `GET /api/jobs/{job_id}`) because SAM tracking takes minutes. Uploads are capped at 300 MB and 3,600 frames, with daily limits on new runs and chat questions (`VIDEOLAB_MAX_*` in `backend/config.py`).

```bash
cd frontend && npm run build && cd ..
.venv/bin/modal secret create videolab-meta MODEL_API_KEY=...   # once
.venv/bin/modal deploy modal_app.py
```

## Ask the lab partner

```bash
.venv/bin/python scripts/new_run.py ../clips/synthetic_pendulum.mp4 pendulum --concept "orange ball" --predict period 1.5 s
.venv/bin/python scripts/ask.py "I filmed run r1 and predicted 1.5 s. Why was I off?"
```
