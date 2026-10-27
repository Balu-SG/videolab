# VideoLab

A phone-based physics measurement instrument. Film an experiment in slow motion with an A4 sheet in frame. SAM 3.1 tracks the object, deterministic code turns pixels into metres and fits the physics, and a Muse Spark agent plans, quality-checks and explains the run.

Built for the Meta Global AI Developer Hackathon (Push the Frontier). **All code in this repo was written during the hackathon, starting after the official start on 2026-10-26 (12:00 PM ET).** The commit history shows the build.

## Status

Day 1: SAM 3.1 video tracking, scored against synthetic clips with known ground truth.

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
```

## Run the API

```bash
.venv/bin/uvicorn backend.main:app --reload
```
