import asyncio
import logging
import uuid
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import runs, sam
from .agent import LabAgent
from .config import MAX_CHATS_PER_DAY, MAX_FRAMES, MAX_RUNS_PER_DAY, MAX_UPLOAD_MB, OUTPUT_DIR, REPO_DIR
from .pipeline import EXPERIMENTS, AnalysisError, analyze
from .preflight import check_frame
from .tools import check_quality, plan_experiment

app = FastAPI(title="VideoLab")
api = APIRouter(prefix="/api")
UPLOAD_DIR = OUTPUT_DIR / "uploads"
_sessions: dict[str, LabAgent] = {}
_jobs: dict[str, dict] = {}
_tasks: set[asyncio.Task] = set()
_chat_day: list = [None, 0]  # [date, questions answered that day]


@api.get("/health")
def health() -> dict:
    return {"ok": True}


@api.get("/plan/{experiment}")
def plan(experiment: str) -> dict:
    result = plan_experiment(experiment)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@api.post("/preflight")
async def preflight(image: UploadFile = File(...)) -> dict:
    data = np.frombuffer(await image.read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "could not read the image")
    return await asyncio.to_thread(check_frame, frame)


@api.get("/runs")
def list_runs() -> list[dict]:
    return runs.list_runs()


@api.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    try:
        run = runs.load(run_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    track = run.pop("track", None)  # large; the UI only needs the tracked centre per frame
    if track and track["points"]:
        main_id = _main_object(track)
        pts = sorted((p for p in track["points"] if p["object_id"] == main_id), key=lambda p: p["frame"])
        run["track_px"] = {"frames": [p["frame"] for p in pts], "xy": [[round(p["cx"], 1), round(p["cy"], 1)] for p in pts]}
    run["quality"] = check_quality(run_id)
    return run


@api.get("/runs/{run_id}/video")
def get_video(run_id: str) -> FileResponse:
    try:
        run = runs.load(run_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return FileResponse(runs.video_path(run))


@api.post("/runs")
async def create_run(
    video: UploadFile = File(...),
    experiment: str = Form(...),
    concept: str = Form(...),
    length_m: float | None = Form(None),
    mass_kg: float | None = Form(None),
    predict_quantity: str | None = Form(None),
    predict_value: float | None = Form(None),
    predict_unit: str | None = Form(None),
) -> dict:
    """Starts a measurement job. Tracking takes minutes, longer than a hosted request may stay open,
    so the client polls GET /jobs/{job_id}."""
    if experiment not in EXPERIMENTS:
        raise HTTPException(400, f"experiment must be one of {EXPERIMENTS}")
    if runs.created_today() + _active_jobs() >= MAX_RUNS_PER_DAY:
        raise HTTPException(429, "VideoLab has measured its daily limit of clips. Please try again tomorrow.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(video.filename or "clip.mp4").suffix.lower() or ".mp4"
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    with path.open("wb") as f:
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                f.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, f"the clip is over {MAX_UPLOAD_MB} MB")
            f.write(chunk)
    try:
        info = await asyncio.to_thread(sam.probe_video, path)
    except Exception:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "could not read that video")
    if info["frames"] > MAX_FRAMES:
        path.unlink(missing_ok=True)
        raise HTTPException(413, f"the clip has {info['frames']} frames; trim it to {MAX_FRAMES} or fewer (15 s at 240 fps)")

    run_meta = {"video": str(path.relative_to(OUTPUT_DIR)), "experiment": experiment, "concept": concept, "length_m": length_m, "mass_kg": mass_kg}
    if predict_quantity and predict_value is not None:
        run_meta["prediction"] = {"quantity": predict_quantity, "value": predict_value, "unit": predict_unit or ""}
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "running", "run_id": None, "error": None}
    _tasks.add(task := asyncio.create_task(_measure(job_id, path, info, run_meta)))
    task.add_done_callback(_tasks.discard)
    return {"job_id": job_id}


@api.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(404, "unknown job")
    return _jobs[job_id]


async def _measure(job_id: str, path: Path, info: dict, run_meta: dict) -> None:
    job = _jobs[job_id]
    try:
        file_id = await sam.upload_video(path)
        points, meta = await sam.track(file_id, run_meta["concept"])
        if points:
            first = min(points, key=lambda p: p.frame)
            points, meta = await sam.track_box(file_id, (info["width"], info["height"]), first.frame, first.box, first.object_id)
        track = sam.to_json(points, info, run_meta["concept"], file_id, meta)
        analysis = await asyncio.to_thread(
            analyze, str(path), track, run_meta["experiment"], length_m=run_meta["length_m"], mass_kg=run_meta["mass_kg"]
        )
        run_id = runs.new_run_id()
        runs.save(run_id, run_meta, track, analysis)
        on_run_saved()
        job.update(status="done", run_id=run_id)
    except AnalysisError as e:
        job.update(status="failed", error=str(e))
    except Exception as e:  # SAM/API failures: report them instead of leaving the client polling forever
        logging.exception("measurement failed")
        job.update(status="failed", error=f"measurement failed: {e}")


def _active_jobs() -> int:
    return sum(1 for j in _jobs.values() if j["status"] == "running")


def on_run_saved() -> None:
    """Hook for the host to persist new runs (the Modal app commits its volume here)."""


class ChatIn(BaseModel):
    message: str = Field(max_length=2000)
    session_id: str | None = None


@api.post("/chat")
async def chat(body: ChatIn) -> dict:
    today = date.today()
    if _chat_day[0] != today:
        _chat_day[:] = [today, 0]
    if _chat_day[1] >= MAX_CHATS_PER_DAY:
        raise HTTPException(429, "The lab partner has answered its daily limit of questions. Please try again tomorrow.")
    _chat_day[1] += 1
    session_id = body.session_id or uuid.uuid4().hex
    if session_id not in _sessions and len(_sessions) >= 500:
        _sessions.pop(next(iter(_sessions)))  # drop the oldest conversation
    agent = _sessions.setdefault(session_id, LabAgent())
    result = await asyncio.to_thread(agent.ask, body.message)
    return {"session_id": session_id, **result}


def _main_object(track: dict) -> str:
    ids = [p["object_id"] for p in track["points"]]
    return max(set(ids), key=ids.count)


app.include_router(api)

# The built front end, served from the same origin in production.
_DIST = REPO_DIR / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="frontend")
