import asyncio
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import runs, sam
from .agent import LabAgent
from .config import OUTPUT_DIR
from .pipeline import EXPERIMENTS, AnalysisError, analyze
from .tools import check_quality

app = FastAPI(title="VideoLab")
UPLOAD_DIR = OUTPUT_DIR / "uploads"
_sessions: dict[str, LabAgent] = {}


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/runs")
def list_runs() -> list[dict]:
    return runs.list_runs()


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    try:
        run = runs.load(run_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    run.pop("track", None)  # large; the analysis has everything the UI draws
    run["quality"] = check_quality(run_id)
    return run


@app.get("/runs/{run_id}/video")
def get_video(run_id: str) -> FileResponse:
    try:
        run = runs.load(run_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return FileResponse(run["video"])


@app.post("/runs")
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
    if experiment not in EXPERIMENTS:
        raise HTTPException(400, f"experiment must be one of {EXPERIMENTS}")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(video.filename or "clip.mp4").suffix.lower() or ".mp4"
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    info = sam.probe_video(path)
    file_id = await sam.upload_video(path)
    points, meta = await sam.track(file_id, concept)
    if points:
        first = min(points, key=lambda p: p.frame)
        points, meta = await sam.track_box(file_id, (info["width"], info["height"]), first.frame, first.box, first.object_id)
    track = sam.to_json(points, info, concept, file_id, meta)
    try:
        analysis = await asyncio.to_thread(analyze, str(path), track, experiment, length_m=length_m, mass_kg=mass_kg)
    except AnalysisError as e:
        raise HTTPException(422, str(e))

    run_meta = {"video": str(path), "experiment": experiment, "concept": concept, "length_m": length_m, "mass_kg": mass_kg}
    if predict_quantity and predict_value is not None:
        run_meta["prediction"] = {"quantity": predict_quantity, "value": predict_value, "unit": predict_unit or ""}
    run_id = runs.new_run_id()
    runs.save(run_id, run_meta, track, analysis)
    return get_run(run_id)


class ChatIn(BaseModel):
    message: str
    session_id: str | None = None


@app.post("/chat")
async def chat(body: ChatIn) -> dict:
    session_id = body.session_id or uuid.uuid4().hex
    agent = _sessions.setdefault(session_id, LabAgent())
    result = await asyncio.to_thread(agent.ask, body.message)
    return {"session_id": session_id, **result}
