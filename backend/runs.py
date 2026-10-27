"""Runs on disk: output/runs/<run_id>/ holds run.json (metadata + prediction), track.json and analysis.json."""

import json
import re
from pathlib import Path

from .config import OUTPUT_DIR

RUNS_DIR = OUTPUT_DIR / "runs"


def _dir(run_id: str) -> Path:
    if not re.fullmatch(r"r\d+", run_id):
        raise KeyError(f"bad run id {run_id!r}")
    return RUNS_DIR / run_id


def new_run_id() -> str:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    taken = [int(p.name[1:]) for p in RUNS_DIR.glob("r*") if p.name[1:].isdigit()]
    return f"r{max(taken, default=0) + 1}"


def save(run_id: str, meta: dict, track: dict | None = None, analysis: dict | None = None) -> None:
    d = _dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({**meta, "run_id": run_id}, indent=1))
    if track is not None:
        (d / "track.json").write_text(json.dumps(track))
    if analysis is not None:
        (d / "analysis.json").write_text(json.dumps(analysis))


def load(run_id: str) -> dict:
    d = _dir(run_id)
    if not (d / "run.json").exists():
        raise KeyError(f"no run {run_id}")
    run = json.loads((d / "run.json").read_text())
    for name in ("track", "analysis"):
        f = d / f"{name}.json"
        run[name] = json.loads(f.read_text()) if f.exists() else None
    return run


def list_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    out = []
    for d in sorted(RUNS_DIR.glob("r*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0):
        if (d / "run.json").exists():
            out.append(json.loads((d / "run.json").read_text()))
    return out
