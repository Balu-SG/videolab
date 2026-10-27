"""One run end to end: SAM track + video -> sheet calibration -> trajectory -> physics fit -> overlay data."""

import numpy as np

from .geometry import detect_sheet, to_metres
from .physics import FitResult, fit_free_fall, fit_pendulum, fit_spring
from .trajectory import from_track

EXPERIMENTS = ("pendulum", "free_fall", "spring")


class AnalysisError(ValueError):
    pass


def analyze(
    video_path: str,
    track: dict,
    experiment: str,
    *,
    length_m: float | None = None,
    mass_kg: float | None = None,
) -> dict:
    if experiment not in EXPERIMENTS:
        raise AnalysisError(f"unknown experiment {experiment!r}; expected one of {EXPERIMENTS}")
    if not track["points"]:
        raise AnalysisError("SAM found no object in this clip")

    n = track["video"]["frames"]
    sheet = detect_sheet(video_path, frames=(0, n // 3, 2 * n // 3))
    if sheet is None:
        raise AnalysisError("no A4 sheet found; put a plain A4 sheet flat in the plane of motion, fully in frame")

    traj = from_track(track, sheet.image_to_m, scale_sigma=sheet.scale_sigma)
    if experiment == "pendulum":
        fit: FitResult = fit_pendulum(traj, length_m=length_m)
    elif experiment == "free_fall":
        fit = fit_free_fall(traj)
    else:
        fit = fit_spring(traj, mass_kg=mass_kg)

    # The ghost overlay is drawn in video pixels: map the fitted ideal motion back through the inverse homography.
    ghost_px = to_metres(np.linalg.inv(sheet.image_to_m), fit.model_xy_m)
    return {
        "experiment": experiment,
        "video": track["video"],
        "sheet": sheet.to_dict(),
        "trajectory": traj.to_dict(),
        "fit": fit.to_dict(),
        "ghost": {"frames": fit.model_frames.tolist(), "xy_px": ghost_px.round(2).tolist()},
    }
