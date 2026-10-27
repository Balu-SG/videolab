"""Run the physics fit on a synthetic clip and compare the measured constants with the truth.

Uses the true pixel track (isolates geometry + fitting) and, if given, a SAM track (the full pipeline).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.geometry import detect_sheet  # noqa: E402
from backend.physics import fit_free_fall, fit_pendulum, fit_spring  # noqa: E402
from backend.trajectory import from_track  # noqa: E402


def truth_track(truth: dict) -> dict:
    return {
        "video": {"fps": truth["fps"]},
        "points": [
            {"frame": i, "object_id": "0", "cx": x, "cy": y, "confidence": 1.0}
            for i, (x, y) in enumerate(truth["track_px"])
        ],
    }


def fit(experiment: str, traj, truth: dict):
    if experiment == "pendulum":
        return fit_pendulum(traj)
    if experiment == "free_fall":
        return fit_free_fall(traj)
    return fit_spring(traj, mass_kg=truth["mass_kg"])


def expected(experiment: str, truth: dict) -> dict:
    if experiment == "pendulum":
        return {"g": truth["g"], "length": truth["length_m"], "period": truth["period_true_s"]}
    if experiment == "free_fall":
        return {"g": truth["g"]}
    return {"k": truth["k_n_per_m"], "period": truth["period_true_s"]}


def report(label: str, result, truth_values: dict) -> None:
    print(f"  {label}: frames {result.window[0]}-{result.window[1]}, fit residual {1000 * result.residual_rms_m:.2f} mm rms")
    for key, true in truth_values.items():
        m = result.measurements[key]
        err = 100 * (m.value - true) / true
        pull = (m.value - true) / m.sigma if m.sigma else float("inf")
        print(f"    {key:7s} {m.value:.4f} ± {m.sigma:.4f} {m.unit}  (true {true}, error {err:+.2f}%, {abs(pull):.1f} sigma)")
    for note in result.notes:
        print(f"    note: {note}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("truth", type=Path)
    ap.add_argument("--track", type=Path)
    args = ap.parse_args()

    truth = json.loads(args.truth.read_text())
    experiment = truth["experiment"]
    n = len(truth["track_px"])
    sheet = detect_sheet(str(args.video), frames=(0, n // 3, 2 * n // 3))
    if sheet is None:
        sys.exit("A4 sheet not found")

    print(f"{experiment}:")
    want = expected(experiment, truth)
    report("true pixels", fit(experiment, from_track(truth_track(truth), sheet.image_to_m, scale_sigma=sheet.scale_sigma), truth), want)
    if args.track:
        report("SAM 3.1", fit(experiment, from_track(json.loads(args.track.read_text()), sheet.image_to_m, scale_sigma=sheet.scale_sigma), truth), want)


if __name__ == "__main__":
    main()
