"""Deterministic quality checks on a run: tracking, calibration and fit. The agent reads these; it doesn't invent them."""

import numpy as np


def _check(name: str, status: str, value, message: str) -> dict:
    return {"check": name, "status": status, "value": value, "message": message}


def check_run(run: dict) -> list[dict]:
    track, analysis = run["track"], run["analysis"]
    out = []
    video = track["video"]
    fps = video["fps"]
    out.append(_check("frame_rate", "ok" if fps >= 200 else "warn" if fps >= 100 else "fail", round(fps, 1),
                      "240 fps slow motion recommended" if fps < 200 else "slow-motion frame rate"))

    pts = sorted((p for p in track["points"] if p["object_id"] == _main_id(track)), key=lambda p: p["frame"])
    coverage = len(pts) / max(1, video["frames"])
    out.append(_check("tracking_coverage", "ok" if coverage >= 0.95 else "warn" if coverage >= 0.8 else "fail",
                      round(coverage, 4), f"object tracked on {100 * coverage:.1f}% of frames"))

    conf = [p["confidence"] for p in pts if p["confidence"] is not None]
    if conf:
        c = float(np.mean(conf))
        out.append(_check("tracking_confidence", "ok" if c >= 0.8 else "warn", round(c, 3), "mean SAM 3.1 confidence"))

    if len(pts) > 2:
        xy = np.array([[p["cx"], p["cy"]] for p in pts])
        frames = np.array([p["frame"] for p in pts])
        step = np.hypot(*np.diff(xy, axis=0).T) / np.diff(frames)
        # Compare each step with its neighbours: a falling ball legitimately speeds up, a tracker swap doesn't ramp.
        padded = np.pad(step, 3, mode="edge")
        local = np.array([np.median(np.r_[padded[i:i + 3], padded[i + 4:i + 7]]) for i in range(len(step))])
        jumps = int(np.sum(step > np.maximum(3 * local, local + 6)))
        out.append(_check("tracking_jumps", "ok" if jumps == 0 else "warn", jumps,
                          "no sudden jumps in the track" if jumps == 0 else f"{jumps} sudden jumps; the tracker may have switched objects"))

    if analysis is None:
        out.append(_check("analysis", "fail", None, "run has not been analysed"))
        return out

    sheet = analysis["sheet"]
    area = sheet["area_frac"]
    out.append(_check("sheet_size", "ok" if area >= 0.015 else "warn", area,
                      f"A4 sheet covers {100 * area:.1f}% of the frame" + ("" if area >= 0.015 else "; move closer or use a larger reference")))
    out.append(_check("calibration_scale", "ok" if sheet["scale_sigma"] <= 0.01 else "warn", sheet["scale_sigma"],
                      f"distance calibration good to about {100 * sheet['scale_sigma']:.2f}%"))

    fit = analysis["fit"]
    rms_mm = 1000 * fit["residual_rms_m"]
    out.append(_check("fit_residual", "ok" if rms_mm <= 2 else "warn" if rms_mm <= 5 else "fail", round(rms_mm, 2),
                      f"motion matches the model to {rms_mm:.2f} mm rms"))

    m = fit["measurements"]
    exp = analysis["experiment"]
    duration = (fit["window"][1] - fit["window"][0]) / fps
    if exp == "pendulum":
        if run.get("length_m") is None:
            rel = m["length"]["sigma"] / m["length"]["value"]
            out.append(_check("length_from_arc", "ok" if rel <= 0.02 else "warn", round(rel, 4),
                              f"length from the swing's arc good to {100 * rel:.1f}%" + ("" if rel <= 0.02 else
                              "; the arc is too flat to pin the length, so measure the string to the bob's centre and enter it, or swing wider")))
        amp = m["amplitude"]["value"]
        out.append(_check("swing_amplitude", "ok" if amp <= 15 else "warn", round(amp, 2),
                          f"mean swing {amp:.1f}°" + ("" if amp <= 15 else "; above 15° the period depends noticeably on amplitude")))
        cycles = duration / m["period"]["value"]
        out.append(_check("cycles_recorded", "ok" if cycles >= 4 else "warn", round(cycles, 1), f"{cycles:.1f} full swings in the fit"))
    elif exp == "free_fall":
        n = fit["window"][1] - fit["window"][0] + 1
        out.append(_check("flight_frames", "ok" if n >= 40 else "warn", n, f"{n} frames of free flight in the fit"))
    else:
        cycles = duration / m["period"]["value"]
        out.append(_check("cycles_recorded", "ok" if cycles >= 4 else "warn", round(cycles, 1), f"{cycles:.1f} oscillations in the fit"))
        if "k" not in m:
            out.append(_check("mass_entered", "warn", None, "enter the hanging mass to get the spring constant"))
    return out


def _main_id(track: dict) -> str:
    ids = [p["object_id"] for p in track["points"]]
    return max(set(ids), key=ids.count) if ids else "0"
