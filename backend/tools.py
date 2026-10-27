"""The agent's tools. All numbers come from deterministic code; every measurement carries a ref like "r1.g"."""

import math

import numpy as np

from . import runs
from .quality import check_run

G_STANDARD = 9.81

PROTOCOLS = {
    "pendulum": {
        "you_need": ["string 0.3-1.0 m long", "small dense bob (a coloured ball or nut)", "rigid pivot", "plain A4 sheet", "phone with 240 fps slow motion"],
        "steps": [
            "Tape the A4 sheet flat on the wall directly behind the swing, in the same plane as the motion.",
            "Put the phone on a stand 1-2 m away, square to the wall, with the pivot, the whole swing and the sheet in frame.",
            "Pull the bob aside by less than 15 degrees and let go without pushing.",
            "Record at least 5 full swings in slow motion.",
        ],
        "measure": "g from the period and the length (the length comes from the swing's arc, or enter it from a tape measure)",
        "limits": {"max_amplitude_deg": 15, "min_cycles": 4, "min_fps": 200},
    },
    "free_fall": {
        "you_need": ["small heavy ball (not a ping-pong ball)", "plain A4 sheet", "phone with 240 fps slow motion"],
        "steps": [
            "Tape the A4 sheet on the wall about halfway down the drop, in the plane the ball falls through.",
            "Frame the whole drop (about 1 m) with the phone square to the wall, 2 m away.",
            "Hold the ball still against the wall line, then release it without throwing.",
        ],
        "measure": "g from a parabola fitted to the fall",
        "limits": {"min_flight_frames": 40, "min_fps": 200},
    },
    "spring": {
        "you_need": ["spring", "hanging mass you can weigh", "plain A4 sheet", "phone with 240 fps slow motion"],
        "steps": [
            "Weigh the hanging mass.",
            "Tape the A4 sheet behind the spring in the plane of motion.",
            "Pull the mass down 2-5 cm and release; keep the bounce vertical.",
            "Record at least 5 oscillations.",
        ],
        "measure": "spring constant k = m ω₀² from the oscillation",
        "limits": {"min_cycles": 4, "min_fps": 200},
    },
}

THEORY = {
    "pendulum": {
        "model": "T₀ = 2π √(L/g) for small swings; damping and finite amplitude are corrected for",
        "assumptions": ["point mass bob", "massless, inextensible string", "small swing angle", "fixed pivot"],
        "error_sources": [
            "length: measured to the bob's centre, not its top",
            "large amplitude lengthens the period (about 0.4% at 15 degrees, corrected)",
            "A4 sheet not in the plane of motion scales every distance",
            "a wobbling pivot or an elliptical swing",
        ],
    },
    "free_fall": {
        "model": "y(t) = y₀ + v₀t − ½gt²",
        "assumptions": ["air drag negligible (dense ball, short drop)", "camera square to the plane of the fall"],
        "error_sources": [
            "A4 sheet far from the fall or tilted scales g directly",
            "light balls feel drag",
            "too few frames in flight",
            "motion blur near the floor",
        ],
    },
    "spring": {
        "model": "ω₀ = √(k/m), observed ω_d = √(ω₀² − γ²)",
        "assumptions": ["spring mass small compared with the hanging mass", "purely vertical motion", "linear spring (Hooke's law)"],
        "error_sources": [
            "spring mass adds about a third of its mass to the effective mass",
            "sideways swinging mixes in a pendulum motion",
            "an inaccurate mass value scales k directly",
        ],
    },
}


def plan_experiment(experiment: str) -> dict:
    if experiment not in PROTOCOLS:
        return {"error": f"unknown experiment {experiment!r}", "known": list(PROTOCOLS)}
    return {"experiment": experiment, **PROTOCOLS[experiment]}


def check_quality(run_id: str) -> dict:
    run = runs.load(run_id)
    checks = check_run(run)
    worst = "fail" if any(c["status"] == "fail" for c in checks) else "warn" if any(c["status"] == "warn" for c in checks) else "ok"
    return {"run_id": run_id, "overall": worst, "checks": checks}


def analyze_run(run_id: str) -> dict:
    run = runs.load(run_id)
    analysis = run["analysis"]
    if analysis is None:
        return {"run_id": run_id, "error": "run has not been analysed"}
    fit = analysis["fit"]
    fps = analysis["video"]["fps"]
    measurements = [
        {
            "ref": f"{run_id}.{key}",
            "name": m["name"],
            "value": _round_sig(m["value"], m["sigma"]),
            "sigma": _round_sig(m["sigma"], m["sigma"]),
            "unit": m["unit"],
        }
        for key, m in fit["measurements"].items()
    ]
    out = {
        "run_id": run_id,
        "experiment": analysis["experiment"],
        "frames_used": fit["window"],
        "seconds_used": [round(fit["window"][0] / fps, 3), round(fit["window"][1] / fps, 3)],
        "fit_residual_mm": round(1000 * fit["residual_rms_m"], 2),
        "measurements": measurements,
        "notes": fit["notes"],
    }
    key = "g" if "g" in fit["measurements"] else None
    if key:
        m = fit["measurements"][key]
        out["vs_standard_g"] = _gap(m["value"], m["sigma"], G_STANDARD, f"{run_id}.g_vs_standard")
    pred = run.get("prediction")
    if pred and pred.get("quantity") in fit["measurements"]:
        m = fit["measurements"][pred["quantity"]]
        gap = _gap(m["value"], m["sigma"], pred["value"], f"{run_id}.prediction_gap", reference_label="predicted")
        gap.pop("sigmas_away")
        out["prediction"] = {**pred, **gap, **_implied(analysis["experiment"], pred, fit["measurements"], run)}
    return out


def _implied(experiment: str, pred: dict, m: dict, run: dict) -> dict:
    """What the prediction would have required, so the explanation can name the physical cause."""
    if experiment == "pendulum" and pred["quantity"] == "period":
        L_needed = G_STANDARD * (pred["value"] / (2 * math.pi)) ** 2
        L = m["length"]["value"]
        T_expected = 2 * math.pi * math.sqrt(L / G_STANDARD)
        return {
            "length_needed_for_prediction_m": round(L_needed, 3),
            "measured_length_m": round(L, 4),
            "length_gap_percent": round(100 * (L_needed - L) / L, 1),
            "period_expected_from_measured_length_s": round(T_expected, 3),
            "hint": "period depends only on length and g (not mass, not amplitude for small swings)",
        }
    if experiment == "spring" and pred["quantity"] == "k" and run.get("mass_kg"):
        omega0 = m["omega0"]["value"]
        return {
            "period_for_predicted_k_s": round(2 * math.pi * math.sqrt(run["mass_kg"] / pred["value"]), 4),
            "measured_period_s": round(2 * math.pi / omega0, 4),
            "hint": "a stiffer spring oscillates faster; k = m ω₀²",
        }
    return {}


def compare_runs(run_ids: list[str]) -> dict:
    loaded = [runs.load(r) for r in run_ids]
    exps = {r["analysis"]["experiment"] for r in loaded if r["analysis"]}
    # g is comparable across pendulum and free fall; k only across springs.
    if "spring" in exps and len(exps) > 1:
        return {"error": "spring runs measure k and can only be compared with other spring runs", "experiments": sorted(exps)}
    exp = exps.pop() if len(exps) == 1 else "mixed_g"
    key = "k" if exp == "spring" else "g"
    rows = []
    for r in loaded:
        m = r["analysis"]["fit"]["measurements"]
        if key in m:
            row = {"run_id": r["run_id"], "experiment": r["analysis"]["experiment"], "ref": f"{r['run_id']}.{key}",
                   "value": m[key]["value"], "sigma": m[key]["sigma"]}
            if exp == "pendulum":
                row["length_m"] = m["length"]["value"]
                row["period_s"] = m["period"]["value"]
            rows.append(row)
    if len(rows) < 2:
        return {"error": f"need at least two analysed runs with {key}"}

    v = np.array([r["value"] for r in rows])
    s = np.array([r["sigma"] for r in rows])
    w = 1 / s**2
    mean = float(np.sum(w * v) / np.sum(w))
    mean_sigma = float(1 / np.sqrt(np.sum(w)))
    chi2 = float(np.sum(((v - mean) / s) ** 2))
    out = {
        "experiment": exp,
        "quantity": key,
        "runs": [{**r, "value": _round_sig(r["value"], r["sigma"]), "sigma": _round_sig(r["sigma"], r["sigma"])} for r in rows],
        "combined": {"ref": "compare." + key, "value": _round_sig(mean, mean_sigma), "sigma": _round_sig(mean_sigma, mean_sigma)},
        "consistency": {
            "chi2_per_dof": round(chi2 / (len(rows) - 1), 2),
            "verdict": "runs agree within their uncertainties" if chi2 / (len(rows) - 1) < 3 else "runs disagree more than their uncertainties allow; look for a changed setup",
        },
        "spread_percent": round(100 * float(np.ptp(v)) / mean, 2),
    }
    if exp == "pendulum" and len({round(r["length_m"], 2) for r in rows}) > 1:
        # T² = (4π²/g) L: the slope across different lengths gives g independently of any single length error offset.
        L = np.array([r["length_m"] for r in rows])
        T2 = np.array([r["period_s"] for r in rows]) ** 2
        slope, intercept = np.polyfit(L, T2, 1)
        out["t_squared_vs_length"] = {"ref": "compare.g_slope", "g_from_slope": round(4 * math.pi**2 / slope, 3), "intercept_s2": round(float(intercept), 4)}
    return out


def explain_physics(experiment: str) -> dict:
    if experiment not in THEORY:
        return {"error": f"unknown experiment {experiment!r}"}
    return {"experiment": experiment, **THEORY[experiment]}


def _gap(value: float, sigma: float, reference: float, ref: str, reference_label: str = "reference") -> dict:
    diff = value - reference
    return {
        "ref": ref,
        reference_label: reference,
        "difference": _round_sig(diff, sigma),
        "difference_percent": round(100 * diff / reference, 2),
        "sigmas_away": round(abs(diff) / sigma, 1) if sigma else None,
    }


def _round_sig(x: float, sigma: float) -> float:
    """Round to the uncertainty's second significant figure, so displayed digits are meaningful."""
    if not sigma or not math.isfinite(sigma):
        return round(x, 4)
    digits = max(0, 1 - int(math.floor(math.log10(abs(sigma)))))
    return round(x, digits)


TOOL_FUNCS = {
    "plan_experiment": plan_experiment,
    "check_quality": check_quality,
    "analyze_run": analyze_run,
    "compare_runs": compare_runs,
    "explain_physics": explain_physics,
}

_EXP = {"type": "string", "enum": ["pendulum", "free_fall", "spring"]}
_RUN = {"type": "string", "description": "run id such as r1"}

TOOL_SCHEMAS = [
    {"type": "function", "name": "plan_experiment",
     "description": "Setup checklist and limits for an experiment, to plan a recording before filming.",
     "parameters": {"type": "object", "properties": {"experiment": _EXP}, "required": ["experiment"], "additionalProperties": False}},
    {"type": "function", "name": "check_quality",
     "description": "Deterministic quality checks for a recorded run: frame rate, tracking coverage and jumps, A4 calibration, fit residual, swing amplitude, cycles.",
     "parameters": {"type": "object", "properties": {"run_id": _RUN}, "required": ["run_id"], "additionalProperties": False}},
    {"type": "function", "name": "analyze_run",
     "description": "Measured constants for a run with uncertainties and refs, the comparison with standard g, and the student's prediction gap if they made one.",
     "parameters": {"type": "object", "properties": {"run_id": _RUN}, "required": ["run_id"], "additionalProperties": False}},
    {"type": "function", "name": "compare_runs",
     "description": "Combine and compare runs: g across pendulum and free-fall runs, or k across spring runs. Weighted mean, consistency, and for pendulums of different lengths a T² vs L fit.",
     "parameters": {"type": "object", "properties": {"run_ids": {"type": "array", "items": _RUN, "minItems": 2}}, "required": ["run_ids"], "additionalProperties": False}},
    {"type": "function", "name": "explain_physics",
     "description": "The model equation, assumptions and known error sources for an experiment, to ground explanations.",
     "parameters": {"type": "object", "properties": {"experiment": _EXP}, "required": ["experiment"], "additionalProperties": False}},
]
