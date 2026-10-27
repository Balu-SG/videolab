"""Deterministic physics fits. Every number VideoLab reports comes from here, never from the LLM.

Each fit returns the measured constants with 1-sigma uncertainties, the frame window it used, and the
fitted model sampled on every frame (in metres) so the UI can draw the ideal motion over the video.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit, least_squares

from .trajectory import Trajectory


@dataclass
class Measurement:
    name: str
    value: float
    sigma: float
    unit: str

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "sigma": self.sigma, "unit": self.unit}


@dataclass
class FitResult:
    experiment: str
    measurements: dict[str, Measurement]
    window: tuple[int, int]  # first and last source frame used
    residual_rms_m: float
    model_frames: np.ndarray  # frames the model is drawn on
    model_xy_m: np.ndarray  # n x 2, fitted ideal position per model frame
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "experiment": self.experiment,
            "measurements": {k: m.to_dict() for k, m in self.measurements.items()},
            "window": list(self.window),
            "residual_rms_m": self.residual_rms_m,
            "model": {"frames": self.model_frames.tolist(), "xy_m": self.model_xy_m.round(6).tolist()},
            "notes": self.notes,
        }


PERIOD_REL_FLOOR = 5e-4  # frame-clock accuracy and model limits (amplitude-dependent period); stat errors are smaller


def _damped_cos(t, amp, gamma, omega, phase, offset):
    return amp * np.exp(-gamma * t) * np.cos(omega * t + phase) + offset


def _fit_damped(t: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit s(t) = A e^(-γt) cos(ωt + φ) + c, seeded from the FFT peak. Returns params and 1-sigma errors."""
    t0 = t - t[0]
    centred = s - s.mean()
    dt = np.median(np.diff(t))
    n = 1 << int(np.ceil(np.log2(len(s) * 8)))
    spectrum = np.abs(np.fft.rfft(centred * np.hanning(len(s)), n))
    freqs = np.fft.rfftfreq(n, dt)
    spectrum[freqs < 1.5 / (t0[-1] + dt)] = 0  # need at least ~1.5 cycles
    omega0 = 2 * np.pi * freqs[spectrum.argmax()]
    amp0 = np.ptp(centred) / 2
    best = None
    for phase_guess in np.linspace(-np.pi, np.pi, 8, endpoint=False):
        try:
            p, cov = curve_fit(_damped_cos, t0, s, p0=[amp0, 0.05, omega0, phase_guess, s.mean()], maxfev=20000)
        except RuntimeError:
            continue
        cost = np.sum((_damped_cos(t0, *p) - s) ** 2)
        if best is None or cost < best[0]:
            best = (cost, p, cov)
    if best is None:
        raise ValueError("damped oscillation fit did not converge")
    _, p, cov = best
    if p[0] < 0:  # canonical form: positive amplitude
        p[0], p[3] = -p[0], p[3] + np.pi
    return p, np.sqrt(np.diag(cov))


def _circle_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Geometric circle fit (centre and radius), seeded by the algebraic Kåsa fit. Returns cx, cy, r, sigma_r."""
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = c[0] / 2, c[1] / 2
    r = np.sqrt(c[2] + cx**2 + cy**2)
    res = least_squares(lambda p: np.hypot(x - p[0], y - p[1]) - p[2], [cx, cy, r])
    J = res.jac
    dof = max(1, len(x) - 3)
    s2 = 2 * res.cost / dof
    cov = np.linalg.pinv(J.T @ J) * s2
    return float(res.x[0]), float(res.x[1]), float(res.x[2]), float(np.sqrt(cov[2, 2]))


def fit_pendulum(traj: Trajectory, length_m: float | None = None) -> FitResult:
    """g from the period and length. The pivot comes from a circle fit to the bob's path unless the length is given."""
    notes = []
    cx, cy, r, r_sigma = _circle_fit(traj.x, traj.y)
    if cy < traj.y.max():
        notes.append("circle fit put the pivot below the bob; the path may be too flat or the camera tilted")
    if length_m is None:
        length, length_sigma = r, float(np.hypot(r_sigma, r * traj.scale_sigma))
        notes.append(f"length from the path's circle fit: {r:.4f} m")
    else:
        length, length_sigma = length_m, 0.002  # tape measure, ±2 mm
        notes.append(f"length entered by the user: {length_m:.4f} m (circle fit says {r:.4f} m)")

    theta = np.arctan2(traj.x - cx, cy - traj.y)
    p, perr = _fit_damped(traj.t, theta)
    amp, gamma, omega_d, phase, offset = p

    # Damping shifts the frequency; finite amplitude lengthens the period (series for the large-angle period).
    omega_0 = np.sqrt(omega_d**2 + gamma**2)
    t0 = traj.t - traj.t[0]
    mean_amp = float(np.mean(amp * np.exp(-gamma * t0)))
    amp_factor = 1 + mean_amp**2 / 16 + 11 * mean_amp**4 / 3072
    omega_small = omega_0 * amp_factor
    period = 2 * np.pi / omega_d
    omega_rel = float(np.hypot(perr[2] / omega_d, PERIOD_REL_FLOOR))
    period_sigma = period * omega_rel
    g = omega_small**2 * length
    g_sigma = g * np.sqrt((2 * omega_rel) ** 2 + (length_sigma / length) ** 2)

    model_theta = _damped_cos(t0, *p)
    model_xy = np.column_stack([cx + r * np.sin(model_theta), cy - r * np.cos(model_theta)])
    residual = np.hypot(model_xy[:, 0] - traj.x, model_xy[:, 1] - traj.y)

    return FitResult(
        "pendulum",
        {
            "g": Measurement("gravitational acceleration", float(g), float(g_sigma), "m/s²"),
            "period": Measurement("period (as measured)", float(period), float(period_sigma), "s"),
            "length": Measurement("pendulum length", float(length), float(length_sigma), "m"),
            "amplitude": Measurement("mean swing amplitude", float(np.degrees(mean_amp)), float(np.degrees(perr[0])), "°"),
            "damping": Measurement("damping rate", float(gamma), float(perr[1]), "1/s"),
        },
        (int(traj.frames[0]), int(traj.frames[-1])),
        float(np.sqrt(np.mean(residual**2))),
        traj.frames,
        model_xy,
        notes,
    )


def _flight_window(t: np.ndarray, y: np.ndarray) -> tuple[int, int]:
    """Indices bracketing free flight: from the release (motion starts) to just before the lowest point (impact)."""
    rest = np.median(y[: max(5, len(y) // 20)])
    noise = max(1e-4, 3 * np.std(y[: max(5, len(y) // 20)]))
    moving = np.nonzero(rest - y > max(0.002, 5 * noise))[0]
    if len(moving) == 0:
        raise ValueError("the object never moved")
    start = moving[0]
    # Impact is the first frame within 1.5 mm of the lowest point. A plain argmin can land on a later frame
    # where the ball sits on the floor and tracking jitter dips a fraction of a millimetre lower.
    floor = y[start:].min()
    low = start + int(np.nonzero(y[start:] <= floor + 0.0015)[0][0])
    # Drop the frames right before impact (deformation, motion blur) and right after release (hand contact).
    margin = max(2, int(0.02 / np.median(np.diff(t))))
    return start + 1, max(start + 6, low - margin)


def fit_free_fall(traj: Trajectory) -> FitResult:
    i0, i1 = _flight_window(traj.t, traj.y)
    t, y, x = traj.t[i0:i1 + 1], traj.y[i0:i1 + 1], traj.x[i0:i1 + 1]
    if len(t) < 10:
        raise ValueError(f"only {len(t)} frames of free flight; need a higher frame rate or a longer drop")
    # Trim tail frames that leave the parabola (bounce, floor contact) until the end of the fit is clean.
    while True:
        tc = t - t.mean()
        coeffs, cov = np.polyfit(tc, y, 2, cov=True)
        res = y - np.polyval(coeffs, tc)
        spread = max(1e-4, 1.4826 * np.median(np.abs(res)))
        if len(t) <= 12 or np.abs(res[-1]) < 4 * spread:
            break
        t, y, x, i1 = t[:-1], y[:-1], x[:-1], i1 - 1
    g = -2 * coeffs[0]
    g_sigma = float(np.hypot(2 * np.sqrt(cov[0, 0]), g * traj.scale_sigma))
    model_y = np.polyval(coeffs, tc)
    model_x = np.polyval(np.polyfit(tc, x, 1), tc)
    residual = np.hypot(model_y - y, model_x - x)
    notes = []
    drift = np.ptp(model_x)
    if drift > 0.02:
        notes.append(f"the ball drifted {1000 * drift:.0f} mm sideways; it may have been thrown, not dropped")
    return FitResult(
        "free_fall",
        {
            "g": Measurement("gravitational acceleration", float(g), float(g_sigma), "m/s²"),
            "drop": Measurement("distance fallen in the fit window", float(np.ptp(model_y)), float(np.ptp(model_y) * traj.scale_sigma), "m"),
            "flight_time": Measurement("time in the fit window", float(t[-1] - t[0]), 1 / 240, "s"),
        },
        (int(traj.frames[i0]), int(traj.frames[i1])),
        float(np.sqrt(np.mean(residual**2))),
        traj.frames[i0:i1 + 1],
        np.column_stack([model_x, model_y]),
        notes,
    )


def fit_spring(traj: Trajectory, mass_kg: float | None = None) -> FitResult:
    """Spring constant from the vertical oscillation: k = m (ω_d² + γ²). Without a mass, report ω₀ only."""
    p, perr = _fit_damped(traj.t, traj.y)
    amp, gamma, omega_d, phase, offset = p
    omega_0 = np.sqrt(omega_d**2 + gamma**2)
    omega_0_sigma = float(np.hypot(np.sqrt((omega_d * perr[2]) ** 2 + (gamma * perr[1]) ** 2) / omega_0, omega_0 * PERIOD_REL_FLOOR))
    t0 = traj.t - traj.t[0]
    model_y = _damped_cos(t0, *p)
    model_x = np.full_like(model_y, np.median(traj.x))
    residual = np.hypot(model_y - traj.y, model_x - traj.x)

    m = {
        "period": Measurement("period (as measured)", float(2 * np.pi / omega_d), float(2 * np.pi / omega_d * np.hypot(perr[2] / omega_d, PERIOD_REL_FLOOR)), "s"),
        "omega0": Measurement("natural angular frequency", float(omega_0), float(omega_0_sigma), "rad/s"),
        "amplitude": Measurement("initial amplitude", float(amp), float(np.hypot(perr[0], amp * traj.scale_sigma)), "m"),
        "damping": Measurement("damping rate", float(gamma), float(perr[1]), "1/s"),
    }
    notes = []
    if mass_kg is not None:
        k = mass_kg * omega_0**2
        m["k"] = Measurement("spring constant", float(k), float(2 * k * omega_0_sigma / omega_0), "N/m")
    else:
        notes.append("enter the hanging mass to get the spring constant")
    return FitResult(
        "spring",
        m,
        (int(traj.frames[0]), int(traj.frames[-1])),
        float(np.sqrt(np.mean(residual**2))),
        traj.frames,
        np.column_stack([model_x, model_y]),
        notes,
    )
