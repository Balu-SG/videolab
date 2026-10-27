"""Pre-flight check on one camera frame, before recording: sheet, lighting, camera angle, sharpness, clutter."""

import cv2
import numpy as np

from .geometry import find_sheet_corners


def _check(name: str, status: str, value, message: str) -> dict:
    return {"check": name, "status": status, "value": value, "message": message}


def check_frame(frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    checks = []

    mean = float(gray.mean())
    if mean < 60:
        checks.append(_check("lighting", "fail", round(mean), "too dark; add light or move near a window"))
    elif mean > 225:
        checks.append(_check("lighting", "warn", round(mean), "very bright; avoid pointing at a window or lamp"))
    else:
        checks.append(_check("lighting", "ok", round(mean), "lighting is fine"))

    corners = find_sheet_corners(frame)
    if corners is None:
        checks.append(_check("a4_sheet", "fail", None, "no A4 sheet found; tape a plain white A4 sheet flat behind the motion, fully in view"))
    else:
        bl, br, tr, tl = corners
        area = cv2.contourArea(corners.astype(np.float32)) / (h * w)
        checks.append(_check("a4_sheet", "ok" if area >= 0.015 else "warn", round(area, 4),
                             f"A4 sheet found ({100 * area:.1f}% of the view)" + ("" if area >= 0.015 else "; move closer so it is bigger")))
        # Opposite sides of the sheet differ in length when the phone isn't square to the wall.
        lr = np.linalg.norm(tl - bl) / np.linalg.norm(tr - br)
        tb = np.linalg.norm(tr - tl) / np.linalg.norm(br - bl)
        skew = max(abs(np.log(lr)), abs(np.log(tb)))
        width = _edge_width(gray.astype(np.float32), corners)
        if width is not None:
            checks.append(_check("sharpness", "ok" if width <= 3.5 else "warn", round(width, 2),
                                 "image is sharp" if width <= 3.5 else "image looks blurry; tap to focus and hold the phone still"))
        checks.append(_check("camera_square", "ok" if skew < 0.05 else "warn", round(float(np.exp(skew) - 1), 3),
                             "phone is square to the wall" if skew < 0.05 else "turn the phone so it faces the wall straight on"))

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
    clutter = float((edges > 0).mean())
    checks.append(_check("background", "ok" if clutter < 0.06 else "warn", round(clutter, 3),
                         "background is plain" if clutter < 0.06 else "busy background; a plain wall helps tracking"))

    worst = "fail" if any(c["status"] == "fail" for c in checks) else "warn" if any(c["status"] == "warn" for c in checks) else "ok"
    return {"overall": worst, "checks": checks, "sheet_corners_px": None if corners is None else corners.round(1).tolist(), "size": [w, h]}


def _edge_width(gray: np.ndarray, corners: np.ndarray) -> float | None:
    """Median blur width of the sheet's edges in pixels: contrast divided by the steepest gradient across the edge.

    A plain wall has no texture, so whole-image sharpness scores call it blurry; the sheet edge is a known step.
    """
    widths = []
    offsets = np.linspace(-10, 10, 81)
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-d[1], d[0]])
        for f in np.linspace(0.2, 0.8, 9):
            base = a + f * (b - a)
            xs = (base[0] + offsets * normal[0]).astype(np.float32).reshape(1, -1)
            ys = (base[1] + offsets * normal[1]).astype(np.float32).reshape(1, -1)
            prof = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR)[0]
            contrast = np.percentile(prof, 90) - np.percentile(prof, 10)
            slope = np.abs(np.gradient(prof, offsets)).max()
            if contrast > 20 and slope > 0:
                widths.append(contrast / slope)
    return float(np.median(widths)) if widths else None
