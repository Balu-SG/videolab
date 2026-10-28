"""Synthetic 240 fps experiment clips with exact ground truth, for the accuracy benchmark.

The scene is a wall plane in metres (x right, y up) seen through a camera homography, with an A4 sheet
taped to the wall. Physics is integrated exactly (RK4 for the large-angle pendulum), so the true values
are known. Conditions vary the camera angle, distance, lighting, sensor noise and motion blur.
"""

import math
import subprocess
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, G = 1920, 1080, 240, 9.81
WORLD = np.float32([[-0.75, -0.10], [0.75, -0.10], [0.75, 1.15], [-0.75, 1.15]])
CAMERAS = {  # where the wall's corners land in the image
    "front": np.float32([[300, 1040], [1660, 1010], [1600, 40], [340, 70]]),
    "tilted": np.float32([[230, 1060], [1720, 980], [1540, 60], [400, 15]]),
    "far": np.float32([[560, 900], [1380, 885], [1350, 250], [585, 262]]),
}
A4 = [(0.30, 0.05), (0.51, 0.05), (0.51, 0.347), (0.30, 0.347)]  # 210 x 297 mm, portrait


@dataclass
class Spec:
    name: str
    experiment: str  # pendulum | free_fall | spring
    params: dict
    camera: str = "front"
    light: float = 1.0  # exposure gain; 0.45 is a dim room
    noise: float = 2.5  # sensor noise sigma, grey levels
    blur: int = 1  # sub-frames averaged per frame (shutter open half the frame time)
    tags: list[str] = field(default_factory=list)


def _scene(spec: Spec):
    hm = cv2.getPerspectiveTransform(WORLD, CAMERAS[spec.camera])
    rng = np.random.default_rng(zlib.crc32(spec.name.encode()))

    def to_px(pts):
        return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), hm).reshape(-1, 2)

    bg = np.full((H, W, 3), (200, 205, 210), np.float32)
    bg += cv2.resize(rng.normal(0, 4, (H // 8, W // 8)).astype(np.float32), (W, H))[..., None]
    bg = bg.astype(np.uint8)
    cv2.fillPoly(bg, [to_px([(-0.75, -0.10), (0.75, -0.10), (0.75, 0.0), (-0.75, 0.0)]).astype(np.int32)], (120, 130, 140), cv2.LINE_AA)
    cv2.fillPoly(bg, [to_px(A4).astype(np.int32)], (250, 250, 248), cv2.LINE_AA)
    noise = [rng.normal(0, spec.noise, (H, W, 1)).astype(np.float32) for _ in range(8)]
    return hm, to_px, bg, noise


def render(spec: Spec, out_dir: Path) -> tuple[Path, dict]:
    """Writes <name>.mp4 and returns (path, truth)."""
    hm, to_px, bg, noise = _scene(spec)
    p = spec.params

    def disc(img, c, r, color):
        a = np.linspace(0, 2 * math.pi, 64, endpoint=False)
        poly = to_px(np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)])
        cv2.fillPoly(img, [np.round(poly * 16).astype(np.int32)], color, cv2.LINE_AA, shift=4)

    def line(img, a, b, color, th):
        a, b = to_px([a, b])
        cv2.line(img, tuple(np.round(a * 16).astype(int)), tuple(np.round(b * 16).astype(int)), color, th, cv2.LINE_AA, shift=4)

    if spec.experiment == "pendulum":
        L, th0, damp, pivot = p["length_m"], math.radians(p["amplitude_deg"]), 0.03, (-0.15, 1.05)
        secs = 6.0

        def accel(y):
            return np.array([y[1], -(G / L) * math.sin(y[0]) - damp * y[1]])

        dt, y, th = 1 / (FPS * 8), np.array([th0, 0.0]), []
        for _ in range(int(secs * FPS * 8)):  # RK4 at 8x the frame rate, sampled for blur sub-frames
            th.append(y[0])
            k1 = accel(y); k2 = accel(y + dt / 2 * k1); k3 = accel(y + dt / 2 * k2); k4 = accel(y + dt * k3)
            y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        def draw(img, t):
            a = th[min(int(round(t * FPS * 8)), len(th) - 1)]
            bob = (pivot[0] + L * math.sin(a), pivot[1] - L * math.cos(a))
            line(img, (pivot[0] - 0.08, pivot[1] + 0.01), (pivot[0] + 0.08, pivot[1] + 0.01), (60, 60, 60), 8)
            line(img, pivot, bob, (40, 40, 40), 2)
            disc(img, bob, 0.022, (30, 120, 240))
            return bob

        truth = {"g": G, "length": L, "concept": "orange ball"}
    elif spec.experiment == "free_fall":
        r, y0, hold = 0.025, p["drop_m"], 0.2
        secs = hold + math.sqrt(2 * (y0 - r) / G) + 0.3

        def draw(img, t):
            tf = max(0.0, t - hold)
            c = (0.0, max(r, y0 - 0.5 * G * tf * tf))
            disc(img, c, r, (40, 40, 210))
            return c

        truth = {"g": G, "concept": "red ball"}
    else:
        m, k, gamma, amp, anchor = p["mass_kg"], p["k"], 0.25, 0.06, (-0.35, 1.05)
        wd = math.sqrt(k / m - gamma**2)
        yeq, secs = anchor[1] - 0.45, 4.0

        def draw(img, t):
            yy = yeq + amp * math.exp(-gamma * t) * math.cos(wd * t)
            line(img, (anchor[0] - 0.08, anchor[1]), (anchor[0] + 0.08, anchor[1]), (60, 60, 60), 8)
            top, bot, n = anchor[1], yy + 0.03, 14
            pts = [(anchor[0], top)] + [(anchor[0] + (0.025 if j % 2 else -0.025), top - (top - bot) * (j + 0.5) / n) for j in range(n)] + [(anchor[0], bot)]
            for a, b in zip(pts, pts[1:]):
                line(img, a, b, (70, 70, 70), 3)
            blk = to_px([(anchor[0] - 0.03, yy + 0.03), (anchor[0] + 0.03, yy + 0.03), (anchor[0] + 0.03, yy - 0.03), (anchor[0] - 0.03, yy - 0.03)])
            cv2.fillPoly(img, [np.round(blk * 16).astype(np.int32)], (60, 160, 60), cv2.LINE_AA, shift=4)
            return (anchor[0], yy)

        truth = {"k": k, "mass_kg": m, "concept": "green block"}

    start = draw(bg.copy(), 0.0)  # where a user would tap on the first frame
    truth["start_px"] = to_px([start])[0].round(1).tolist()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.name}.mp4"
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE,
    )
    for i in range(int(secs * FPS)):
        acc = np.zeros((H, W, 3), np.float32)
        for s in range(spec.blur):  # the shutter is open for the first half of the frame
            img = bg.copy()
            draw(img, (i + 0.5 * s / spec.blur) / FPS)
            acc += img
        frame = acc / spec.blur * spec.light + noise[i % 8]
        enc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()
    return path, {**truth, "a4_image_px": to_px(A4).round(2).tolist()}
