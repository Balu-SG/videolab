"""A4 sheet detection and the pixel -> metre homography.

The sheet must lie in the plane of motion (taped to the wall behind a pendulum, for example).
Output frame: origin at the sheet's bottom-left corner, x right, y up, metres.
"""

from dataclasses import dataclass

import cv2
import numpy as np

A4_SHORT_M = 0.210
A4_LONG_M = 0.297
CORNER_SIGMA_PX = 1.0


@dataclass
class SheetFit:
    corners_px: np.ndarray  # 4x2 image px: bottom-left, bottom-right, top-right, top-left
    portrait: bool
    image_to_m: np.ndarray  # 3x3 homography
    reprojection_px: float
    area_frac: float  # sheet area / frame area, for quality checks

    @property
    def size_m(self) -> tuple[float, float]:
        return (A4_SHORT_M, A4_LONG_M) if self.portrait else (A4_LONG_M, A4_SHORT_M)

    @property
    def scale_sigma(self) -> float:
        """Relative 1-sigma scale uncertainty, assuming each corner is good to about 1 px."""
        bl, br, tr, tl = self.corners_px
        side_px = (np.linalg.norm(br - bl) + np.linalg.norm(tr - tl) + np.linalg.norm(tl - bl) + np.linalg.norm(tr - br)) / 4
        return float(np.sqrt(2) * CORNER_SIGMA_PX / side_px)

    def to_dict(self) -> dict:
        return {
            "corners_px": self.corners_px.round(2).tolist(),
            "scale_sigma": round(self.scale_sigma, 5),
            "portrait": self.portrait,
            "image_to_m": self.image_to_m.tolist(),
            "reprojection_px": round(self.reprojection_px, 3),
            "area_frac": round(self.area_frac, 4),
        }


def read_frame(video_path: str, index: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"could not read frame {index} of {video_path}")
    return frame


def find_sheet_corners(frame: np.ndarray) -> np.ndarray | None:
    """Largest bright, convex quadrilateral with a plausible A4 shape, corners refined to sub-pixel.

    A single global threshold (Otsu) merges a pale wall with the paper, so sweep thresholds from bright to dark
    and keep the biggest clean quad that doesn't touch the frame edge.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape
    frame_area = h * w
    kernel = np.ones((5, 5), np.uint8)

    best, best_area = None, 0.0
    for level in range(int(np.percentile(blur, 99.5)), 80, -8):
        mask = cv2.morphologyEx((blur >= level).astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.002 * frame_area or area > 0.6 * frame_area or area <= best_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
                continue
            quad = _as_a4_quad(c, area)
            if quad is not None:
                best, best_area = quad, area
    if best is None:
        return None

    # A sweep level near the wall's brightness lets speckle bleed into a corner. Re-cut at the midpoint
    # between paper and surround, where a step edge sits exactly on the true boundary.
    level = _midpoint_level(blur, best)
    mask = cv2.morphologyEx((blur >= level).astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centre = tuple(float(v) for v in best.mean(axis=0))
    for c in contours:
        if cv2.pointPolygonTest(c, centre, False) > 0:
            quad = _as_a4_quad(c, cv2.contourArea(c))
            if quad is not None:
                best = quad
            break

    corners = order_corners(best.astype(np.float64))
    refined = refine_corners_by_edges(blur.astype(np.float32), corners)
    return refined if refined is not None else corners


def _midpoint_level(gray: np.ndarray, quad: np.ndarray) -> float:
    inside = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(inside, quad.astype(np.int32), 1)
    grown = cv2.dilate(inside, np.ones((25, 25), np.uint8))
    shrunk = cv2.erode(inside, np.ones((9, 9), np.uint8))
    ring = (grown > 0) & ~(cv2.dilate(inside, np.ones((7, 7), np.uint8)) > 0)
    return (float(np.median(gray[shrunk > 0])) + float(np.median(gray[ring]))) / 2


def _as_a4_quad(contour: np.ndarray, area: float) -> np.ndarray | None:
    # The tightest tolerance that still gives four vertices keeps the corners close to the real ones.
    perimeter = cv2.arcLength(contour, True)
    for eps in (0.004, 0.007, 0.01, 0.015, 0.02, 0.03):
        quad = cv2.approxPolyDP(contour, eps * perimeter, True)
        if len(quad) <= 4:
            break
    if len(quad) != 4 or not cv2.isContourConvex(quad):
        return None
    # Fill ratio rejects blobs that only roughly simplify to four points.
    if area / cv2.contourArea(quad) < 0.93:
        return None
    sides = np.linalg.norm(np.roll(quad[:, 0], -1, axis=0) - quad[:, 0], axis=1)
    pair_a, pair_b = sides[0] + sides[2], sides[1] + sides[3]
    if not 1.1 < max(pair_a, pair_b) / min(pair_a, pair_b) < 1.8:  # A4 is 1.414; allow perspective
        return None
    return quad[:, 0].astype(np.float32)


def refine_corners_by_edges(gray: np.ndarray, corners: np.ndarray, samples: int = 60, reach: float = 8.0) -> np.ndarray | None:
    """Sub-pixel corners: locate each edge precisely along perpendicular intensity profiles, fit a line, intersect.

    cornerSubPix is built for chessboard saddles and drifts on a plain L-shaped paper corner; line fits average
    over hundreds of edge samples instead.
    """
    lines = []
    offsets = np.linspace(-reach, reach, int(4 * reach) + 1)
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-d[1], d[0]])
        edge_pts = []
        for f in np.linspace(0.12, 0.88, samples):  # stay clear of the corners
            base = a + f * (b - a)
            xs = (base[0] + offsets * normal[0]).astype(np.float32).reshape(1, -1)
            ys = (base[1] + offsets * normal[1]).astype(np.float32).reshape(1, -1)
            prof = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR)[0]
            lo, hi = np.percentile(prof, 10), np.percentile(prof, 90)
            if hi - lo < 15:
                continue
            mid = (lo + hi) / 2
            above = prof > mid
            cross = np.nonzero(above[1:] != above[:-1])[0]
            if len(cross) != 1:  # noise, an occluding object, or a second edge in the window
                continue
            j = cross[0]
            t = offsets[j] + (mid - prof[j]) / (prof[j + 1] - prof[j]) * (offsets[j + 1] - offsets[j])
            edge_pts.append(base + t * normal)
        if len(edge_pts) < samples // 3:
            return None
        vx, vy, x0, y0 = cv2.fitLine(np.array(edge_pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))

    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[(i - 1) % 4], lines[i]  # corner i joins edge i-1 and edge i
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        s_, _ = np.linalg.solve(A, p2 - p1)
        out.append(p1 + s_ * d1)
    out = np.array(out)
    # A refinement that moves a corner far means the edge fit latched onto something else.
    return out if np.abs(out - corners).max() < 6 else None


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Bottom-left, bottom-right, top-right, top-left in image coordinates (y down). Assumes the phone is roughly upright."""
    by_y = pts[np.argsort(pts[:, 1])]
    top, bottom = by_y[:2], by_y[2:]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([bl, br, tr, tl], dtype=np.float64)


def fit_sheet(corners_px: np.ndarray, frame_shape: tuple[int, int]) -> SheetFit:
    bl, br, tr, tl = corners_px
    width_px = (np.linalg.norm(br - bl) + np.linalg.norm(tr - tl)) / 2
    height_px = (np.linalg.norm(tl - bl) + np.linalg.norm(tr - br)) / 2
    portrait = bool(height_px > width_px)
    w, h = (A4_SHORT_M, A4_LONG_M) if portrait else (A4_LONG_M, A4_SHORT_M)
    world = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    H = cv2.getPerspectiveTransform(corners_px.astype(np.float32), world.astype(np.float32)).astype(np.float64)

    back = cv2.perspectiveTransform(world.reshape(-1, 1, 2), np.linalg.inv(H))[:, 0]
    reproj = float(np.abs(back - corners_px).max())
    area = cv2.contourArea(corners_px.astype(np.float32)) / (frame_shape[0] * frame_shape[1])
    return SheetFit(corners_px, portrait, H, reproj, float(area))


def detect_sheet(video_path: str, frames: tuple[int, ...] = (0,)) -> SheetFit | None:
    """Detect the sheet on a few frames and take the per-corner median, so a ball passing in front of one frame doesn't matter."""
    found, shape = [], None
    for i in frames:
        frame = read_frame(video_path, i)
        shape = frame.shape[:2]
        c = find_sheet_corners(frame)
        if c is not None:
            found.append(c)
    if not found:
        return None
    return fit_sheet(np.median(np.stack(found), axis=0), shape)


def to_metres(H: np.ndarray, xy_px: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(xy_px, dtype=np.float64).reshape(-1, 1, 2), H)[:, 0]
