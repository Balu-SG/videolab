"""Turn a SAM track into a time series in metres."""

from dataclasses import dataclass

import numpy as np

from .geometry import to_metres


@dataclass
class Trajectory:
    t: np.ndarray  # seconds, frame / fps
    frames: np.ndarray
    x: np.ndarray  # metres in the sheet frame (x right, y up)
    y: np.ndarray
    confidence: np.ndarray
    scale_sigma: float = 0.0  # relative calibration uncertainty from the sheet fit
    m_per_px: float = 0.0  # local image scale where the object moves

    def to_dict(self) -> dict:
        return {
            "t": self.t.round(6).tolist(),
            "frames": self.frames.tolist(),
            "x_m": self.x.round(6).tolist(),
            "y_m": self.y.round(6).tolist(),
            "confidence": self.confidence.round(4).tolist(),
        }


def from_track(track: dict, image_to_m: np.ndarray, object_id: str | None = None, scale_sigma: float = 0.0) -> Trajectory:
    points = track["points"]
    if object_id is None:
        ids = [p["object_id"] for p in points]
        object_id = max(set(ids), key=ids.count)
    pts = sorted((p for p in points if p["object_id"] == object_id), key=lambda p: p["frame"])

    frames = np.array([p["frame"] for p in pts])
    xy_px = np.array([[p["cx"], p["cy"]] for p in pts])
    xy_m = to_metres(image_to_m, xy_px)
    conf = np.array([p["confidence"] if p["confidence"] is not None else np.nan for p in pts])
    c = xy_px.mean(axis=0)
    probe = to_metres(image_to_m, np.array([c, c + [1, 0], c + [0, 1]]))
    m_per_px = float((np.linalg.norm(probe[1] - probe[0]) + np.linalg.norm(probe[2] - probe[0])) / 2)
    return Trajectory(frames / track["video"]["fps"], frames, xy_m[:, 0], xy_m[:, 1], conf, scale_sigma, m_per_px)
