"""Check A4 detection and pixel -> metre conversion against a synthetic clip's ground truth.

Optionally pass a SAM track to see the end-to-end position error in millimetres.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.geometry import detect_sheet, to_metres  # noqa: E402
from backend.trajectory import from_track  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("truth", type=Path)
    ap.add_argument("--track", type=Path)
    args = ap.parse_args()

    truth = json.loads(args.truth.read_text())
    n = len(truth["track_px"])
    sheet = detect_sheet(str(args.video), frames=(0, n // 3, 2 * n // 3))
    if sheet is None:
        sys.exit("A4 sheet not found")

    corner_err = np.linalg.norm(sheet.corners_px - np.array(truth["a4_image_px"]), axis=1)
    print(f"sheet: {'portrait' if sheet.portrait else 'landscape'}, {100 * sheet.area_frac:.2f}% of frame")
    print(f"corner error px: {' '.join(f'{e:.2f}' for e in corner_err)} (max {corner_err.max():.2f})")

    # True world position of every pixel, shifted into the sheet frame (origin at the sheet's bottom-left corner).
    world_to_img = np.array(truth["world_to_image_homography"])
    origin = np.array(truth["a4_world_m"][0])
    gt_px = np.array(truth["track_px"])
    gt_m = to_metres(np.linalg.inv(world_to_img), gt_px) - origin

    est_m = to_metres(sheet.image_to_m, gt_px)
    err_mm = 1000 * np.linalg.norm(est_m - gt_m, axis=1)
    print(f"homography only (true px -> m): mean {err_mm.mean():.2f} mm, max {err_mm.max():.2f} mm")
    # A constant offset doesn't change any fitted constant; scale does (free-fall g scales 1:1 with it).
    d_est, d_gt = est_m - est_m.mean(axis=0), gt_m - gt_m.mean(axis=0)
    scale = float((d_est * d_gt).sum() / (d_gt * d_gt).sum())
    shape_mm = 1000 * np.linalg.norm(d_est - d_gt, axis=1)
    print(f"scale error {100 * (scale - 1):+.2f}%, error after removing the offset: mean {shape_mm.mean():.2f} mm, max {shape_mm.max():.2f} mm")

    if args.track:
        traj = from_track(json.loads(args.track.read_text()), sheet.image_to_m)
        e = 1000 * np.hypot(traj.x - gt_m[traj.frames, 0], traj.y - gt_m[traj.frames, 1])
        print(f"end to end (SAM + homography): mean {e.mean():.2f} mm, p95 {np.percentile(e, 95):.2f} mm, max {e.max():.2f} mm")
        span = np.ptp(gt_m[:, 0]), np.ptp(gt_m[:, 1])
        print(f"motion span: x {1000 * span[0]:.0f} mm, y {1000 * span[1]:.0f} mm")


if __name__ == "__main__":
    main()
