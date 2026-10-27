"""Score a SAM track against a synthetic clip's ground-truth track_px."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("track", type=Path)
    ap.add_argument("truth", type=Path)
    ap.add_argument("--object-id", help="defaults to the id seen on the most frames")
    args = ap.parse_args()

    track = json.loads(args.track.read_text())
    truth = json.loads(args.truth.read_text())
    gt = np.array(truth["track_px"])
    n = len(gt)

    counts = Counter(p["object_id"] for p in track["points"])
    oid = args.object_id or counts.most_common(1)[0][0]
    pts = {p["frame"]: p for p in track["points"] if p["object_id"] == oid}

    frames = sorted(f for f in pts if f < n)
    err = np.array([np.hypot(pts[f]["cx"] - gt[f][0], pts[f]["cy"] - gt[f][1]) for f in frames])
    missing = n - len(frames)
    worst = frames[int(err.argmax())]

    print(f"object ids seen: {dict(counts)} -> scoring id {oid}")
    print(f"frames: {len(frames)}/{n} tracked, {missing} missing ({100 * missing / n:.2f}%)")
    print(f"centroid error px: mean {err.mean():.2f}, median {np.median(err):.2f}, p95 {np.percentile(err, 95):.2f}, max {err.max():.2f} (frame {worst})")
    jumps = np.hypot(*np.diff(np.array([[pts[f]["cx"], pts[f]["cy"]] for f in frames]), axis=0).T)
    gt_jumps = np.hypot(*np.diff(gt[frames], axis=0).T)
    print(f"frame-to-frame step px: max {jumps.max():.2f} (truth max {gt_jumps.max():.2f})")
    confs = [pts[f]["confidence"] for f in frames if pts[f]["confidence"] is not None]
    if confs:
        print(f"confidence: min {min(confs):.3f}, mean {np.mean(confs):.3f}")


if __name__ == "__main__":
    main()
