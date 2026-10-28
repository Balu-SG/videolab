"""Create a run: track a clip with SAM 3.1 (or reuse a saved track), analyse it and store it under output/runs/."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import runs, sam  # noqa: E402
from backend.pipeline import analyze  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("experiment", choices=["pendulum", "free_fall", "spring"])
    ap.add_argument("--concept", help="SAM concept prompt, e.g. 'orange ball'")
    ap.add_argument("--track", type=Path, help="reuse a saved track instead of calling SAM")
    ap.add_argument("--length-m", type=float)
    ap.add_argument("--mass-kg", type=float)
    ap.add_argument("--predict", nargs=3, metavar=("QUANTITY", "VALUE", "UNIT"), help="e.g. period 1.5 s")
    args = ap.parse_args()

    if args.track:
        track = json.loads(args.track.read_text())
    else:
        if not args.concept:
            sys.exit("--concept is required without --track")
        video = sam.probe_video(args.video)
        file_id = await sam.upload_video(args.video)
        points, meta = await sam.track_clip(file_id, video, args.concept)
        track = sam.to_json(points, video, args.concept, file_id, meta)

    analysis = analyze(str(args.video), track, args.experiment, length_m=args.length_m, mass_kg=args.mass_kg)
    meta = {"video": str(args.video.resolve()), "experiment": args.experiment, "concept": track.get("concept"),
            "length_m": args.length_m, "mass_kg": args.mass_kg}
    if args.predict:
        q, v, u = args.predict
        meta["prediction"] = {"quantity": q, "value": float(v), "unit": u}
    run_id = runs.new_run_id()
    runs.save(run_id, meta, track, analysis)
    print(run_id, {k: round(m["value"], 4) for k, m in analysis["fit"]["measurements"].items()})


if __name__ == "__main__":
    asyncio.run(main())
