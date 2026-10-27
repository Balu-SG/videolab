"""Upload a clip to the Meta Model API and track one concept with SAM 3.1.

Writes output/<clip>.track.json. Pass --file-id to reuse an earlier upload.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import sam  # noqa: E402
from backend.config import OUTPUT_DIR  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--file-id")
    ap.add_argument("--refine", action="store_true",
                    help="re-prompt with the first box found and track both directions, to recover early frames")
    args = ap.parse_args()

    video = sam.probe_video(args.video)
    print(f"{args.video.name}: {video['width']}x{video['height']} @ {video['fps']:.2f} fps, {video['frames']} frames")

    file_id = args.file_id
    if not file_id:
        t = time.time()
        file_id = await sam.upload_video(args.video)
        print(f"uploaded as {file_id} in {time.time() - t:.1f}s")

    t = time.time()
    points, meta = await sam.track(file_id, args.concept)
    ids = sorted({p.object_id for p in points})
    print(f"tracked '{args.concept}' in {time.time() - t:.1f}s: {len(points)} records, object ids {ids}, outcome {meta['outcome']}")
    if meta["diagnostics"]:
        print(f"{len(meta['diagnostics'])} diagnostics, first: {meta['diagnostics'][0]}")

    if args.refine and points:
        first = min(points, key=lambda p: p.frame)
        t = time.time()
        points, meta = await sam.track_box(file_id, (video["width"], video["height"]), first.frame, first.box, first.object_id)
        print(f"refined from frame {first.frame} in {time.time() - t:.1f}s: {len(points)} records, outcome {meta['outcome']}")
        if meta["diagnostics"]:
            print(f"{len(meta['diagnostics'])} diagnostics, first: {meta['diagnostics'][0]}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f"{args.video.stem}.track.json"
    out.write_text(json.dumps(sam.to_json(points, video, args.concept, file_id, meta), indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
