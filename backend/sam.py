"""SAM 3.1 video tracking: upload a clip, stream the segmentation, reduce it to per-frame centroids."""

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

from meta_sam_parser import (
    FramePrompt,
    PromptBox,
    PromptObject,
    PromptPoint,
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    build_object_prompt,
    decode_mask_to_raster,
    parse_responses_stream,
    video_segmentation_format,
)
from openai import AsyncOpenAI

from .config import META_BASE_URL, SAM_MODEL, api_key


@dataclass
class FramePoint:
    frame: int
    object_id: str
    cx: float  # centroid, source pixels
    cy: float
    box: tuple[float, float, float, float]  # left, top, right, bottom (half-open)
    area_px: int  # mask foreground pixels, 0 if no mask
    confidence: float | None


def probe_video(path: Path) -> dict:
    """fps, frame count and size from ffprobe. Time = frame / fps."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
            "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_packets",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    s = json.loads(out.stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": float(num) / float(den),
        "frames": int(s["nb_read_packets"]),
    }


def client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=META_BASE_URL, api_key=api_key())


async def upload_video(path: Path) -> str:
    with open(path, "rb") as f:
        file = await client().files.create(purpose="user_data", file=f)
    return file.id


async def track(file_id: str, concept: str) -> tuple[list[FramePoint], dict]:
    """Stream sam-3.1 over an uploaded video and return per-frame centroids for every tracked object."""
    return await _segment(file_id, concept, {})


async def track_box(
    file_id: str, size: tuple[int, int], frame: int, box: tuple[float, float, float, float], object_id: str = "0"
) -> tuple[list[FramePoint], dict]:
    """Track one object from a box on one frame, in both directions.

    Concept prompts can lock on a few frames late; re-prompting with the first box found recovers the start.
    """
    prompt = build_object_prompt(
        size=size,
        objects=[PromptObject(id=object_id, prompts=[FramePrompt(frame=frame, box=PromptBox(*box))])],
    )
    return await _segment(file_id, prompt, {"propagation_direction": "both"})


async def track_point(
    file_id: str, size: tuple[int, int], frame: int, xy: tuple[float, float], object_id: str = "0"
) -> tuple[list[FramePoint], dict]:
    """Track the object the user tapped on one frame, in both directions.

    Small or plain objects (a marble seen from 2 m) often don't match any text concept; a tap always names them.
    """
    point = PromptPoint(int(round(xy[0])), int(round(xy[1])))
    prompt = build_object_prompt(size=size, objects=[PromptObject(id=object_id, prompts=[FramePrompt(frame=frame, points=[point])])])
    return await _segment(file_id, prompt, {"propagation_direction": "both"})


async def track_clip(
    file_id: str, video: dict, concept: str, tap: tuple[int, float, float] | None = None
) -> tuple[list[FramePoint], dict]:
    """The app's tracking recipe. A tap (frame, x, y) is used as is; a text concept is followed by a box
    re-prompt from its first detection, which fills the frames before the concept locked on."""
    size = (video["width"], video["height"])
    if tap:
        frame, x, y = tap
        return await track_point(file_id, size, frame, (x, y))
    points, meta = await track(file_id, concept)
    if points:
        first = min(points, key=lambda p: p.frame)
        points, meta = await track_box(file_id, size, first.frame, first.box, first.object_id)
    return points, meta


async def _segment(file_id: str, text: str, extra_metadata: dict) -> tuple[list[FramePoint], dict]:
    stream = await client().responses.create(
        model=SAM_MODEL,
        input=[{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": text},
                {"type": "input_video", "file_id": file_id},
            ],
        }],
        metadata={"mask_encoding": "one_bit", "include_confidence": "true", **extra_metadata},
        stream=True,
    )
    result = await parse_responses_stream(stream, video_segmentation_format()).final_result()

    boxes: dict[tuple[int, str], SegmentationBoxRecord] = {}
    masks: dict[tuple[int, str], SegmentationMaskRecord] = {}
    for rec in result.records:
        if rec.frame is None:
            continue
        key = (rec.frame.frame_index, str(rec.object_id))
        if isinstance(rec, SegmentationBoxRecord):
            boxes[key] = rec
        elif isinstance(rec, SegmentationMaskRecord):
            masks[key] = rec

    points = [_frame_point(key, box, masks.get(key)) for key, box in boxes.items()]
    points.sort(key=lambda p: (p.frame, p.object_id))
    meta = {
        "outcome": type(result.outcome).__name__,
        "diagnostics": [str(d) for d in result.diagnostics],
    }
    return points, meta


def _frame_point(key, box: SegmentationBoxRecord, mask: SegmentationMaskRecord | None) -> FramePoint:
    frame, object_id = key
    cx, cy = (box.left + box.right) / 2, (box.top + box.bottom) / 2
    area = 0
    if mask is not None:
        m = mask.mask
        raster = decode_mask_to_raster(m)  # row-major 0/1 bytes, height x width
        b = mask.bounds
        sx, sy = (b.right - b.left) / m.width, (b.bottom - b.top) / m.height
        sum_x = sum_y = 0.0
        for row in range(m.height):
            line = raster[row * m.width:(row + 1) * m.width]
            n = line.count(1)
            if n:
                cols = [c for c, v in enumerate(line) if v]
                sum_x += sum(cols) + 0.5 * n
                sum_y += (row + 0.5) * n
                area += n
        if area:
            # Mask pixel centres mapped back to source pixels; -0.5 converts to pixel-centre coordinates.
            cx = b.left + sx * sum_x / area - 0.5
            cy = b.top + sy * sum_y / area - 0.5
    conf = box.confidence if box.confidence is not None else (mask.confidence if mask else None)
    return FramePoint(frame, object_id, cx, cy, (box.left, box.top, box.right, box.bottom), area, conf)


def to_json(points: list[FramePoint], video: dict, concept: str, file_id: str, meta: dict) -> dict:
    return {
        "concept": concept,
        "file_id": file_id,
        "video": video,
        "meta": meta,
        "points": [asdict(p) for p in points],
    }
