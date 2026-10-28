"""Accuracy benchmark: 20 synthetic runs through the real pipeline (SAM 3.1 -> A4 homography -> physics fit),
scored against exact ground truth. Writes docs/ACCURACY.md and output/benchmark/results.json.

    .venv/bin/python scripts/benchmark.py            # renders, tracks (cached) and scores
    .venv/bin/python scripts/benchmark.py --only p1  # one run

SAM cost for the full set is about $8.
"""

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import sam  # noqa: E402
from backend.config import OUTPUT_DIR, REPO_DIR  # noqa: E402
from backend.pipeline import AnalysisError, analyze  # noqa: E402
from synth import Spec, render  # noqa: E402

OUT = OUTPUT_DIR / "benchmark"

SPECS = [
    Spec("p1", "pendulum", {"length_m": 0.30, "amplitude_deg": 10}),
    Spec("p2", "pendulum", {"length_m": 0.45, "amplitude_deg": 15}),
    Spec("p3", "pendulum", {"length_m": 0.60, "amplitude_deg": 8}, camera="tilted"),
    Spec("p4", "pendulum", {"length_m": 0.75, "amplitude_deg": 20}),
    Spec("p5", "pendulum", {"length_m": 0.90, "amplitude_deg": 12}, camera="far"),
    Spec("p6", "pendulum", {"length_m": 0.50, "amplitude_deg": 12}, light=0.45, noise=6),
    Spec("p7", "pendulum", {"length_m": 0.50, "amplitude_deg": 15}, blur=4),
    Spec("p8", "pendulum", {"length_m": 0.60, "amplitude_deg": 10}, camera="tilted", light=0.45, noise=6),
    Spec("f1", "free_fall", {"drop_m": 0.60}),
    Spec("f2", "free_fall", {"drop_m": 0.80}, camera="tilted"),
    Spec("f3", "free_fall", {"drop_m": 1.00}, camera="far"),
    Spec("f4", "free_fall", {"drop_m": 1.05}, blur=4),
    Spec("f5", "free_fall", {"drop_m": 0.90}, light=0.45, noise=6),
    Spec("f6", "free_fall", {"drop_m": 0.70}),
    Spec("s1", "spring", {"mass_kg": 0.20, "k": 10}),
    Spec("s2", "spring", {"mass_kg": 0.20, "k": 15}, camera="tilted"),
    Spec("s3", "spring", {"mass_kg": 0.20, "k": 20}, camera="far"),
    Spec("s4", "spring", {"mass_kg": 0.30, "k": 30}, blur=4),
    Spec("s5", "spring", {"mass_kg": 0.25, "k": 40}, light=0.45, noise=6),
    Spec("s6", "spring", {"mass_kg": 0.15, "k": 25}),
]


def condition(s: Spec) -> str:
    parts = [] if s.camera == "front" else [f"{s.camera} camera"]
    if s.light < 1:
        parts.append("dim + noisy")
    if s.blur > 1:
        parts.append("motion blur")
    return ", ".join(parts) or "baseline"


async def track(spec: Spec, video: Path, truth: dict, sem: asyncio.Semaphore) -> tuple[dict, str]:
    """The app's recipe: the text concept first; if SAM finds nothing, a tap on frame 0 (simulated at the true centre)."""
    for prompt in ("text", "tap"):
        cache = OUT / f"{spec.name}.{prompt}.track.json"
        if not cache.exists():
            async with sem:
                info = sam.probe_video(video)
                file_id = await sam.upload_video(video)
                tap = (0, *truth["start_px"]) if prompt == "tap" else None
                points, meta = await sam.track_clip(file_id, info, truth["concept"], tap)
                t = sam.to_json(points, info, truth["concept"], file_id, meta)
            cache.write_text(json.dumps(t))
            print(f"tracked {spec.name} ({prompt}): {len(t['points'])}/{info['frames']} frames", flush=True)
        t = json.loads(cache.read_text())
        if t["points"]:
            return t, prompt
    return t, "tap"


async def run(spec: Spec, sem: asyncio.Semaphore) -> dict:
    video, truth = OUT / f"{spec.name}.mp4", None
    truth_file = OUT / f"{spec.name}.truth.json"
    if not video.exists() or not truth_file.exists():
        video, truth = await asyncio.to_thread(render, spec, OUT)
        truth_file.write_text(json.dumps(truth))
    truth = json.loads(truth_file.read_text())
    t, prompt = await track(spec, video, truth, sem)
    row = {"name": spec.name, "experiment": spec.experiment, "params": spec.params, "condition": condition(spec), "prompt": prompt}
    try:
        a = await asyncio.to_thread(analyze, str(video), t, spec.experiment, mass_kg=truth.get("mass_kg"))
    except AnalysisError as e:
        return {**row, "error": str(e)}
    if spec.experiment == "pendulum":  # the same track, with the string length typed in as the app allows
        b = await asyncio.to_thread(analyze, str(video), t, "pendulum", length_m=spec.params["length_m"])
        row["g_with_length"] = b["fit"]["measurements"]["g"]
    q = "k" if spec.experiment == "spring" else "g"
    m = a["fit"]["measurements"][q]
    err = (m["value"] - truth[q]) / truth[q] * 100
    return {**row, "quantity": q, "true": truth[q], "measured": m["value"], "sigma": m["sigma"], "error_pct": err,
            "z": (m["value"] - truth[q]) / m["sigma"] if m["sigma"] else None,
            "tracked_frames": len(t["points"]), "frames": t["video"]["frames"]}


def report(rows: list[dict]) -> str:
    unit = {"g": "m/s²", "k": "N/m"}
    label = {"pendulum": lambda p: f"L = {p['length_m']} m, {p['amplitude_deg']}°", "free_fall": lambda p: f"drop {p['drop_m']} m",
             "spring": lambda p: f"m = {p['mass_kg']} kg, k = {p['k']} N/m"}
    ok = [r for r in rows if "error" not in r]
    lines = [
        "# Accuracy benchmark",
        "",
        "20 synthetic 240 fps clips with exact ground truth, run through the same pipeline as the app: SAM 3.1 tracking "
        "(concept prompt, then a box re-prompt), A4 sheet detection and homography, then the physics fit. "
        "Conditions vary the camera angle and distance, lighting, sensor noise and motion blur. "
        "Reproduce with `scripts/benchmark.py`; the clips come from `scripts/synth.py`.",
        "",
        "SAM prompt: `text` is the colour-and-shape concept (e.g. \"red ball\") followed by a box re-prompt; `tap` is the "
        "fallback when the concept finds nothing, a point prompt on frame 0 as a user's tap in the app.",
        "",
        "These are rendered scenes, not real footage. They test the measurement chain against known answers; real-phone results are reported separately.",
        "",
        "## Summary",
        "",
        "| Experiment | Runs | Mean abs. error | Worst error | Within 2σ |",
        "|---|---|---|---|---|",
    ]
    for exp in ("pendulum", "free_fall", "spring"):
        rs = [r for r in ok if r["experiment"] == exp]
        if rs:
            errs = [abs(r["error_pct"]) for r in rs]
            within = sum(1 for r in rs if r["z"] is not None and abs(r["z"]) <= 2)
            lines.append(f"| {exp.replace('_', ' ')} ({rs[0]['quantity']}) | {len(rs)} | {statistics.mean(errs):.2f}% | {max(errs):.2f}% | {within}/{len(rs)} |")
    if ok:
        errs = [abs(r["error_pct"]) for r in ok]
        lines.append(f"| **all** | {len(ok)} | {statistics.mean(errs):.2f}% | {max(errs):.2f}% | {sum(1 for r in ok if r['z'] is not None and abs(r['z']) <= 2)}/{len(ok)} |")
    lines += ["", "## Runs", "", "| Run | Setup | Condition | SAM prompt | True | Measured | Error | z | Tracked frames |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        setup = label[r["experiment"]](r["params"])
        if "error" in r:
            lines.append(f"| {r['name']} | {setup} | {r['condition']} | {r['prompt']} | | failed: {r['error']} | | | |")
            continue
        u = unit[r["quantity"]]
        lines.append(
            f"| {r['name']} | {setup} | {r['condition']} | {r['prompt']} | {r['quantity']} = {r['true']:g} {u} | {r['measured']:.3f} ± {r['sigma']:.3f} | "
            f"{r['error_pct']:+.2f}% | {r['z']:+.1f} | {r['tracked_frames']}/{r['frames']} |"
        )
    pend = [r for r in ok if "g_with_length" in r]
    if pend:
        lines += [
            "",
            "## Pendulum with the string length entered",
            "",
            "From the swing alone, the length comes from the arc's curvature, which is weak for short or narrow swings "
            "(p1, p3), and the error bar says so. The period is always precise, so typing in the measured string length "
            "fixes it. Below, the same tracks with the true length entered; a real tape measure adds about ±2 mm, which "
            "the error bar includes.",
            "",
            "| Run | Measured g | Error |",
            "|---|---|---|",
        ]
        for r in pend:
            g = r["g_with_length"]
            lines.append(f"| {r['name']} | {g['value']:.3f} ± {g['sigma']:.3f} | {(g['value'] / r['true'] - 1) * 100:+.2f}% |")
    return "\n".join(lines) + "\n"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="run names, e.g. p1 f2")
    args = ap.parse_args()
    specs = [s for s in SPECS if not args.only or s.name in args.only]
    OUT.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(4)
    rows = await asyncio.gather(*(run(s, sem) for s in specs))
    for r in rows:
        print(r["name"], r.get("error") or f"{r['quantity']} {r['measured']:.4f} ({r['error_pct']:+.2f}%, z {r['z']:+.1f})")
    (OUT / "results.json").write_text(json.dumps(rows, indent=1))
    if not args.only:
        (REPO_DIR / "docs").mkdir(exist_ok=True)
        (REPO_DIR / "docs" / "ACCURACY.md").write_text(report(rows))
        print("wrote docs/ACCURACY.md")


if __name__ == "__main__":
    asyncio.run(main())
