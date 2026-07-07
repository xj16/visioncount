"""Generate demo artifacts used in the README.

Runs the default pipeline on the built-in synthetic clip and writes:

* ``docs/demo_frame.png``   - an annotated frame with boxes, IDs and lines
* ``docs/demo_heatmap.png`` - the accumulated motion heatmap overlaid on a frame
* ``docs/demo_report.png``  - the two-panel Matplotlib analytics report
* ``docs/demo.gif``         - (``--gif``) a looping clip of the live pipeline:
                              boxes, track IDs, trajectories, two counting lines
                              incrementing, a zone filling, and the heatmap blooming

Run from the repo root::

    python examples/generate_demo.py           # PNGs only (fast)
    python examples/generate_demo.py --gif      # also write docs/demo.gif

The GIF uses only Pillow (a Matplotlib dependency), so no ffmpeg is required.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from visioncount import CountingLine, Pipeline, Zone
from visioncount.report import render_report
from visioncount.synthetic import frames

W, H = 480, 320
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")


def _build_pipeline() -> Pipeline:
    return Pipeline(
        W,
        H,
        lines=[
            CountingLine("entry", (W // 2, 0), (W // 2, H)),
            CountingLine("upper", (0, H // 3), (W, H // 3)),
        ],
        zones=[Zone("lobby", [(60, 90), (200, 90), (200, 230), (60, 230)])],
    )


def _bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def generate_images() -> None:
    pipe = _build_pipeline()
    last_frame = None
    last_result = None
    series = []
    t0 = time.time()
    for i, frame in enumerate(frames(width=W, height=H, n_frames=140)):
        last_result = pipe.process(frame)
        last_frame = frame
        grand = sum(c["total"] for c in last_result.totals.values())
        series.append((t0 + i * 0.04, grand))

    assert last_frame is not None and last_result is not None
    annotated = pipe.annotate(last_frame, last_result)
    cv2.imwrite(os.path.join(OUT_DIR, "demo_frame.png"), annotated)

    heat_overlay = pipe.heatmap.overlay(last_frame, alpha=0.55)
    cv2.imwrite(os.path.join(OUT_DIR, "demo_heatmap.png"), heat_overlay)

    render_report(
        pipe.totals(),
        series,
        os.path.join(OUT_DIR, "demo_report.png"),
        title="VisionCount analytics report",
    )
    print("Counts:", pipe.totals())
    print("Wrote docs/demo_frame.png, docs/demo_heatmap.png and docs/demo_report.png")


def generate_gif(
    n_frames: int = 190,
    fps: int = 16,
    stride: int = 3,
    scale: float = 0.72,
    colors: int = 96,
) -> None:
    """Render a compact looping GIF of the live pipeline to ``docs/demo.gif``.

    Downscaled and palette-reduced so the result stays comfortably small enough to
    embed at the top of the README (GitHub renders inline GIFs up to ~10 MB).
    """
    from PIL import Image

    pipe = _build_pipeline()
    gw, gh = int(W * scale), int(H * scale)
    pil_frames = []
    for i, frame in enumerate(frames(width=W, height=H, n_frames=n_frames)):
        result = pipe.process(frame)
        annotated = pipe.annotate(frame, result)
        annotated = pipe.heatmap.overlay(annotated, alpha=0.4)
        if i % stride == 0:
            small = cv2.resize(annotated, (gw, gh), interpolation=cv2.INTER_AREA)
            img = Image.fromarray(_bgr_to_rgb(small)).convert("RGB")
            # Reduce to an adaptive palette per frame (keeps the file small).
            pil_frames.append(img.quantize(colors=colors, method=Image.MEDIANCUT))

    if not pil_frames:
        raise RuntimeError("no frames rendered for GIF")

    duration_ms = int(1000 / fps * stride)
    out = os.path.join(OUT_DIR, "demo.gif")
    pil_frames[0].save(
        out,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_kb = os.path.getsize(out) / 1024
    print(f"Wrote {out} ({len(pil_frames)} frames, {gw}x{gh}, {size_kb:.0f} KB)")
    print("Final counts:", pipe.totals())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VisionCount demo artifacts.")
    parser.add_argument("--gif", action="store_true", help="also render docs/demo.gif")
    parser.add_argument(
        "--gif-only", action="store_true", help="render only the GIF (skip PNGs)"
    )
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.gif_only:
        generate_images()
    if args.gif or args.gif_only:
        generate_gif()


if __name__ == "__main__":
    main()
