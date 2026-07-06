"""Generate demo artifacts used in the README.

Runs the default pipeline on the built-in synthetic clip and writes:

* ``docs/demo_frame.png``  - an annotated frame with boxes, IDs and the line
* ``docs/demo_heatmap.png`` - the accumulated motion heatmap overlaid on a frame

Run from the repo root::

    python examples/generate_demo.py
"""

from __future__ import annotations

import os

import time

import cv2

from visioncount import CountingLine, Pipeline
from visioncount.report import render_report
from visioncount.synthetic import frames

W, H = 480, 320
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    pipe = Pipeline(
        W,
        H,
        lines=[
            CountingLine("entry", (W // 2, 0), (W // 2, H)),
            CountingLine("upper", (0, H // 3), (W, H // 3)),
        ],
    )
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
        pipe.totals(), series, os.path.join(OUT_DIR, "demo_report.png"),
        title="VisionCount analytics report",
    )

    print("Counts:", pipe.totals())
    print("Wrote docs/demo_frame.png, docs/demo_heatmap.png and docs/demo_report.png")


if __name__ == "__main__":
    main()
