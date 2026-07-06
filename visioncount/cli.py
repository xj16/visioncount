"""Command-line entrypoint for VisionCount.

Examples
--------
Run the headless pipeline on the built-in synthetic clip and print counts::

    python -m visioncount --source synthetic --frames 200 --no-window

Process a video file and write an annotated output::

    python -m visioncount --source path/to/video.mp4 --output out.mp4

Live webcam (device 0) with a vertical center line::

    python -m visioncount --source 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .counter import CountingLine
from .pipeline import Pipeline, PipelineConfig
from .synthetic import frames as synthetic_frames


def _parse_line(spec: str) -> Tuple[str, Tuple[int, int], Tuple[int, int]]:
    """Parse ``name:x1,y1,x2,y2`` into (name, start, end)."""
    try:
        name, coords = spec.split(":", 1)
        x1, y1, x2, y2 = (int(v) for v in coords.split(","))
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(
            f"invalid --line '{spec}', expected name:x1,y1,x2,y2"
        ) from exc
    return name, (x1, y1), (x2, y2)


def _iter_source(
    source: str, max_frames: int, width: int, height: int
) -> Iterator[np.ndarray]:
    """Yield frames from a synthetic clip, a device index, or a file path."""
    if source == "synthetic":
        yield from synthetic_frames(width=width, height=height, n_frames=max_frames)
        return

    cap_arg: object = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_arg)
    if not cap.isOpened():
        raise SystemExit(f"error: could not open source '{source}'")
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            count += 1
            if max_frames and count >= max_frames:
                break
    finally:
        cap.release()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="visioncount",
        description="Detect, track and count objects crossing virtual lines.",
    )
    p.add_argument(
        "--source",
        default="synthetic",
        help="'synthetic', a webcam index like '0', or a video file path.",
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Max frames to process (0 = unlimited for real sources).",
    )
    p.add_argument(
        "--line",
        action="append",
        type=_parse_line,
        metavar="name:x1,y1,x2,y2",
        help="Counting line(s). Repeatable. Defaults to a vertical center line.",
    )
    p.add_argument("--min-area", type=int, default=400)
    p.add_argument("--output", help="Optional path to write an annotated MP4.")
    p.add_argument(
        "--report",
        metavar="PATH.png",
        help="Write a Matplotlib PNG analytics report at the end.",
    )
    p.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open a GUI window (required on headless machines).",
    )
    p.add_argument(
        "--yolo",
        metavar="WEIGHTS",
        help="Use the optional YOLO detector with these weights (needs torch).",
    )
    p.add_argument("--json", action="store_true", help="Print final totals as JSON.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # First frame determines the real resolution for file/webcam sources.
    source_iter = _iter_source(args.source, args.frames, args.width, args.height)
    try:
        first = next(source_iter)
    except StopIteration:
        raise SystemExit("error: source produced no frames")

    height, width = first.shape[:2]

    lines: List[CountingLine] = []
    if args.line:
        for name, start, end in args.line:
            lines.append(CountingLine(name=name, start=start, end=end))

    detector = None
    if args.yolo:
        from .yolo_detector import YoloDetector  # lazy, may need torch

        detector = YoloDetector(weights=args.yolo)

    pipe = Pipeline(
        width,
        height,
        lines=lines or None,
        detector=detector,
        config=PipelineConfig(min_area=args.min_area),
    )

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, 25.0, (width, height))

    series: List[Tuple[float, int]] = []

    def handle(frame: np.ndarray) -> None:
        result = pipe.process(frame)
        for ev in result.events:
            print(f"[frame {result.frame_index}] {ev.line}: {ev.direction} #{ev.track_id}")
        grand = sum(c["total"] for c in result.totals.values())
        series.append((time.time(), grand))
        if writer is not None or not args.no_window:
            annotated = pipe.annotate(frame, result)
            annotated = pipe.heatmap.overlay(annotated, alpha=0.4)
            if writer is not None:
                writer.write(annotated)
            if not args.no_window:
                cv2.imshow("VisionCount", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

    try:
        handle(first)
        for frame in source_iter:
            handle(frame)
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    totals = pipe.totals()
    if args.report:
        from .report import render_report

        render_report(totals, series, args.report)
        print(f"Wrote report to {args.report}")

    if args.json:
        print(json.dumps(totals, indent=2))
    else:
        print("\nFinal counts:")
        for name, c in totals.items():
            print(f"  {name}: in={c['in']} out={c['out']} total={c['total']} net={c['net']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
