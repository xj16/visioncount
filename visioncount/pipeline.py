"""End-to-end VisionCount pipeline.

Wires together a detector, the centroid tracker, the line counter and the
heatmap into a single ``process()`` call, and can annotate frames for display.
The detector is pluggable: the default is
:class:`~visioncount.detector.BackgroundSubtractorDetector`, but any object with
a ``detect(frame) -> list[Detection]`` method works (including the optional
:class:`~visioncount.yolo_detector.YoloDetector`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

import cv2
import numpy as np

from .counter import CountingLine, CrossEvent, LineCounter
from .detector import BackgroundSubtractorDetector, Detection
from .heatmap import Heatmap
from .tracker import CentroidTracker, Track


class DetectorLike(Protocol):
    def detect(self, frame: np.ndarray) -> List[Detection]:  # pragma: no cover
        ...


@dataclass
class PipelineConfig:
    """Tuning knobs for the pipeline."""

    min_area: int = 400
    max_disappeared: int = 20
    max_distance: float = 80.0
    heatmap_decay: float = 0.99
    heatmap_radius: int = 16
    draw_trajectories: bool = True


@dataclass
class FrameResult:
    """Everything the pipeline produced for one frame."""

    detections: List[Detection]
    tracks: Dict[int, Track]
    events: List[CrossEvent]
    totals: Dict[str, Dict[str, int]]
    frame_index: int


# A small palette for drawing distinct track IDs.
_PALETTE = [
    (66, 135, 245),
    (46, 204, 113),
    (231, 76, 60),
    (241, 196, 15),
    (155, 89, 182),
    (26, 188, 156),
    (230, 126, 34),
]


class Pipeline:
    """Orchestrates detection -> tracking -> counting -> heatmap."""

    def __init__(
        self,
        width: int,
        height: int,
        lines: Optional[List[CountingLine]] = None,
        detector: Optional[DetectorLike] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.config = config or PipelineConfig()
        self.detector: DetectorLike = detector or BackgroundSubtractorDetector(
            min_area=self.config.min_area
        )
        self.tracker = CentroidTracker(
            max_disappeared=self.config.max_disappeared,
            max_distance=self.config.max_distance,
        )
        self.counter = LineCounter(lines or self._default_lines())
        self.heatmap = Heatmap(
            self.width,
            self.height,
            decay=self.config.heatmap_decay,
            blob_radius=self.config.heatmap_radius,
        )
        self.frame_index = -1

    def _default_lines(self) -> List[CountingLine]:
        """A single vertical line down the middle of the frame."""
        mid = self.width // 2
        return [
            CountingLine(
                name="center",
                start=(mid, 0),
                end=(mid, self.height),
            )
        ]

    def process(self, frame: np.ndarray) -> FrameResult:
        """Run one frame through the full pipeline."""
        self.frame_index += 1
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections)
        events = self.counter.update(tracks)
        self.heatmap.update_from_tracks(tracks)
        return FrameResult(
            detections=detections,
            tracks=tracks,
            events=events,
            totals=self.counter.totals(),
            frame_index=self.frame_index,
        )

    # ---- drawing helpers -------------------------------------------------

    def annotate(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Return a copy of ``frame`` with boxes, ids, lines and counts drawn."""
        out = frame.copy()

        # Counting lines.
        for line in self.counter.lines:
            p1 = (int(line.start[0]), int(line.start[1]))
            p2 = (int(line.end[0]), int(line.end[1]))
            cv2.line(out, p1, p2, (0, 215, 255), 2)
            label = f"{line.name}: in {line.count_in} / out {line.count_out}"
            cv2.putText(
                out,
                label,
                (max(0, p1[0] - 40), max(20, p1[1] + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 215, 255),
                1,
                cv2.LINE_AA,
            )

        # Tracks.
        for track in result.tracks.values():
            color = _PALETTE[track.id % len(_PALETTE)]
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                f"#{track.id}",
                (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            if self.config.draw_trajectories and len(track.trajectory) >= 2:
                pts = np.array(track.trajectory, dtype=np.int32)
                cv2.polylines(out, [pts], isClosed=False, color=color, thickness=1)

        return out

    def reset(self) -> None:
        self.tracker.reset()
        self.counter.reset()
        self.heatmap.reset()
        self.frame_index = -1
        if hasattr(self.detector, "reset"):
            self.detector.reset()  # type: ignore[attr-defined]

    def totals(self) -> Dict[str, Dict[str, int]]:
        return self.counter.totals()
