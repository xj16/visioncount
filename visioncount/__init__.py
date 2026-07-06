"""VisionCount - real-time object counting & line-crossing analytics.

A dependency-light OpenCV pipeline that detects, tracks and counts objects
crossing virtual lines in a video / webcam feed, plus a Flask dashboard for
live counts and motion heatmaps.

The default detector uses OpenCV background subtraction + centroid tracking so
the whole pipeline runs with only ``opencv-python-headless`` and ``numpy``.
An optional YOLO (ultralytics) detector is available and is imported lazily so a
missing ``torch``/``ultralytics`` never breaks ``import visioncount``.
"""

from __future__ import annotations

from .detector import BackgroundSubtractorDetector, Detection
from .tracker import CentroidTracker, Track
from .counter import LineCounter, CountingLine
from .heatmap import Heatmap
from .pipeline import Pipeline, PipelineConfig, FrameResult

__all__ = [
    "BackgroundSubtractorDetector",
    "Detection",
    "CentroidTracker",
    "Track",
    "LineCounter",
    "CountingLine",
    "Heatmap",
    "Pipeline",
    "PipelineConfig",
    "FrameResult",
    "__version__",
]

__version__ = "0.1.0"
