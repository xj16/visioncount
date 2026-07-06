"""End-to-end pipeline tests on a synthetic clip.

These are the core "does the whole thing actually work" tests required by the
project spec: the OpenCV pipeline must run on a synthetic clip via pytest.
"""

from __future__ import annotations

import numpy as np

from visioncount import CountingLine, Detection, Pipeline
from visioncount.detector import BackgroundSubtractorDetector
from visioncount.synthetic import frames, write_clip


def test_detector_finds_moving_blob():
    det = BackgroundSubtractorDetector(min_area=100)
    found_any = False
    for frame in frames(width=320, height=240, n_frames=40):
        dets = det.detect(frame)
        if dets:
            found_any = True
    assert found_any, "detector should find the moving blobs"


def test_pipeline_runs_and_counts_on_synthetic_clip():
    W, H = 320, 240
    pipe = Pipeline(
        W,
        H,
        lines=[CountingLine("center", (W // 2, 0), (W // 2, H))],
    )
    last = None
    for frame in frames(width=W, height=H, n_frames=160):
        last = pipe.process(frame)

    assert last is not None
    assert last.frame_index == 159
    totals = pipe.totals()
    # The default objects sweep across the center line -> at least one crossing.
    assert totals["center"]["total"] >= 1


def test_pipeline_builds_heatmap():
    W, H = 320, 240
    pipe = Pipeline(W, H)
    for frame in frames(width=W, height=H, n_frames=60):
        pipe.process(frame)
    assert pipe.heatmap.buffer.max() > 0.0
    colorized = pipe.heatmap.colorized()
    assert colorized.shape == (H, W, 3)


def test_annotate_returns_same_shape():
    W, H = 320, 240
    pipe = Pipeline(W, H)
    frame = next(iter(frames(width=W, height=H, n_frames=1)))
    result = pipe.process(frame)
    annotated = pipe.annotate(frame, result)
    assert annotated.shape == frame.shape
    assert annotated is not frame  # must be a copy


def test_custom_detector_interface():
    """A user-supplied detector with .detect() should plug in."""

    class FakeDetector:
        def detect(self, frame):
            # Always report one object walking rightward at row 50.
            x = int((getattr(self, "_i", 0)) * 12)
            self._i = getattr(self, "_i", 0) + 1
            return [Detection(x=x, y=45, w=10, h=10)]

    W, H = 200, 100
    pipe = Pipeline(
        W,
        H,
        lines=[CountingLine("gate", (100, 0), (100, H))],
        detector=FakeDetector(),
    )
    for _ in range(20):
        pipe.process(np.zeros((H, W, 3), dtype=np.uint8))
    assert pipe.totals()["gate"]["total"] == 1


def test_write_and_read_clip(tmp_path):
    path = str(tmp_path / "clip.mp4")
    write_clip(path, width=160, height=120, n_frames=20)
    import os

    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
