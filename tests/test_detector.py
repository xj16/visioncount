"""Tests for the MOG2 background-subtraction detector."""

from __future__ import annotations

import numpy as np
import pytest

from visioncount.detector import BackgroundSubtractorDetector, Detection


def test_detection_geometry():
    d = Detection(x=10, y=20, w=30, h=40)
    assert d.centroid == (10 + 15, 20 + 20)
    assert d.area == 30 * 40
    assert d.as_bbox() == (10, 20, 40, 60)


def test_rejects_bad_area_bounds():
    with pytest.raises(ValueError):
        BackgroundSubtractorDetector(min_area=-1)
    with pytest.raises(ValueError):
        BackgroundSubtractorDetector(min_area=100, max_area=50)


def test_empty_frame_guard_returns_no_detections():
    det = BackgroundSubtractorDetector()
    assert det.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []
    assert det.detect(None) == []  # type: ignore[arg-type]


def test_detects_a_bright_moving_square():
    det = BackgroundSubtractorDetector(min_area=50, detect_shadows=False)
    bg = np.zeros((120, 160, 3), dtype=np.uint8)
    # Prime the background model with several static frames.
    for _ in range(10):
        det.detect(bg)
    # A bright square appears where there was none: it must register as
    # foreground on the frames right after it appears (a static object is
    # eventually absorbed into the background model, which is correct MOG2
    # behaviour, so we assert detection *ever* fired, not on the last frame).
    frame = bg.copy()
    frame[40:80, 60:100] = 255
    found = []
    for _ in range(5):
        found.extend(det.detect(frame))
    assert any(d.area >= 50 for d in found)
    # The detected box should roughly cover the square.
    assert any(d.x <= 65 and d.y <= 45 and d.w >= 30 for d in found)


def test_min_area_filters_small_blobs():
    big = BackgroundSubtractorDetector(min_area=5, detect_shadows=False)
    strict = BackgroundSubtractorDetector(min_area=100000, detect_shadows=False)
    bg = np.zeros((120, 160, 3), dtype=np.uint8)
    for _ in range(8):
        big.detect(bg)
        strict.detect(bg)
    frame = bg.copy()
    frame[50:70, 70:90] = 255
    big_all, strict_all = [], []
    for _ in range(4):
        big_all.extend(big.detect(frame))
        strict_all.extend(strict.detect(frame))
    # The lenient threshold sees the blob; the huge one filters everything out.
    assert len(big_all) >= 1
    assert len(strict_all) == 0


def test_reset_clears_last_mask():
    det = BackgroundSubtractorDetector()
    det.detect(np.zeros((40, 40, 3), dtype=np.uint8))
    det.reset()
    assert det.last_mask is None
