"""Tests for the motion heatmap: stamping, edge clipping, decay, overlay."""

from __future__ import annotations

import numpy as np
import pytest

from visioncount.heatmap import Heatmap


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        Heatmap(0, 10)
    with pytest.raises(ValueError):
        Heatmap(10, 10, decay=0.0)
    with pytest.raises(ValueError):
        Heatmap(10, 10, decay=1.5)


def test_stamp_accumulates_at_point():
    hm = Heatmap(100, 100, decay=1.0, blob_radius=5)
    hm.add_points([(50, 50)])
    assert hm.buffer[50, 50] > 0.0
    # Far away stays cold.
    assert hm.buffer[0, 0] == 0.0


def test_stamp_clipped_at_frame_edge_does_not_crash_or_overflow():
    hm = Heatmap(40, 40, decay=1.0, blob_radius=8)
    # Points right at / beyond the corners must clip cleanly.
    hm.add_points([(0, 0), (39, 39), (-5, 20), (45, 45)])
    assert hm.buffer.shape == (40, 40)
    assert np.isfinite(hm.buffer).all()
    assert hm.buffer[0, 0] > 0.0  # corner still received part of a stamp


def test_fully_offscreen_point_is_skipped():
    hm = Heatmap(20, 20, decay=1.0, blob_radius=3)
    hm.add_points([(1000, 1000)])
    assert hm.buffer.max() == 0.0


def test_decay_reduces_old_activity():
    hm = Heatmap(30, 30, decay=0.5, blob_radius=3)
    hm.add_points([(15, 15)])
    peak1 = hm.buffer[15, 15]
    # A second update with no new points near the old peak halves it (decay 0.5).
    hm.add_points([(0, 0)])
    assert hm.buffer[15, 15] == pytest.approx(peak1 * 0.5, rel=1e-5)


def test_normalized_is_uint8_in_range():
    hm = Heatmap(30, 30, blob_radius=4)
    hm.add_points([(15, 15)])
    norm = hm.normalized()
    assert norm.dtype == np.uint8
    assert norm.max() <= 255 and norm.min() >= 0
    assert norm.max() == 255  # peak normalises to full scale


def test_empty_heatmap_normalizes_to_zeros():
    hm = Heatmap(10, 10)
    norm = hm.normalized()
    assert norm.max() == 0


def test_colorized_and_overlay_shapes():
    hm = Heatmap(64, 48, blob_radius=4)
    hm.add_points([(32, 24)])
    colorized = hm.colorized()
    assert colorized.shape == (48, 64, 3)

    frame = np.full((48, 64, 3), 30, dtype=np.uint8)
    blended = hm.overlay(frame, alpha=0.5)
    assert blended.shape == frame.shape
    assert blended.dtype == np.uint8


def test_overlay_resizes_when_frame_differs():
    hm = Heatmap(32, 32, blob_radius=4)
    hm.add_points([(16, 16)])
    frame = np.zeros((64, 64, 3), dtype=np.uint8)  # different size
    blended = hm.overlay(frame)
    assert blended.shape == (64, 64, 3)


def test_reset_clears_buffer():
    hm = Heatmap(10, 10, blob_radius=2)
    hm.add_points([(5, 5)])
    hm.reset()
    assert hm.buffer.max() == 0.0
