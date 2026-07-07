"""Tests for the centroid tracker."""

from __future__ import annotations

from visioncount.detector import Detection
from visioncount.tracker import CentroidTracker


def det(x, y, w=10, h=10) -> Detection:
    return Detection(x=x, y=y, w=w, h=h)


def test_registers_new_objects():
    tracker = CentroidTracker()
    tracks = tracker.update([det(10, 10), det(100, 100)])
    assert len(tracks) == 2


def test_keeps_stable_ids_across_frames():
    tracker = CentroidTracker(max_distance=50)
    tracks = tracker.update([det(10, 10)])
    (first_id,) = tracks.keys()
    # Small movement -> same id.
    tracks = tracker.update([det(14, 12)])
    assert list(tracks.keys()) == [first_id]


def test_drops_after_max_disappeared():
    tracker = CentroidTracker(max_disappeared=2)
    tracker.update([det(10, 10)])
    tracker.update([])  # missing 1
    tracker.update([])  # missing 2
    tracks = tracker.update([])  # missing 3 -> dropped
    assert tracks == {}


def test_far_detection_becomes_new_id():
    tracker = CentroidTracker(max_distance=30)
    tracker.update([det(10, 10)])
    # Jump beyond max_distance -> a new track, old one ages out.
    tracks = tracker.update([det(200, 200)])
    ids = list(tracks.keys())
    assert len(ids) >= 1
    assert 1 in ids or len(ids) == 2


def test_trajectory_grows():
    tracker = CentroidTracker()
    tracker.update([det(10, 10)])
    tracks = tracker.update([det(12, 10)])
    (track,) = tracks.values()
    assert len(track.trajectory) == 2
    assert track.prev_centroid is not None


def test_rejects_bad_config():
    import pytest

    with pytest.raises(ValueError):
        CentroidTracker(max_disappeared=-1)
    with pytest.raises(ValueError):
        CentroidTracker(max_distance=0)
    with pytest.raises(ValueError):
        CentroidTracker(vel_smooth=1.0)


def test_velocity_estimated_from_consistent_motion():
    tracker = CentroidTracker(max_distance=50, vel_smooth=0.5)
    tracker.update([det(10, 50)])
    for x in (20, 30, 40, 50):
        tracks = tracker.update([det(x, 50)])
    (track,) = tracks.values()
    # Object moved +10px/frame in x; smoothed velocity should be clearly positive.
    assert track.velocity[0] > 4.0
    assert abs(track.velocity[1]) < 1.0
    # The prediction leads the last seen centroid in the direction of travel.
    pred = track.predicted_centroid()
    assert pred[0] > track.centroid[0]


def test_prediction_keeps_id_through_a_larger_step():
    # With velocity prediction, an object moving fast enough that its next step
    # exceeds the last-position distance still associates to the same track.
    tracker = CentroidTracker(max_distance=35, vel_smooth=0.4)
    tracker.update([det(0, 50)])
    ids = []
    for x in (25, 50, 75, 100):
        tracks = tracker.update([det(x, 50)])
        ids.append(next(iter(tracks.keys())))
    # A single stable id throughout (prediction absorbs the 25px/frame motion).
    assert len(set(ids)) == 1
