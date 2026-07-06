"""Tests for the line-crossing geometry and counter."""

from __future__ import annotations

from collections import deque

from visioncount.counter import (
    CountingLine,
    LineCounter,
    _segments_intersect,
)
from visioncount.tracker import Track


def make_track(track_id: int, prev, curr) -> Track:
    t = Track(id=track_id, centroid=curr, bbox=(0, 0, 1, 1))
    t.trajectory = deque([prev, curr], maxlen=32)
    return t


def test_segments_intersect_true():
    assert _segments_intersect((0, 0), (10, 10), (0, 10), (10, 0))


def test_segments_intersect_false():
    assert not _segments_intersect((0, 0), (1, 1), (5, 5), (6, 6))


def test_counts_a_single_crossing():
    line = CountingLine(name="c", start=(50, 0), end=(50, 100))
    counter = LineCounter([line])
    # Move left -> right across the vertical line at x=50.
    track = make_track(1, prev=(40, 50), curr=(60, 50))
    events = counter.update({1: track})
    assert len(events) == 1
    assert line.total == 1


def test_no_double_count_for_same_track():
    line = CountingLine(name="c", start=(50, 0), end=(50, 100))
    counter = LineCounter([line])
    t1 = make_track(1, prev=(40, 50), curr=(60, 50))
    counter.update({1: t1})
    # Same track, another apparent crossing -> must not be counted again.
    t1.trajectory.append((70, 50))
    counter.update({1: t1})
    assert line.total == 1


def test_direction_in_vs_out_is_opposite():
    line = CountingLine(name="c", start=(50, 0), end=(50, 100))
    counter = LineCounter([line])
    left_to_right = make_track(1, prev=(40, 50), curr=(60, 50))
    right_to_left = make_track(2, prev=(60, 20), curr=(40, 20))
    counter.update({1: left_to_right, 2: right_to_left})
    # One should be 'in', the other 'out' -> net cancels to 0.
    assert line.count_in == 1
    assert line.count_out == 1
    assert line.net == 0


def test_totals_serialisable_shape():
    line = CountingLine(name="c", start=(0, 0), end=(10, 0))
    counter = LineCounter([line])
    totals = counter.totals()
    assert totals["c"] == {"in": 0, "out": 0, "total": 0, "net": 0}


def test_reset_clears_counts():
    line = CountingLine(name="c", start=(50, 0), end=(50, 100))
    counter = LineCounter([line])
    counter.update({1: make_track(1, (40, 50), (60, 50))})
    assert line.total == 1
    counter.reset()
    assert line.total == 0
    assert line._counted == set()
