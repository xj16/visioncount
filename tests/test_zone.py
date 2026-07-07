"""Tests for polygon zone occupancy, dwell and point-in-polygon geometry."""

from __future__ import annotations

from collections import deque

from visioncount.tracker import Track
from visioncount.zone import Zone, ZoneCounter, point_in_polygon

SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def make_track(track_id: int, centroid) -> Track:
    t = Track(id=track_id, centroid=centroid, bbox=(0, 0, 1, 1))
    t.trajectory = deque([centroid], maxlen=32)
    return t


def test_point_inside_and_outside():
    assert point_in_polygon((50, 50), SQUARE)
    assert not point_in_polygon((150, 50), SQUARE)
    assert not point_in_polygon((-10, 50), SQUARE)


def test_degenerate_polygon_is_never_inside():
    assert not point_in_polygon((0, 0), [(0, 0), (1, 1)])  # < 3 points


def test_concave_polygon():
    # An arrow/chevron notch; a point in the notch must read as outside.
    chevron = [(0, 0), (100, 0), (50, 40), (100, 100), (0, 100)]
    assert point_in_polygon((10, 50), chevron)
    assert not point_in_polygon((90, 50), chevron)  # inside the notch -> outside


def test_zone_occupancy_counts_tracks_inside():
    zone = Zone("z", SQUARE)
    tracks = {
        1: make_track(1, (50, 50)),   # inside
        2: make_track(2, (200, 200)),  # outside
        3: make_track(3, (10, 90)),   # inside
    }
    occ = zone.update(tracks, frame_index=0)
    assert occ == 2
    assert zone.occupancy == 2
    assert zone.peak_occupancy == 2


def test_zone_entries_and_dwell():
    zone = Zone("z", SQUARE)
    t = make_track(1, (50, 50))
    zone.update({1: t}, frame_index=0)
    t.centroid = (60, 60)
    zone.update({1: t}, frame_index=5)
    assert zone.total_entries == 1  # same track, still one entry
    assert zone.max_dwell == 5  # 5 frames since entry

    # Track leaves, then re-enters -> a second entry, dwell resets.
    zone.update({}, frame_index=6)
    assert zone.occupancy == 0
    zone.update({1: make_track(1, (50, 50))}, frame_index=7)
    assert zone.total_entries == 2
    assert zone.max_dwell == 0


def test_peak_occupancy_is_high_water_mark():
    zone = Zone("z", SQUARE)
    zone.update({1: make_track(1, (10, 10)), 2: make_track(2, (20, 20))}, 0)
    assert zone.peak_occupancy == 2
    zone.update({1: make_track(1, (10, 10))}, 1)  # one leaves
    assert zone.occupancy == 1
    assert zone.peak_occupancy == 2  # peak stays


def test_zone_counter_updates_all_and_serialises():
    zc = ZoneCounter([Zone("a", SQUARE), Zone("b", [(200, 200), (300, 200), (300, 300)])])
    zc.update({1: make_track(1, (50, 50))}, 0)
    totals = zc.totals()
    assert totals["a"]["occupancy"] == 1
    assert totals["b"]["occupancy"] == 0
    assert set(totals["a"].keys()) >= {"occupancy", "peak", "entries", "max_dwell", "polygon"}


def test_zone_reset_clears_state():
    zone = Zone("z", SQUARE)
    zone.update({1: make_track(1, (50, 50))}, 0)
    zone.reset()
    assert zone.occupancy == 0
    assert zone.peak_occupancy == 0
    assert zone.total_entries == 0
