"""Polygon zone (ROI) occupancy + dwell tracking.

Line crossing answers *"how many passed this point?"*. A zone answers the other
question every real counting deployment asks: *"how many are inside this region
right now, and how long do they stay?"*.

A :class:`Zone` is a closed polygon. On every frame it is updated with the set of
active tracks; it reports current occupancy (how many track centroids fall inside
the polygon) and per-track dwell time. Point-in-polygon uses the standard
ray-casting test -- pure Python, no dependencies -- so it is cheap and trivially
unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Tuple

Point = Tuple[float, float]


def point_in_polygon(point: Point, polygon: List[Point]) -> bool:
    """Ray-casting point-in-polygon test.

    Returns True if ``point`` lies inside ``polygon`` (a list of >= 3 vertices).
    Points exactly on an edge are treated as inside for the horizontal-ray case,
    which is fine for occupancy counting where a pixel of slop is irrelevant.
    """
    n = len(polygon)
    if n < 3:
        return False
    x, y = point
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Does the horizontal ray from (x, y) cross edge (i, j)?
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


@dataclass
class Zone:
    """A named polygonal region of interest that tracks occupancy and dwell.

    Attributes
    ----------
    name:
        Human-readable label shown on the dashboard.
    polygon:
        Ordered list of ``(x, y)`` vertices (>= 3). Implicitly closed.
    peak_occupancy:
        Highest simultaneous occupancy seen so far.
    total_entries:
        Cumulative number of distinct track *entries* into the zone.
    """

    name: str
    polygon: List[Point]
    peak_occupancy: int = 0
    total_entries: int = 0
    # track_id -> frame_index of first entry (for dwell in frames).
    _entered_at: Dict[int, int] = field(default_factory=dict, repr=False)
    _inside: set[int] = field(default_factory=set, repr=False)
    _current: int = field(default=0, repr=False)
    _dwell: Dict[int, int] = field(default_factory=dict, repr=False)

    def contains(self, point: Point) -> bool:
        return point_in_polygon(point, self.polygon)

    def update(self, tracks: Mapping[int, Any], frame_index: int) -> int:
        """Refresh occupancy from the current tracks; return live occupancy.

        ``tracks`` is the pipeline's ``{id: Track}`` mapping; only ``.id`` and
        ``.centroid`` are used, so any object exposing those works (keeps this
        module import-free of the tracker).
        """
        now_inside: set[int] = set()
        for tid, track in tracks.items():
            centroid = getattr(track, "centroid")
            if self.contains(centroid):
                now_inside.add(tid)
                if tid not in self._inside:
                    # Fresh entry.
                    self.total_entries += 1
                    self._entered_at[tid] = frame_index
                # Dwell in frames since entry.
                self._dwell[tid] = frame_index - self._entered_at.get(tid, frame_index)

        # Clean up dwell/entry bookkeeping for tracks that left.
        left = self._inside - now_inside
        for tid in left:
            self._entered_at.pop(tid, None)
            self._dwell.pop(tid, None)

        self._inside = now_inside
        self._current = len(now_inside)
        self.peak_occupancy = max(self.peak_occupancy, self._current)
        return self._current

    @property
    def occupancy(self) -> int:
        return self._current

    @property
    def max_dwell(self) -> int:
        """Longest current dwell, in frames (0 if empty)."""
        return max(self._dwell.values(), default=0)

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "occupancy": self._current,
            "peak": self.peak_occupancy,
            "entries": self.total_entries,
            "max_dwell": self.max_dwell,
            "polygon": [[int(px), int(py)] for px, py in self.polygon],
        }

    def reset(self) -> None:
        self.peak_occupancy = 0
        self.total_entries = 0
        self._entered_at.clear()
        self._inside.clear()
        self._dwell.clear()
        self._current = 0


class ZoneCounter:
    """Maintains a set of zones and updates them together each frame."""

    def __init__(self, zones: Iterable[Zone] | None = None) -> None:
        self.zones: List[Zone] = list(zones) if zones else []

    def add_zone(self, zone: Zone) -> Zone:
        self.zones.append(zone)
        return zone

    def update(self, tracks: Mapping[int, Any], frame_index: int) -> None:
        for zone in self.zones:
            zone.update(tracks, frame_index)

    def totals(self) -> Dict[str, Dict[str, object]]:
        return {z.name: z.as_dict() for z in self.zones}

    def reset(self) -> None:
        for zone in self.zones:
            zone.reset()
