"""Virtual line-crossing counter.

Given tracked objects (each with a current and previous centroid), this module
decides when an object *crosses* a user-defined line segment and in which
direction, then maintains cumulative counts. Direction is derived from the sign
of the 2D cross product of the line vector with the object's movement vector, so
"in" vs "out" is unambiguous and independent of line orientation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from .tracker import Track

Point = Tuple[float, float]


def _cross_sign(a: Point, b: Point, p: Point) -> float:
    """Signed area of triangle (a, b, p).

    Positive if ``p`` is to the left of directed line ``a -> b``, negative if to
    the right, zero if collinear.
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Return True if segment ``p1p2`` intersects segment ``p3p4``."""
    d1 = _cross_sign(p3, p4, p1)
    d2 = _cross_sign(p3, p4, p2)
    d3 = _cross_sign(p1, p2, p3)
    d4 = _cross_sign(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True

    # Collinear-overlap edge cases.
    def on_segment(a: Point, b: Point, c: Point) -> bool:
        return (
            min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        )

    if d1 == 0 and on_segment(p3, p4, p1):
        return True
    if d2 == 0 and on_segment(p3, p4, p2):
        return True
    if d3 == 0 and on_segment(p1, p2, p3):
        return True
    if d4 == 0 and on_segment(p1, p2, p4):
        return True
    return False


@dataclass
class CountingLine:
    """A named, directional counting line segment from ``start`` to ``end``.

    Crossings are split into two directions we label ``in`` and ``out``:
    an object moving from the *right* side of the directed segment to the *left*
    counts as ``in``; the reverse counts as ``out``.
    """

    name: str
    start: Point
    end: Point
    count_in: int = 0
    count_out: int = 0
    # Tracks that have already been counted for this line (avoid double counting).
    _counted: set[int] = field(default_factory=set, repr=False)

    @property
    def total(self) -> int:
        return self.count_in + self.count_out

    @property
    def net(self) -> int:
        return self.count_in - self.count_out

    def side_of(self, point: Point) -> float:
        """Signed side of ``point`` relative to the directed line."""
        return _cross_sign(self.start, self.end, point)


@dataclass
class CrossEvent:
    """Emitted when a track crosses a line."""

    line: str
    track_id: int
    direction: str  # "in" or "out"
    point: Tuple[int, int]


class LineCounter:
    """Maintains a set of counting lines and updates them from tracks."""

    def __init__(self, lines: Iterable[CountingLine] | None = None) -> None:
        self.lines: List[CountingLine] = list(lines) if lines else []

    def add_line(self, line: CountingLine) -> CountingLine:
        self.lines.append(line)
        return line

    def update(self, tracks: Dict[int, Track]) -> List[CrossEvent]:
        """Check every track against every line, return new crossing events."""
        events: List[CrossEvent] = []
        for track in tracks.values():
            prev = track.prev_centroid
            if prev is None:
                continue
            curr = track.centroid
            for line in self.lines:
                if track.id in line._counted:
                    continue
                if not _segments_intersect(prev, curr, line.start, line.end):
                    continue
                # Determine direction from which side the object moved toward.
                side_prev = line.side_of(prev)
                side_curr = line.side_of(curr)
                if side_prev == side_curr:
                    continue  # tangent / grazing; ignore
                direction = "in" if side_prev < 0 else "out"
                if direction == "in":
                    line.count_in += 1
                else:
                    line.count_out += 1
                line._counted.add(track.id)
                events.append(
                    CrossEvent(
                        line=line.name,
                        track_id=track.id,
                        direction=direction,
                        point=(int(curr[0]), int(curr[1])),
                    )
                )
        return events

    def totals(self) -> Dict[str, Dict[str, int]]:
        """Return a serialisable summary of every line's counts."""
        return {
            line.name: {
                "in": line.count_in,
                "out": line.count_out,
                "total": line.total,
                "net": line.net,
            }
            for line in self.lines
        }

    def reset(self) -> None:
        for line in self.lines:
            line.count_in = 0
            line.count_out = 0
            line._counted.clear()
