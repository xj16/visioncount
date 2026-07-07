"""Lightweight centroid tracker.

Associates detections across frames by nearest-centroid matching, assigns a
stable integer ID to each object, and remembers a short trajectory so the line
counter can tell which side of a line an object came from.

This is a pure NumPy implementation - no scipy, no filterpy - so it stays cheap
enough for a Raspberry Pi while remaining accurate for well-separated objects.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Sequence, Tuple

import numpy as np

from .detector import Detection


@dataclass
class Track:
    """A tracked object with a stable id and recent trajectory."""

    id: int
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    missing: int = 0
    trajectory: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=32))
    # Estimated constant-velocity (px/frame), updated on each match.
    velocity: Tuple[float, float] = (0.0, 0.0)

    @property
    def prev_centroid(self) -> Tuple[int, int] | None:
        if len(self.trajectory) >= 2:
            return self.trajectory[-2]
        return None

    def predicted_centroid(self) -> Tuple[float, float]:
        """Where the object is expected next, from its constant velocity.

        Matching against the *prediction* rather than the last seen position
        keeps IDs stable when objects cross paths or briefly occlude.
        """
        return (self.centroid[0] + self.velocity[0], self.centroid[1] + self.velocity[1])


class CentroidTracker:
    """Greedy nearest-neighbour centroid tracker.

    Parameters
    ----------
    max_disappeared:
        Number of consecutive frames a track may go unmatched before it is
        dropped.
    max_distance:
        Maximum pixel distance allowed when matching a detection to an existing
        track. Prevents teleporting IDs across the frame.
    """

    def __init__(
        self,
        max_disappeared: int = 20,
        max_distance: float = 80.0,
        vel_smooth: float = 0.5,
    ) -> None:
        if max_disappeared < 0:
            raise ValueError("max_disappeared must be >= 0")
        if max_distance <= 0:
            raise ValueError("max_distance must be > 0")
        if not (0.0 <= vel_smooth < 1.0):
            raise ValueError("vel_smooth must be in [0, 1)")
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        # Smoothing factor for the velocity estimate: higher = steadier, slower
        # to react. 0.5 balances responsiveness against jitter for CPU tracking.
        self._vel_smooth = vel_smooth
        self._next_id = 0
        self.tracks: "OrderedDict[int, Track]" = OrderedDict()

    def _register(self, det: Detection) -> None:
        track = Track(id=self._next_id, centroid=det.centroid, bbox=det.as_bbox())
        track.trajectory.append(det.centroid)
        self.tracks[self._next_id] = track
        self._next_id += 1

    def _deregister(self, track_id: int) -> None:
        del self.tracks[track_id]

    def update(self, detections: Sequence[Detection]) -> Dict[int, Track]:
        """Advance the tracker by one frame and return the active tracks."""
        # No detections: age every track, drop the stale ones.
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].missing += 1
                if self.tracks[track_id].missing > self.max_disappeared:
                    self._deregister(track_id)
            return dict(self.tracks)

        input_centroids = np.array(
            [d.centroid for d in detections], dtype=np.float64
        )

        # No existing tracks: register everything.
        if len(self.tracks) == 0:
            for det in detections:
                self._register(det)
            return dict(self.tracks)

        track_ids = list(self.tracks.keys())
        # Match against each track's *predicted* next position, not its last one,
        # so objects that keep moving during a brief miss stay associated.
        track_centroids = np.array(
            [self.tracks[tid].predicted_centroid() for tid in track_ids],
            dtype=np.float64,
        )

        # Distance matrix: rows = tracks, cols = detections.
        diff = track_centroids[:, np.newaxis, :] - input_centroids[np.newaxis, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))

        # Greedy matching: repeatedly take the globally smallest distance.
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        # Order candidate (row, col) pairs by ascending distance.
        pairs = np.dstack(np.unravel_index(np.argsort(dist, axis=None), dist.shape))[0]
        for row, col in pairs:
            row, col = int(row), int(col)
            if row in used_rows or col in used_cols:
                continue
            if dist[row, col] > self.max_distance:
                continue
            track = self.tracks[track_ids[row]]
            det = detections[col]
            new_centroid = det.centroid
            # Exponentially-smoothed velocity estimate (px/frame).
            vx = new_centroid[0] - track.centroid[0]
            vy = new_centroid[1] - track.centroid[1]
            track.velocity = (
                self._vel_smooth * track.velocity[0] + (1 - self._vel_smooth) * vx,
                self._vel_smooth * track.velocity[1] + (1 - self._vel_smooth) * vy,
            )
            track.centroid = new_centroid
            track.bbox = det.as_bbox()
            track.missing = 0
            track.trajectory.append(new_centroid)
            used_rows.add(row)
            used_cols.add(col)

        # Unmatched existing tracks -> age them.
        for row, tid in enumerate(track_ids):
            if row not in used_rows:
                self.tracks[tid].missing += 1
                if self.tracks[tid].missing > self.max_disappeared:
                    self._deregister(tid)

        # Unmatched detections -> new tracks.
        for col, det in enumerate(detections):
            if col not in used_cols:
                self._register(det)

        return dict(self.tracks)

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 0
