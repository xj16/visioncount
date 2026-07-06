"""Synthetic video generation for tests and demos.

Produces frames of moving blobs on a static background so the whole pipeline can
be exercised deterministically without any camera, video file, or network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import cv2
import numpy as np


@dataclass
class MovingObject:
    """A blob that moves linearly and wraps around the frame."""

    x: float
    y: float
    vx: float
    vy: float
    radius: int
    color: Tuple[int, int, int]

    def step(self, width: int, height: int) -> None:
        self.x += self.vx
        self.y += self.vy
        # Wrap so long clips keep producing crossings.
        if self.x < -self.radius:
            self.x = width + self.radius
        elif self.x > width + self.radius:
            self.x = -self.radius
        if self.y < -self.radius:
            self.y = height + self.radius
        elif self.y > height + self.radius:
            self.y = -self.radius


def default_objects(width: int, height: int) -> List[MovingObject]:
    """A small deterministic set of objects crossing the frame center."""
    cy = height // 2
    return [
        MovingObject(x=-20, y=cy - 40, vx=6.0, vy=0.0, radius=18, color=(60, 200, 90)),
        MovingObject(x=-20, y=cy + 30, vx=4.5, vy=0.0, radius=22, color=(200, 120, 60)),
        MovingObject(
            x=width + 20, y=cy, vx=-5.0, vy=0.0, radius=16, color=(80, 90, 220)
        ),
    ]


def frames(
    width: int = 320,
    height: int = 240,
    n_frames: int = 120,
    objects: List[MovingObject] | None = None,
    noise: float = 3.0,
    seed: int = 0,
) -> Iterator[np.ndarray]:
    """Yield ``n_frames`` BGR frames of moving blobs on a static background."""
    rng = np.random.default_rng(seed)
    if objects is None:
        objects = default_objects(width, height)

    # A mildly textured static background so background subtraction has to work.
    base = np.full((height, width, 3), 40, dtype=np.uint8)
    base += (rng.random((height, width, 3)) * 15).astype(np.uint8)

    for _ in range(n_frames):
        frame = base.copy()
        if noise > 0:
            n = (rng.standard_normal((height, width, 3)) * noise).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + n, 0, 255).astype(np.uint8)
        for obj in objects:
            cv2.circle(
                frame,
                (int(obj.x), int(obj.y)),
                obj.radius,
                obj.color,
                thickness=-1,
            )
            obj.step(width, height)
        yield frame


def write_clip(path: str, width: int = 320, height: int = 240, n_frames: int = 120) -> str:
    """Write a synthetic clip to ``path`` (MP4). Returns the path."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 25.0, (width, height))
    try:
        for frame in frames(width=width, height=height, n_frames=n_frames):
            writer.write(frame)
    finally:
        writer.release()
    return path
