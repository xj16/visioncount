"""Motion / occupancy heatmap accumulation.

Accumulates where objects appear over time into a float32 buffer, with optional
exponential decay so the heatmap reflects *recent* activity rather than the
entire history. Rendering to a color overlay uses an OpenCV colormap.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np



class Heatmap:
    """Accumulates a 2D activity map at (a downscaled) frame resolution.

    Parameters
    ----------
    width, height:
        Resolution of the accumulation buffer (usually the frame size).
    decay:
        Per-update multiplicative decay in ``(0, 1]``. ``1.0`` keeps the full
        history; ``0.97`` emphasises recent motion.
    blob_radius:
        Radius (px) of the gaussian-ish stamp added at each object centroid.
    """

    def __init__(
        self,
        width: int,
        height: int,
        decay: float = 0.99,
        blob_radius: int = 16,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if not (0.0 < decay <= 1.0):
            raise ValueError("decay must be in (0, 1]")
        self.width = int(width)
        self.height = int(height)
        self.decay = float(decay)
        self.blob_radius = int(blob_radius)
        self.buffer = np.zeros((self.height, self.width), dtype=np.float32)
        self._stamp = self._make_stamp(self.blob_radius)

    @staticmethod
    def _make_stamp(radius: int) -> np.ndarray:
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        dist2 = xx ** 2 + yy ** 2
        sigma = max(1.0, radius / 2.0)
        stamp = np.exp(-dist2 / (2.0 * sigma ** 2)).astype(np.float32)
        return stamp

    def add_points(self, points: Iterable[Tuple[int, int]]) -> None:
        """Stamp a soft blob at each ``(x, y)`` point."""
        if self.decay < 1.0:
            self.buffer *= self.decay
        r = self.blob_radius
        for x, y in points:
            x, y = int(x), int(y)
            x0, x1 = x - r, x + r + 1
            y0, y1 = y - r, y + r + 1
            # Clip the stamp to the buffer bounds.
            sx0 = max(0, -x0)
            sy0 = max(0, -y0)
            bx0, by0 = max(0, x0), max(0, y0)
            bx1, by1 = min(self.width, x1), min(self.height, y1)
            if bx1 <= bx0 or by1 <= by0:
                continue
            sx1 = sx0 + (bx1 - bx0)
            sy1 = sy0 + (by1 - by0)
            self.buffer[by0:by1, bx0:bx1] += self._stamp[sy0:sy1, sx0:sx1]

    def update_from_tracks(self, tracks) -> None:
        """Convenience: stamp every track's current centroid."""
        pts = [t.centroid for t in tracks.values()]
        self.add_points(pts)

    def normalized(self) -> np.ndarray:
        """Return the buffer scaled to ``uint8`` in ``[0, 255]``."""
        buf = self.buffer
        peak = float(buf.max())
        if peak <= 1e-6:
            return np.zeros_like(buf, dtype=np.uint8)
        norm = np.clip(buf / peak, 0.0, 1.0)
        return (norm * 255.0).astype(np.uint8)

    def colorized(self, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Return a BGR color image of the heatmap."""
        return cv2.applyColorMap(self.normalized(), colormap)

    def overlay(self, frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Blend the colorized heatmap over ``frame`` (resized to match)."""
        heat = self.colorized()
        if heat.shape[:2] != frame.shape[:2]:
            heat = cv2.resize(heat, (frame.shape[1], frame.shape[0]))
        # Only blend where there is meaningful activity to avoid washing out.
        mask = (self.normalized() > 12).astype(np.float32)
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask3 = np.dstack([mask] * 3)
        blended = frame.astype(np.float32) * (1 - alpha * mask3) + heat.astype(
            np.float32
        ) * (alpha * mask3)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def reset(self) -> None:
        self.buffer.fill(0.0)
