"""Foreground object detection via OpenCV background subtraction.

This is the *default* detector for VisionCount. It needs no model weights and no
heavy dependencies - just OpenCV and NumPy - so it runs anywhere, including a
Raspberry Pi. It works well for a mostly-static camera where the objects of
interest move against the background (people, cars, packages on a belt, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single detected object in one frame.

    Coordinates are in pixels, ``(x, y)`` is the top-left corner of the
    axis-aligned bounding box.
    """

    x: int
    y: int
    w: int
    h: int
    score: float = 1.0
    label: str = "object"

    @property
    def centroid(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_bbox(self) -> tuple[int, int, int, int]:
        """Return ``(x1, y1, x2, y2)``."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)


class BackgroundSubtractorDetector:
    """Detect moving foreground blobs with MOG2 background subtraction.

    Parameters
    ----------
    history:
        Number of frames used to build the background model.
    var_threshold:
        MOG2 variance threshold; higher = less sensitive.
    detect_shadows:
        Whether MOG2 should mark shadows (they are then removed from the mask).
    min_area:
        Minimum blob area (px^2) to be reported as a detection. Filters noise.
    max_area:
        Optional upper bound on blob area; ``None`` disables it.
    morph_kernel:
        Size of the elliptical kernel used for opening/closing the mask.
    learning_rate:
        Background model update rate passed to ``apply``. ``-1`` lets OpenCV
        choose automatically.
    """

    def __init__(
        self,
        history: int = 300,
        var_threshold: float = 24.0,
        detect_shadows: bool = True,
        min_area: int = 400,
        max_area: int | None = None,
        morph_kernel: int = 5,
        learning_rate: float = -1.0,
    ) -> None:
        if min_area < 0:
            raise ValueError("min_area must be >= 0")
        if max_area is not None and max_area <= min_area:
            raise ValueError("max_area must be greater than min_area")

        self.min_area = min_area
        self.max_area = max_area
        self.learning_rate = learning_rate
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        k = max(1, int(morph_kernel))
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.last_mask: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Return the list of foreground detections in ``frame`` (BGR image)."""
        if frame is None or frame.size == 0:
            return []

        mask = self._subtractor.apply(frame, learningRate=self.learning_rate)

        # MOG2 marks shadows as gray (127); keep only strong foreground (255).
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        # Clean the mask: opening removes speckle, closing fills holes.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        self.last_mask = mask

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            # Score = how "solid" the blob is (fill ratio of its bbox), a cheap
            # proxy for confidence in [0, 1].
            fill = float(area) / float(max(1, w * h))
            detections.append(
                Detection(x=int(x), y=int(y), w=int(w), h=int(h), score=fill)
            )
        return detections

    def reset(self) -> None:
        """Forget the learned background (e.g. after a scene change)."""
        # Re-create with the same tuning by re-applying an empty learning cycle.
        self._subtractor.clear()
        self.last_mask = None
