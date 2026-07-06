"""Optional YOLO (ultralytics) detector.

This module is intentionally import-safe: importing it never pulls in
``ultralytics`` or ``torch``. Those heavy dependencies are imported lazily only
when :class:`YoloDetector` is actually instantiated. If they are missing, a
clear :class:`YoloUnavailable` error is raised, and the rest of VisionCount keeps
working on the default background-subtraction detector.

Usage::

    from visioncount.yolo_detector import YoloDetector, yolo_available
    if yolo_available():
        det = YoloDetector(weights="yolov8n.pt", classes=[0])  # person only
"""

from __future__ import annotations

from typing import List, Sequence

from .detector import Detection


class YoloUnavailable(RuntimeError):
    """Raised when the optional YOLO stack is not installed."""


def yolo_available() -> bool:
    """Return True if ``ultralytics`` can be imported."""
    try:
        import importlib.util

        return importlib.util.find_spec("ultralytics") is not None
    except Exception:  # pragma: no cover - extremely defensive
        return False


class YoloDetector:
    """Detect objects using an ultralytics YOLO model.

    Parameters
    ----------
    weights:
        Path or model name understood by ultralytics (e.g. ``"yolov8n.pt"``).
    conf:
        Minimum confidence threshold.
    classes:
        Optional list of COCO class indices to keep (e.g. ``[0]`` for person).
    device:
        Torch device string (``"cpu"``, ``"cuda:0"``); ``None`` = auto.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.35,
        classes: Sequence[int] | None = None,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO  # noqa: WPS433 (lazy import by design)
        except Exception as exc:  # pragma: no cover - depends on environment
            raise YoloUnavailable(
                "ultralytics is not installed. Install the optional extra with "
                "`pip install visioncount[yolo]` (this pulls in torch), or use the "
                "default BackgroundSubtractorDetector."
            ) from exc

        self.conf = conf
        self.classes = list(classes) if classes is not None else None
        self.device = device
        self._model = YOLO(weights)
        # Human-readable class names, if the model exposes them.
        self._names = getattr(self._model, "names", {}) or {}

    def detect(self, frame) -> List[Detection]:
        """Run inference on a BGR frame and return VisionCount detections."""
        results = self._model.predict(
            frame,
            conf=self.conf,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        detections: List[Detection] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = (int(v) for v in xyxy)
                score = float(box.conf[0]) if box.conf is not None else 1.0
                cls_idx = int(box.cls[0]) if box.cls is not None else -1
                label = self._names.get(cls_idx, str(cls_idx))
                detections.append(
                    Detection(
                        x=x1,
                        y=y1,
                        w=max(1, x2 - x1),
                        h=max(1, y2 - y1),
                        score=score,
                        label=label,
                    )
                )
        return detections
