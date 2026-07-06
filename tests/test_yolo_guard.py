"""The optional YOLO path must be import-safe even without torch/ultralytics."""

from __future__ import annotations

import pytest


def test_yolo_module_imports_without_ultralytics():
    # Importing the module must never require the heavy stack.
    import visioncount.yolo_detector as y

    assert hasattr(y, "YoloDetector")
    assert hasattr(y, "yolo_available")


def test_yolo_available_returns_bool():
    from visioncount.yolo_detector import yolo_available

    assert isinstance(yolo_available(), bool)


def test_instantiating_without_ultralytics_raises_clear_error():
    from visioncount.yolo_detector import (
        YoloDetector,
        YoloUnavailable,
        yolo_available,
    )

    if yolo_available():
        pytest.skip("ultralytics is installed; guard path not exercised")

    with pytest.raises(YoloUnavailable):
        YoloDetector(weights="yolov8n.pt")
