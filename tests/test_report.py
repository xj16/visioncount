"""Tests for the Matplotlib analytics report."""

from __future__ import annotations

import os

from visioncount.report import render_report


def test_render_report_creates_png(tmp_path):
    totals = {
        "door": {"in": 5, "out": 3, "total": 8, "net": 2},
        "aisle": {"in": 1, "out": 4, "total": 5, "net": -3},
    }
    series = [(1000.0, 0), (1001.0, 4), (1002.0, 9), (1003.0, 13)]
    path = str(tmp_path / "report.png")
    out = render_report(totals, series, path, title="Test")
    assert out == path
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_render_report_handles_empty(tmp_path):
    path = str(tmp_path / "empty.png")
    render_report({}, [], path)
    assert os.path.exists(path)
