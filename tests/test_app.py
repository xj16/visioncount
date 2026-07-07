"""Tests for the Flask dashboard (import-safe, no background thread needed)."""

from __future__ import annotations


import pytest


@pytest.fixture()
def client(monkeypatch):
    # Keep the OpenCV worker thread fully off during tests: no eager autostart
    # and no lazy start on request. The endpoints still respond (with empty /
    # placeholder data) without a running worker.
    monkeypatch.setenv("VISIONCOUNT_AUTOSTART", "0")
    monkeypatch.setenv("VISIONCOUNT_LAZY_START", "0")
    from visioncount.app import VideoProcessor, create_app

    processor = VideoProcessor(source="synthetic")
    app = create_app(processor)
    app.config.update(TESTING=True)
    try:
        yield app.test_client()
    finally:
        processor.stop()


def test_app_imports():
    import visioncount.app as m

    assert hasattr(m, "create_app")
    assert hasattr(m, "app")


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_index_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"VisionCount" in resp.data


def test_counts_endpoint_shape(client):
    resp = client.get("/api/counts")
    assert resp.status_code == 200
    data = resp.get_json()
    assert set(data.keys()) == {"lines", "zones", "grand_total", "series", "fps"}


def test_heatmap_endpoint_returns_png(client):
    resp = client.get("/heatmap.png")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
