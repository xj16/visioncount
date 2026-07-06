"""Tests for the Flask dashboard (import-safe, no background thread needed)."""

from __future__ import annotations


import pytest


@pytest.fixture()
def client(monkeypatch):
    # Do not autostart the worker thread during tests.
    monkeypatch.setenv("VISIONCOUNT_AUTOSTART", "0")
    from visioncount.app import VideoProcessor, create_app

    app = create_app(VideoProcessor(source="synthetic"))
    app.config.update(TESTING=True)
    return app.test_client()


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
    assert set(data.keys()) == {"lines", "grand_total", "series", "fps"}


def test_heatmap_endpoint_returns_png(client):
    resp = client.get("/heatmap.png")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
