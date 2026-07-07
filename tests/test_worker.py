"""Flagship tests for the live Flask worker -- the hardest, most-advertised part.

Unlike ``test_app.py`` (which keeps the OpenCV thread off), these tests actually
*start* :class:`VideoProcessor` against the synthetic source and exercise the real
concurrent behaviour the README sells: a running MJPEG stream, advancing counts,
event persistence, live line/zone editing, error surfacing on a bad source, and
the security guards.
"""

from __future__ import annotations

import time

import pytest

from visioncount.app import RateLimiter, VideoProcessor, create_app


def _wait(pred, timeout=10.0, interval=0.05):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(interval)
    return False


@pytest.fixture()
def running(monkeypatch):
    """A VideoProcessor actually running the synthetic pipeline."""
    monkeypatch.setenv("VISIONCOUNT_AUTOSTART", "0")
    monkeypatch.setenv("VISIONCOUNT_LAZY_START", "0")
    proc = VideoProcessor(source="synthetic", width=320, height=240)
    proc.start()
    assert _wait(lambda: proc.get_jpeg() is not None), "worker never produced a frame"
    try:
        yield proc
    finally:
        proc.stop()


def test_worker_produces_valid_jpeg(running):
    jpeg = running.get_jpeg()
    assert jpeg is not None
    # JPEG SOI / EOI markers.
    assert jpeg[:2] == b"\xff\xd8"
    assert jpeg[-2:] == b"\xff\xd9"


def test_health_reports_ok_and_frames_advance(running):
    assert _wait(lambda: running.health()["frames_seen"] > 3)
    h = running.health()
    assert h["status"] in {"ok", "starting"}
    assert h["error"] is None


def test_counts_advance_and_events_recorded(running):
    # The synthetic objects sweep across the default center line.
    assert _wait(lambda: running.get_counts()["grand_total"] >= 1, timeout=15.0)
    assert _wait(lambda: running.store.count() >= 1, timeout=15.0)
    recent = running.store.recent(5)
    assert recent and recent[0]["direction"] in {"in", "out"}


def test_video_feed_stream_yields_jpeg_boundary(running):
    app = create_app(running)
    client = app.test_client()
    resp = client.get("/video_feed")
    assert resp.status_code == 200
    assert "multipart/x-mixed-replace" in resp.content_type
    # Pull a couple of chunks off the generator and confirm a real JPEG rides in.
    it = resp.response
    blob = b""
    for _ in range(4):
        try:
            blob += next(it)
        except StopIteration:  # pragma: no cover
            break
        if b"\xff\xd8" in blob:
            break
    assert b"--frame" in blob
    assert b"Content-Type: image/jpeg" in blob
    assert b"\xff\xd8" in blob  # an actual JPEG payload was streamed
    it.close()


def test_bad_source_surfaces_as_degraded_and_503():
    proc = VideoProcessor(source="/no/such/file.mp4", loop=False)
    proc.start()
    assert _wait(lambda: proc.health()["status"] == "degraded")
    h = proc.health()
    assert "cannot open source" in (h["error"] or "")
    # /video_feed must fail fast rather than hang forever on a dead worker.
    app = create_app(proc)
    resp = app.test_client().get("/video_feed")
    assert resp.status_code == 503
    proc.stop()


def test_live_line_add_and_remove(running):
    app = create_app(running)
    client = app.test_client()
    before = len(client.get("/api/lines").get_json()["lines"])
    r = client.post("/api/lines", json={"name": "gate", "start": [5, 5], "end": [5, 200]})
    assert r.status_code == 201
    lines = client.get("/api/lines").get_json()["lines"]
    assert len(lines) == before + 1
    assert any(ln["name"] == "gate" for ln in lines)
    r = client.delete("/api/lines?name=gate")
    assert r.status_code == 200 and r.get_json()["removed"] is True


def test_live_zone_add_shows_up_in_counts(running):
    app = create_app(running)
    client = app.test_client()
    r = client.post(
        "/api/zones",
        json={"name": "roi", "polygon": [[40, 40], [200, 40], [200, 160], [40, 160]]},
    )
    assert r.status_code == 201
    assert _wait(lambda: "roi" in client.get("/api/counts").get_json()["zones"])


def test_export_csv_streams_recorded_events(running):
    assert _wait(lambda: running.store.count() >= 1, timeout=15.0)
    app = create_app(running)
    body = app.test_client().get("/export.csv").get_data(as_text=True)
    lines = body.strip().splitlines()
    assert lines[0] == "seq,ts,iso,line,direction,track_id,x,y"
    assert len(lines) >= 2


# ---- security guards ------------------------------------------------------


def test_rate_limiter_blocks_after_budget():
    rl = RateLimiter(per_minute=3)
    assert [rl.allow("a") for _ in range(4)] == [True, True, True, False]
    # A different client has its own budget.
    assert rl.allow("b") is True


def test_api_key_required_when_configured(monkeypatch):
    monkeypatch.setenv("VISIONCOUNT_AUTOSTART", "0")
    monkeypatch.setenv("VISIONCOUNT_LAZY_START", "0")
    monkeypatch.setenv("VISIONCOUNT_API_KEY", "s3cret")
    proc = VideoProcessor(source="synthetic")
    app = create_app(proc)
    client = app.test_client()
    # No key -> 401 on a mutating endpoint.
    r = client.post("/api/lines", json={"name": "x", "start": [0, 0], "end": [0, 10]})
    assert r.status_code == 401
    # Correct key -> allowed.
    r = client.post(
        "/api/lines",
        json={"name": "x", "start": [0, 0], "end": [0, 10]},
        headers={"X-API-Key": "s3cret"},
    )
    assert r.status_code == 201
    # Reads stay open.
    assert client.get("/api/counts").status_code == 200
    proc.stop()


def test_no_cors_header_by_default(monkeypatch):
    monkeypatch.setenv("VISIONCOUNT_AUTOSTART", "0")
    monkeypatch.setenv("VISIONCOUNT_LAZY_START", "0")
    monkeypatch.delenv("VISIONCOUNT_CORS_ORIGIN", raising=False)
    proc = VideoProcessor(source="synthetic")
    client = create_app(proc).test_client()
    resp = client.get("/api/counts")
    assert "Access-Control-Allow-Origin" not in resp.headers
    proc.stop()


def test_cors_header_when_opted_in(monkeypatch):
    monkeypatch.setenv("VISIONCOUNT_AUTOSTART", "0")
    monkeypatch.setenv("VISIONCOUNT_LAZY_START", "0")
    monkeypatch.setenv("VISIONCOUNT_CORS_ORIGIN", "https://example.com")
    proc = VideoProcessor(source="synthetic")
    client = create_app(proc).test_client()
    resp = client.get("/api/counts")
    assert resp.headers.get("Access-Control-Allow-Origin") == "https://example.com"
    proc.stop()


def test_invalid_geometry_rejected():
    proc = VideoProcessor(source="synthetic")
    client = create_app(proc).test_client()
    # missing name
    assert client.post("/api/lines", json={"start": [0, 0], "end": [1, 1]}).status_code == 400
    # bad point
    assert client.post(
        "/api/lines", json={"name": "n", "start": [0], "end": [1, 1]}
    ).status_code == 400
    # zone with too few points
    assert client.post(
        "/api/zones", json={"name": "z", "polygon": [[0, 0], [1, 1]]}
    ).status_code == 400
    proc.stop()
