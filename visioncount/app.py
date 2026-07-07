"""Flask dashboard for VisionCount.

Runs the pipeline in a background thread and serves a live dashboard plus a small
JSON/CSV analytics API:

* ``/``                  - HTML dashboard (live video, counts, zones, heatmap,
                           interactive line/zone drawing)
* ``/video_feed``        - MJPEG stream of annotated frames
* ``/heatmap.png``       - current motion heatmap as a PNG
* ``/api/counts``        - line counts + rolling time series + zone occupancy
* ``/api/events``        - recent individual crossings (JSON)
* ``/export.csv``        - streamed CSV of every crossing recorded this run
* ``/api/lines``         - GET current lines; POST add; DELETE remove (live)
* ``/api/zones``         - GET current zones; POST add a polygon zone (live)
* ``/api/health``        - liveness/health probe (reports degraded/stale states)

The video source defaults to the built-in synthetic clip so the dashboard works
with zero configuration (great for demos and CI). Point it at a webcam or video
file with ``VISIONCOUNT_SOURCE``.

Security (all opt-in, sensible defaults):
    * A tiny in-process token-bucket rate limiter guards the mutating and
      export endpoints (``VISIONCOUNT_RATE_LIMIT`` req/min, default 120).
    * An optional API key (``VISIONCOUNT_API_KEY``) is required on mutating
      endpoints when set (via ``X-API-Key`` or ``?api_key=``).
    * CORS is locked down: no ``Access-Control-Allow-Origin`` unless you set
      ``VISIONCOUNT_CORS_ORIGIN`` explicitly.
    * All geometry input is validated and clamped to the frame.

Run::

    python -m visioncount.app
    # or
    flask --app visioncount.app run
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template_string,
    request,
)

from .counter import CountingLine
from .events import EventStore
from .pipeline import Pipeline, PipelineConfig
from .synthetic import frames as synthetic_frames
from .templates import INDEX_HTML
from .zone import Zone

# Seconds without a fresh frame before the feed is considered "stale".
STALE_AFTER = 5.0
# Hard ceilings so a malicious/buggy client can't allocate unbounded geometry.
MAX_LINES = 32
MAX_ZONES = 16
MAX_POLYGON_POINTS = 32


class VideoProcessor:
    """Background thread that keeps a fresh annotated frame + counts ready.

    Thread-safe: the latest encoded JPEG, totals, series and error state are all
    guarded by a single lock. Exceptions inside the worker are captured into
    ``last_error`` and surfaced on ``/api/health`` instead of dying silently.
    """

    def __init__(
        self,
        source: str = "synthetic",
        width: int = 640,
        height: int = 480,
        lines: Optional[List[CountingLine]] = None,
        zones: Optional[List[Zone]] = None,
        loop: bool = True,
        store: Optional[EventStore] = None,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.lines = lines
        self.zones = zones
        self.loop = loop
        self.store = store or EventStore(
            maxlen=int(os.environ.get("VISIONCOUNT_EVENT_BUFFER", "500")),
            db_path=os.environ.get("VISIONCOUNT_DB") or None,
            webhook_url=os.environ.get("VISIONCOUNT_WEBHOOK") or None,
        )

        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_heatmap_png: Optional[bytes] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.pipeline: Optional[Pipeline] = None
        # Rolling time series of total crossings for the dashboard chart.
        self._series: Deque[Tuple[float, int]] = deque(maxlen=120)
        self._fps = 0.0
        self.last_error: Optional[str] = None
        self._last_frame_at = 0.0
        self._frames_seen = 0

    # ---- source frames ---------------------------------------------------

    def _open_frames(self):
        if self.source == "synthetic":
            # Effectively endless synthetic feed.
            while True:
                yield from synthetic_frames(
                    width=self.width, height=self.height, n_frames=250
                )
                if not self.loop:
                    return
        else:
            cap_arg = int(self.source) if self.source.isdigit() else self.source
            while True:
                cap = cv2.VideoCapture(cap_arg)
                if not cap.isOpened():
                    raise RuntimeError(f"cannot open source '{self.source}'")
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        yield frame
                finally:
                    cap.release()
                if not self.loop:
                    return

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> "VideoProcessor":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        # Wrap the whole worker so a bad source (unreadable file, missing webcam)
        # is captured and surfaced via /api/health rather than dying silently in
        # a daemon thread and leaving a frozen blank feed.
        try:
            self._run_inner()
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            with self._lock:
                self.last_error = f"{type(exc).__name__}: {exc}"
            self._running = False

    def _run_inner(self) -> None:
        frame_iter = self._open_frames()
        first = next(frame_iter)
        h, w = first.shape[:2]
        self.pipeline = Pipeline(
            w,
            h,
            lines=self.lines,
            zones=self.zones,
            config=PipelineConfig(min_area=350),
        )

        def process_and_store(frame: np.ndarray) -> None:
            assert self.pipeline is not None
            result = self.pipeline.process(frame)
            # Persist each crossing (ring + optional SQLite + optional webhook).
            if result.events:
                self.store.record_many(result.events)
            annotated = self.pipeline.annotate(frame, result)
            annotated = self.pipeline.heatmap.overlay(annotated, alpha=0.4)
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            heat = self.pipeline.heatmap.colorized()
            ok_h, buf_h = cv2.imencode(".png", heat)
            total = sum(c["total"] for c in result.totals.values())
            with self._lock:
                if ok:
                    self._latest_jpeg = buf.tobytes()
                if ok_h:
                    self._latest_heatmap_png = buf_h.tobytes()
                self._series.append((time.time(), total))
                self._last_frame_at = time.time()
                self._frames_seen += 1
                self.last_error = None

        last = time.time()
        process_and_store(first)
        for frame in frame_iter:
            if not self._running:
                break
            process_and_store(frame)
            now = time.time()
            dt = now - last
            last = now
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
            # Pace to a sensible rate so a synthetic feed does not peg a core.
            time.sleep(0.03)

    # ---- live line/zone mutation ----------------------------------------

    def add_line(self, name: str, start, end) -> None:
        """Add a counting line to the *running* pipeline (thread-safe)."""
        with self._lock:
            if self.pipeline is None:
                # Not started yet: stash so it is created with the pipeline.
                self.lines = list(self.lines or [])
                self.lines.append(CountingLine(name=name, start=start, end=end))
            else:
                self.pipeline.counter.add_line(
                    CountingLine(name=name, start=start, end=end)
                )

    def remove_line(self, name: str) -> bool:
        with self._lock:
            if self.pipeline is None:
                before = len(self.lines or [])
                self.lines = [ln for ln in (self.lines or []) if ln.name != name]
                return len(self.lines) < before
            counter = self.pipeline.counter
            before = len(counter.lines)
            counter.lines = [ln for ln in counter.lines if ln.name != name]
            return len(counter.lines) < before

    def line_count(self) -> int:
        with self._lock:
            if self.pipeline is not None:
                return len(self.pipeline.counter.lines)
            return len(self.lines or [])

    def add_zone(self, name: str, polygon) -> None:
        with self._lock:
            if self.pipeline is None:
                self.zones = list(self.zones or [])
                self.zones.append(Zone(name=name, polygon=polygon))
            else:
                self.pipeline.zone_counter.add_zone(Zone(name=name, polygon=polygon))

    def zone_count(self) -> int:
        with self._lock:
            if self.pipeline is not None:
                return len(self.pipeline.zone_counter.zones)
            return len(self.zones or [])

    # ---- accessors -------------------------------------------------------

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def get_heatmap_png(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_heatmap_png

    def get_counts(self) -> Dict:
        with self._lock:
            totals = self.pipeline.totals() if self.pipeline else {}
            zones = self.pipeline.zones() if self.pipeline else {}
            series = [{"t": round(t, 2), "total": v} for t, v in self._series]
            fps = round(self._fps, 1)
        return {
            "lines": totals,
            "zones": zones,
            "grand_total": sum(c["total"] for c in totals.values()),
            "series": series,
            "fps": fps,
        }

    def health(self) -> Dict:
        now = time.time()
        with self._lock:
            error = self.last_error
            last_frame_at = self._last_frame_at
            frames_seen = self._frames_seen
            running = self._running
        stale = running and last_frame_at > 0 and (now - last_frame_at) > STALE_AFTER
        if error is not None:
            status = "degraded"
        elif stale:
            status = "stale"
        elif frames_seen == 0:
            status = "starting" if running else "ok"
        else:
            status = "ok"
        return {
            "status": status,
            "source": self.source,
            "error": error,
            "frames_seen": frames_seen,
            "seconds_since_frame": (
                round(now - last_frame_at, 2) if last_frame_at else None
            ),
            "events_recorded": self.store.count(),
        }


# ---- security helpers ----------------------------------------------------


class RateLimiter:
    """Minimal fixed-window per-client rate limiter (thread-safe).

    Not a distributed limiter -- just enough to stop a single client from
    hammering the mutating/export endpoints of a self-hosted instance.
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = max(1, per_minute)
        self._lock = threading.Lock()
        self._hits: Dict[str, Deque[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        window = 60.0
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True


def _clamp(v: float, lo: float, hi: float) -> int:
    return int(max(lo, min(hi, v)))


def create_app(processor: Optional[VideoProcessor] = None) -> Flask:
    """Application factory. If no processor is given, one is created + started."""
    app = Flask(__name__)
    source = os.environ.get("VISIONCOUNT_SOURCE", "synthetic")
    api_key = os.environ.get("VISIONCOUNT_API_KEY") or None
    cors_origin = os.environ.get("VISIONCOUNT_CORS_ORIGIN") or None
    limiter = RateLimiter(int(os.environ.get("VISIONCOUNT_RATE_LIMIT", "120")))

    if processor is None:
        processor = VideoProcessor(source=source)

    app.config["PROCESSOR"] = processor
    # Cap request bodies (line/zone JSON is tiny); rejects oversized payloads.
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

    # Worker lifecycle. Importing the module must never spin up an OpenCV
    # background thread (keeps tests / CI / `flask --app` cheap and avoids a
    # daemon thread aborting during interpreter teardown on some platforms).
    #   VISIONCOUNT_AUTOSTART=0 (default): do not start on import; start lazily
    #                                      on the first HTTP request instead.
    #   VISIONCOUNT_AUTOSTART=1:           start eagerly here.
    # Tests set VISIONCOUNT_LAZY_START=0 to keep the worker fully off.
    autostart = os.environ.get("VISIONCOUNT_AUTOSTART", "0") == "1"
    lazy_start = os.environ.get("VISIONCOUNT_LAZY_START", "1") == "1"
    if autostart:
        processor.start()

    if lazy_start and not autostart:

        @app.before_request
        def _ensure_started():  # pragma: no cover - trivial guard
            processor.start()  # idempotent; no-op once running

    # ---- security wiring -------------------------------------------------

    @app.after_request
    def _cors(resp: Response) -> Response:
        # Locked down by default: only emit CORS headers when explicitly opted in.
        if cors_origin:
            resp.headers["Access-Control-Allow-Origin"] = cors_origin
            resp.headers["Vary"] = "Origin"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        return resp

    def _require_key() -> None:
        if api_key is None:
            return
        supplied = request.headers.get("X-API-Key") or request.args.get("api_key")
        if supplied != api_key:
            abort(401, description="invalid or missing API key")

    def _rate_limit() -> None:
        client = request.remote_addr or "anon"
        if not limiter.allow(client):
            abort(429, description="rate limit exceeded")

    # ---- pages / streams -------------------------------------------------

    @app.route("/")
    def index():
        return render_template_string(INDEX_HTML, feed_source=processor.source)

    @app.route("/video_feed")
    def video_feed():
        # If the worker has permanently failed, do not hang a client on a stream
        # that will never produce a frame: return a clear 503 instead.
        h = processor.health()
        if h["status"] == "degraded" and h["frames_seen"] == 0:
            abort(503, description=h["error"] or "video worker unavailable")

        def gen():
            boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            while True:
                jpeg = processor.get_jpeg()
                if jpeg is not None:
                    yield boundary + jpeg + b"\r\n"
                time.sleep(0.05)

        return Response(
            gen(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    @app.route("/heatmap.png")
    def heatmap_png():
        png = processor.get_heatmap_png()
        if png is None:
            # 1x1 transparent placeholder until the first frame is ready.
            blank = np.zeros((2, 2, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".png", blank)
            png = buf.tobytes()
        return Response(png, mimetype="image/png")

    # ---- analytics API ---------------------------------------------------

    @app.route("/api/counts")
    def api_counts():
        return jsonify(processor.get_counts())

    @app.route("/api/events")
    def api_events():
        try:
            limit = int(request.args.get("limit", "100"))
        except ValueError:
            abort(400, description="limit must be an integer")
        limit = max(1, min(limit, 1000))
        return jsonify({"events": processor.store.recent(limit)})

    @app.route("/export.csv")
    def export_csv():
        _rate_limit()
        return Response(
            processor.store.iter_csv(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=visioncount_crossings.csv"
            },
        )

    @app.route("/api/health")
    def api_health():
        return jsonify(processor.health())

    # ---- live line/zone editing -----------------------------------------

    @app.route("/api/lines", methods=["GET", "POST", "DELETE", "OPTIONS"])
    def api_lines():
        if request.method == "OPTIONS":
            return ("", 204)
        if request.method == "GET":
            counts = processor.get_counts()
            lines = []
            pipe = processor.pipeline
            if pipe is not None:
                for ln in pipe.counter.lines:
                    lines.append(
                        {
                            "name": ln.name,
                            "start": [int(ln.start[0]), int(ln.start[1])],
                            "end": [int(ln.end[0]), int(ln.end[1])],
                            **counts["lines"].get(ln.name, {}),
                        }
                    )
            return jsonify({"lines": lines})

        _require_key()
        _rate_limit()

        if request.method == "DELETE":
            name = (request.args.get("name") or "").strip()
            if not name:
                abort(400, description="name is required")
            removed = processor.remove_line(name)
            return jsonify({"removed": removed}), (200 if removed else 404)

        # POST: add a line.
        data = request.get_json(silent=True) or {}
        name, start, end = _validate_line(data, processor)
        if processor.line_count() >= MAX_LINES:
            abort(400, description=f"too many lines (max {MAX_LINES})")
        processor.add_line(name, start, end)
        return jsonify({"ok": True, "name": name}), 201

    @app.route("/api/zones", methods=["GET", "POST", "OPTIONS"])
    def api_zones():
        if request.method == "OPTIONS":
            return ("", 204)
        if request.method == "GET":
            return jsonify({"zones": processor.get_counts()["zones"]})

        _require_key()
        _rate_limit()
        data = request.get_json(silent=True) or {}
        name, polygon = _validate_zone(data, processor)
        if processor.zone_count() >= MAX_ZONES:
            abort(400, description=f"too many zones (max {MAX_ZONES})")
        processor.add_zone(name, polygon)
        return jsonify({"ok": True, "name": name}), 201

    @app.errorhandler(400)
    @app.errorhandler(401)
    @app.errorhandler(404)
    @app.errorhandler(429)
    @app.errorhandler(503)
    def _json_error(err):
        return (
            jsonify({"error": getattr(err, "description", str(err))}),
            getattr(err, "code", 500),
        )

    return app


# ---- input validation ----------------------------------------------------


def _validate_line(data: Dict, processor: VideoProcessor):
    name = str(data.get("name", "")).strip()[:40]
    if not name:
        abort(400, description="name is required")
    start = data.get("start")
    end = data.get("end")
    if not (_is_point(start) and _is_point(end)):
        abort(400, description="start and end must be [x, y] points")
    w, h = processor.width, processor.height
    pipe = processor.pipeline
    if pipe is not None:
        w, h = pipe.width, pipe.height
    s = (_clamp(start[0], 0, w), _clamp(start[1], 0, h))
    e = (_clamp(end[0], 0, w), _clamp(end[1], 0, h))
    if s == e:
        abort(400, description="line start and end must differ")
    return name, s, e


def _validate_zone(data: Dict, processor: VideoProcessor):
    name = str(data.get("name", "")).strip()[:40]
    if not name:
        abort(400, description="name is required")
    polygon = data.get("polygon")
    if not isinstance(polygon, list) or len(polygon) < 3:
        abort(400, description="polygon must be a list of >= 3 [x, y] points")
    if len(polygon) > MAX_POLYGON_POINTS:
        abort(400, description=f"polygon has too many points (max {MAX_POLYGON_POINTS})")
    w, h = processor.width, processor.height
    pipe = processor.pipeline
    if pipe is not None:
        w, h = pipe.width, pipe.height
    clamped = []
    for pt in polygon:
        if not _is_point(pt):
            abort(400, description="polygon points must be [x, y]")
        clamped.append((_clamp(pt[0], 0, w), _clamp(pt[1], 0, h)))
    return name, clamped


def _is_point(p) -> bool:
    return (
        isinstance(p, (list, tuple))
        and len(p) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in p)
    )


# Module-level app for `flask --app visioncount.app run`.
app = create_app()


def main() -> None:  # pragma: no cover - manual run
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    processor: VideoProcessor = app.config["PROCESSOR"]
    processor.start()
    try:
        app.run(host=host, port=port, threaded=True)
    finally:
        processor.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
