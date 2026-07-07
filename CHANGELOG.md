# Changelog

All notable changes to VisionCount are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-07-07

The "real analytics + live demo" release. VisionCount stops being a live-counter
you have to watch and becomes an analytics tool that records, exports and streams
its crossings — plus a browser-only demo so the moving result is visible on
GitHub without cloning.

### Added

- **Crossing persistence & export.** A new `EventStore` records every crossing
  (`seq`, ISO timestamp, line, direction, `track_id`, `x`, `y`) to a bounded
  in-memory ring and, optionally, to SQLite (`VISIONCOUNT_DB` / `--db`). Sequence
  numbers resume across restarts so ids never collide.
- **Analytics API.** `GET /api/events` returns recent crossings as JSON;
  `GET /export.csv` streams every recorded crossing as CSV; each crossing can be
  POSTed to an outbound webhook (`VISIONCOUNT_WEBHOOK`), fired best-effort off the
  video loop.
- **Zone / ROI occupancy counting.** A new `Zone` polygon primitive reports live
  occupancy, peak occupancy, entries and dwell time via point-in-polygon testing,
  rendered as a translucent region and a second dashboard table. `--zone` on the
  CLI and `POST /api/zones` on the dashboard.
- **Interactive dashboard.** Draw, name and delete counting **lines** (drag) and
  **zones** (click a polygon) directly on the live feed; changes mutate the
  running pipeline via `GET/POST/DELETE /api/lines` and `POST /api/zones`. The
  crossings chart is now inline SVG, with a recent-crossings feed and a CSV export
  button.
- **Browser-only live demo** (`web/`). A faithful, dependency-free JavaScript port
  of the whole core pipeline (detector → centroid tracker with velocity
  prediction → cross-product line counter → decaying heatmap → polygon zones) runs
  entirely client-side on a `<canvas>`, on a synthetic feed or the webcam. No
  server, no MJPEG, embeddable as a static bundle.
- **Looping demo GIF.** `python examples/generate_demo.py --gif` renders
  `docs/demo.gif` (boxes, IDs, trajectories, incrementing lines, a filling zone,
  the blooming heatmap), embedded at the top of the README and produced as a CI
  artifact.
- **Docker & production server.** A slim multi-stage `Dockerfile`
  (opencv-python-headless, non-root, healthcheck) and `docker-compose.yml`
  (including a `demo` profile with SQLite logging) run the dashboard under
  **waitress** with one command.
- **Constant-velocity tracking.** The centroid tracker now matches detections
  against each track's *predicted* next position and maintains a smoothed velocity
  estimate, reducing ID swaps when objects cross paths or briefly occlude.
- **Security controls.** Per-client rate limiting (`VISIONCOUNT_RATE_LIMIT`),
  optional API key on mutating endpoints (`VISIONCOUNT_API_KEY`), locked-down CORS
  (off unless `VISIONCOUNT_CORS_ORIGIN` is set), request-body cap, and validated,
  frame-clamped geometry input.
- **Coverage.** CI now runs `pytest --cov` with an 80% gate; a self-contained
  coverage badge (`docs/coverage.svg`) and status badges are shown in the README.
- **Docs.** `ARCHITECTURE.md`, a `.env.example`, and a rewritten README with a
  hero, badge row, architecture blurb and quickstart.

### Changed

- **Worker hardening.** Exceptions inside the background video thread are now
  captured into `last_error` and surfaced on `/api/health` as `degraded` (with the
  message) instead of dying silently behind a frozen feed; a stale feed reports
  `stale`; `/video_feed` returns a clean `503` when the worker never produced a
  frame. `get_counts()` now reads pipeline totals **inside** the lock.
- `GET /api/counts` now includes a `zones` object alongside `lines`.
- The CLI `--json` output is now `{"lines": …, "zones": …}` (was the bare lines
  map).
- The dashboard HTML moved from an inline string in `app.py` to `templates.py`.

### Fixed

- Race between the worker thread mutating line counts and a request reading
  `totals()` outside the lock.
- Stale feeds and unreadable sources no longer present an indistinguishable blank
  frame with no error signal.

## [0.1.0] — 2026

Initial release: OpenCV MOG2 background-subtraction detector, pure-NumPy centroid
tracker, directional line-crossing counter, motion heatmap, Matplotlib report,
CLI, Flask dashboard with MJPEG streaming, optional YOLO detector, synthetic feed,
and a pytest suite with GitHub Actions CI.

[0.2.0]: https://github.com/xj16/visioncount/releases/tag/v0.2.0
[0.1.0]: https://github.com/xj16/visioncount/releases/tag/v0.1.0
