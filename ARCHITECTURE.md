# Architecture

VisionCount is a small, layered computer-vision pipeline with three front-ends
(a CLI, a Flask dashboard, and a browser-only demo) sitting on one shared core.
Every stage is a plain Python class with a narrow interface, so pieces can be
swapped or tested in isolation.

## The pipeline

```
                 ┌───────────────────────────────────────────────────────────┐
   BGR frame ───▶│  Detector          detect(frame) -> [Detection]            │
                 │    default: MOG2 background subtraction (detector.py)       │
                 │    optional: YOLO / ultralytics    (yolo_detector.py)       │
                 └───────────────────────────────┬───────────────────────────┘
                                                 ▼
                 ┌───────────────────────────────────────────────────────────┐
                 │  CentroidTracker   update([Detection]) -> {id: Track}       │
                 │    greedy nearest-centroid + constant-velocity prediction   │
                 │    (tracker.py)                                             │
                 └───────────────┬───────────────────────────┬───────────────┘
                                 ▼                           ▼
        ┌────────────────────────────────┐   ┌────────────────────────────────┐
        │ LineCounter  (counter.py)      │   │ ZoneCounter  (zone.py)         │
        │  cross-product direction test  │   │  point-in-polygon occupancy     │
        │  -> [CrossEvent]               │   │  + dwell / peak / entries       │
        └───────────────┬────────────────┘   └────────────────────────────────┘
                        ▼                                    │
        ┌────────────────────────────────┐                  ▼
        │ EventStore   (events.py)       │   ┌────────────────────────────────┐
        │  ring + SQLite + webhook       │   │ Heatmap  (heatmap.py)          │
        │  -> /api/events, /export.csv   │   │  decaying float32 accumulator   │
        └────────────────────────────────┘   └────────────────────────────────┘
```

`Pipeline` (`pipeline.py`) wires these together behind a single
`process(frame) -> FrameResult` call and an `annotate(frame, result)` drawing
helper. `FrameResult` carries the detections, tracks, crossing events, per-line
totals and per-zone occupancy for that frame.

### Why these algorithms

- **Background subtraction, not a neural net, by default.** MOG2 needs no weights
  and no `torch`, so `pip install visioncount` is light and runs on a Raspberry Pi.
  The `DetectorLike` protocol means a YOLO detector (or any `detect()`-shaped
  object) drops into the same pipeline when accuracy matters more than footprint.
- **Cross-product line direction.** A crossing is detected when the segment
  between a track's previous and current centroid intersects the counting-line
  segment. The *sign* of the 2D cross product of the line vector with the object's
  position decides `in` vs `out`, so direction is unambiguous regardless of how
  the line is drawn. Each track is counted at most once per line.
- **Velocity-predicted association.** The tracker matches detections against each
  track's *predicted* next centroid (last position + smoothed velocity), not its
  last seen position, which keeps IDs stable through brief occlusions and crossing
  paths.
- **Decaying heatmap.** Activity is stamped with a Gaussian blob into a float32
  buffer that is multiplied by a per-frame decay, so the heatmap reflects recent
  motion rather than all history.

## Front-ends

- **CLI** (`cli.py`) — headless or windowed runs on the synthetic clip, a webcam
  index, or a video file; can write an annotated MP4, a Matplotlib PNG report, and
  a SQLite event log.
- **Flask dashboard** (`app.py`, `templates.py`) — a `VideoProcessor` runs the
  pipeline on a background daemon thread and keeps the latest annotated JPEG,
  heatmap PNG, counts, and event log ready behind a lock. Flask serves an MJPEG
  stream, the analytics API, and an interactive HTML dashboard. Worker errors are
  captured and surfaced on `/api/health`.
- **Browser demo** (`web/`) — a dependency-free JavaScript re-implementation of
  the *same* core algorithms (`web/visioncount.js`) running on a `<canvas>`. The
  server pipeline (OpenCV thread + MJPEG) cannot run in a tab, so this port makes
  the result embeddable as a static bundle with zero backend.

## Concurrency & safety model

- The `VideoProcessor` worker thread is the only writer of frames/counts/events;
  request threads read snapshots under the same `threading.Lock`.
- The worker never starts on import (keeps CI and `flask --app` cheap); it starts
  eagerly with `VISIONCOUNT_AUTOSTART=1` or lazily on the first request.
- Mutating endpoints are rate-limited, optionally API-key-gated, and validate and
  clamp all geometry to the frame. CORS is off unless explicitly configured.

## Testing

`tests/` covers each layer in isolation (detector, tracker + velocity, counter
geometry, heatmap edge cases, zone occupancy/dwell, event store + SQLite +
CSV) plus a flagship `test_worker.py` that actually starts the background thread
against the synthetic source and asserts the MJPEG stream yields a valid JPEG,
counts advance, events persist, live line/zone edits take effect, and a bad
source surfaces as `degraded` with a `503`. CI runs the suite with an 80%
coverage gate on Linux (3.11–3.13) and Windows.
