# VisionCount — browser demo

A **dependency-free, self-contained** port of the VisionCount core pipeline that
runs entirely in the browser. No server, no build step, no CDN — just open the
page.

```bash
# from the repo root
python -m http.server -d web 8000
# open http://127.0.0.1:8000
```

Or drop the four files (`index.html`, `app.js`, `visioncount.js`, `scene.js`) on
any static host.

## What it is

`visioncount.js` re-implements the same algorithms as the Python package —
cross-product line-crossing geometry, greedy centroid tracking with constant-
velocity prediction, a decaying motion heatmap, and point-in-polygon zone
occupancy. `scene.js` mirrors the synthetic feed. `app.js` runs the pipeline on a
`<canvas>` each frame and draws boxes, IDs, trajectories, lines, zones and the
heatmap, updating a live counts panel and an inline SVG chart.

The server pipeline (OpenCV worker thread + MJPEG) cannot run in a browser tab, so
this faithful JS port is what makes VisionCount embeddable as a static live demo.

## Try it

- **Drag** on the video to draw a counting line.
- Switch to **Draw zone**, click polygon points, then **Finish zone**.
- Switch the source to **Webcam** — frames stay entirely on your device.
