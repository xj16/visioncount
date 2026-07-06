"""Flask dashboard for VisionCount.

Runs the pipeline in a background thread and serves:

* ``/``                  - an HTML dashboard (live video, counts, heatmap)
* ``/video_feed``        - MJPEG stream of annotated frames
* ``/heatmap.png``       - current motion heatmap as a PNG
* ``/api/counts``        - JSON of current line counts + a rolling time series
* ``/api/health``        - liveness probe

The video source defaults to the built-in synthetic clip so the dashboard works
with zero configuration (great for demos and CI). Point it at a webcam or video
file with the ``VISIONCOUNT_SOURCE`` environment variable.

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
from flask import Flask, Response, jsonify, render_template_string

from .counter import CountingLine
from .pipeline import Pipeline, PipelineConfig
from .synthetic import frames as synthetic_frames


class VideoProcessor:
    """Background thread that keeps a fresh annotated frame + counts ready.

    Thread-safe: the latest encoded JPEG and the totals are guarded by a lock.
    """

    def __init__(
        self,
        source: str = "synthetic",
        width: int = 640,
        height: int = 480,
        lines: Optional[List[CountingLine]] = None,
        loop: bool = True,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.lines = lines
        self.loop = loop

        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_heatmap_png: Optional[bytes] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.pipeline: Optional[Pipeline] = None
        # Rolling time series of total crossings for the dashboard chart.
        self._series: Deque[Tuple[float, int]] = deque(maxlen=120)
        self._fps = 0.0

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
        frame_iter = self._open_frames()
        first = next(frame_iter)
        h, w = first.shape[:2]
        self.pipeline = Pipeline(
            w,
            h,
            lines=self.lines,
            config=PipelineConfig(min_area=350),
        )

        def process_and_store(frame: np.ndarray) -> None:
            assert self.pipeline is not None
            result = self.pipeline.process(frame)
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

    # ---- accessors -------------------------------------------------------

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def get_heatmap_png(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_heatmap_png

    def get_counts(self) -> Dict:
        totals = self.pipeline.totals() if self.pipeline else {}
        with self._lock:
            series = [
                {"t": round(t, 2), "total": v} for t, v in self._series
            ]
        return {
            "lines": totals,
            "grand_total": sum(c["total"] for c in totals.values()),
            "series": series,
            "fps": round(self._fps, 1),
        }


_INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>VisionCount Dashboard</title>
<style>
  :root { --bg:#0f2f4a; --bg2:#163a52; --accent:#1f7a8c; --card:#ffffff; --ink:#12232e; --muted:#5b7280; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: "Segoe UI", system-ui, sans-serif; color:var(--ink); background:#eef3f6; }
  header { background:linear-gradient(120deg,var(--bg),var(--bg2)); color:#fff; padding:20px 28px; }
  header h1 { margin:0; font-size:22px; letter-spacing:.3px; }
  header p { margin:4px 0 0; opacity:.8; font-size:13px; }
  .wrap { max-width:1100px; margin:22px auto; padding:0 18px; display:grid; grid-template-columns:2fr 1fr; gap:18px; }
  @media (max-width:820px){ .wrap{ grid-template-columns:1fr; } }
  .card { background:var(--card); border-radius:12px; box-shadow:0 2px 10px rgba(15,47,74,.08); padding:16px; }
  .card h2 { margin:0 0 12px; font-size:14px; text-transform:uppercase; letter-spacing:.6px; color:var(--muted); }
  img.feed { width:100%; border-radius:8px; display:block; background:#000; }
  .stat { display:flex; align-items:baseline; gap:10px; margin-bottom:10px; }
  .stat .num { font-size:34px; font-weight:700; color:var(--accent); }
  .stat .lbl { color:var(--muted); font-size:13px; }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  th,td { text-align:left; padding:7px 6px; border-bottom:1px solid #eef1f3; }
  th { color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; }
  .pill { display:inline-block; padding:2px 9px; border-radius:999px; font-size:12px; background:#e6f2f4; color:var(--accent); }
  canvas { width:100%; height:120px; }
  footer { text-align:center; color:var(--muted); font-size:12px; padding:18px; }
  code { background:#eef1f3; padding:1px 6px; border-radius:5px; }
</style>
</head>
<body>
<header>
  <h1>VisionCount</h1>
  <p>Real-time object counting &amp; line-crossing analytics &middot; source: <code>{{ feed_source }}</code></p>
</header>
<div class="wrap">
  <div class="card">
    <h2>Live feed</h2>
    <img class="feed" src="/video_feed" alt="live annotated feed"/>
  </div>
  <div>
    <div class="card" style="margin-bottom:18px;">
      <h2>Totals</h2>
      <div class="stat"><span class="num" id="grand">0</span><span class="lbl">total crossings</span></div>
      <div class="stat"><span class="num" id="fps" style="font-size:20px;">0</span><span class="lbl">fps</span></div>
      <table id="lines"><thead><tr><th>Line</th><th>In</th><th>Out</th><th>Net</th></tr></thead><tbody></tbody></table>
    </div>
    <div class="card" style="margin-bottom:18px;">
      <h2>Crossings over time</h2>
      <canvas id="chart" width="320" height="120"></canvas>
    </div>
    <div class="card">
      <h2>Heatmap</h2>
      <img class="feed" id="heat" src="/heatmap.png" alt="motion heatmap"/>
    </div>
  </div>
</div>
<footer>VisionCount &middot; OpenCV background-subtraction pipeline &middot; MIT licensed</footer>
<script>
function drawChart(series){
  const c = document.getElementById('chart'); const ctx = c.getContext('2d');
  const W=c.width,H=c.height; ctx.clearRect(0,0,W,H);
  if(!series.length) return;
  const vals = series.map(p=>p.total);
  const max = Math.max(1, ...vals);
  ctx.strokeStyle='#1f7a8c'; ctx.lineWidth=2; ctx.beginPath();
  series.forEach((p,i)=>{
    const x=(i/(series.length-1||1))*W;
    const y=H-(p.total/max)*(H-8)-4;
    i?ctx.lineTo(x,y):ctx.moveTo(x,y);
  });
  ctx.stroke();
}
async function tick(){
  try{
    const r = await fetch('/api/counts'); const d = await r.json();
    document.getElementById('grand').textContent = d.grand_total;
    document.getElementById('fps').textContent = d.fps;
    const tb = document.querySelector('#lines tbody'); tb.innerHTML='';
    for(const [name,c] of Object.entries(d.lines)){
      const tr=document.createElement('tr');
      tr.innerHTML=`<td><span class="pill">${name}</span></td><td>${c.in}</td><td>${c.out}</td><td>${c.net}</td>`;
      tb.appendChild(tr);
    }
    drawChart(d.series);
  }catch(e){ /* ignore transient errors */ }
}
setInterval(tick, 1000); tick();
// refresh heatmap periodically
setInterval(()=>{ document.getElementById('heat').src='/heatmap.png?'+Date.now(); }, 2000);
</script>
</body>
</html>
"""


def create_app(processor: Optional[VideoProcessor] = None) -> Flask:
    """Application factory. If no processor is given, one is created + started."""
    app = Flask(__name__)
    source = os.environ.get("VISIONCOUNT_SOURCE", "synthetic")

    if processor is None:
        processor = VideoProcessor(source=source)

    app.config["PROCESSOR"] = processor
    # Start lazily so importing the module (e.g. in tests / CI) is cheap.
    if os.environ.get("VISIONCOUNT_AUTOSTART", "1") == "1":
        processor.start()

    @app.route("/")
    def index():
        return render_template_string(_INDEX_HTML, feed_source=processor.source)

    @app.route("/video_feed")
    def video_feed():
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

    @app.route("/api/counts")
    def api_counts():
        return jsonify(processor.get_counts())

    @app.route("/api/health")
    def api_health():
        return jsonify({"status": "ok", "source": processor.source})

    return app


# Module-level app for `flask --app visioncount.app run`.
app = create_app()


def main() -> None:  # pragma: no cover - manual run
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":  # pragma: no cover
    main()
