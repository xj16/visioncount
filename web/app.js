/*
 * app.js — drives the in-browser VisionCount demo.
 *
 * Renders a source (synthetic scene or webcam) to a canvas, runs the ported
 * pipeline on each frame, draws boxes/ids/trajectories/lines/zones, updates the
 * live counts panel + heatmap + sparkline, and lets the viewer draw counting
 * lines and polygon zones directly on the video.
 */

import {
  MotionDetector, CentroidTracker, LineCounter, CountingLine, Heatmap, Zone, PALETTE,
} from './visioncount.js';
import { SyntheticScene } from './scene.js';

const W = 480, H = 360;
const feed = document.getElementById('feed');
const heatC = document.getElementById('heat');
const overlay = document.getElementById('overlay');
const fctx = feed.getContext('2d', { willReadFrequently: true });
const hctx = heatC.getContext('2d');
const octx = overlay.getContext('2d');

let detector = new MotionDetector(W, H, { minArea: 200, threshold: 26 });
let tracker = new CentroidTracker({ maxDistance: 80, maxDisappeared: 14 });
let counter = new LineCounter([new CountingLine('center', [W / 2, 0], [W / 2, H])]);
let zones = [];
let heat = new Heatmap(W, H, { decay: 0.985, radius: 13 });
let series = [];
let totalIn = 0, totalOut = 0;
let lineColors = { center: PALETTE[0] };
let colorIdx = 1;

let scene = new SyntheticScene(W, H);
let usingWebcam = false;
let webcamVideo = null;

// ---- source ---------------------------------------------------------------

function drawSource() {
  if (usingWebcam && webcamVideo && webcamVideo.readyState >= 2) {
    // Cover-fit the webcam into the 4:3 canvas.
    const vw = webcamVideo.videoWidth, vh = webcamVideo.videoHeight;
    const scale = Math.max(W / vw, H / vh);
    const dw = vw * scale, dh = vh * scale;
    fctx.drawImage(webcamVideo, (W - dw) / 2, (H - dh) / 2, dw, dh);
  } else {
    scene.step(fctx);
  }
}

// ---- colors ---------------------------------------------------------------

function colorFor(name) {
  if (!(name in lineColors)) lineColors[name] = PALETTE[(colorIdx++) % PALETTE.length];
  return lineColors[name];
}

// ---- main loop ------------------------------------------------------------

let last = performance.now(), fps = 0;

function frame(now) {
  drawSource();
  const image = fctx.getImageData(0, 0, W, H);
  const dets = detector.detect(image);
  const tracks = tracker.update(dets);
  const events = counter.update(tracks);
  for (const e of events) { if (e.direction === 'in') totalIn++; else totalOut++; }
  for (const z of zones) z.update(tracks);

  // Heatmap from centroids.
  const pts = [];
  for (const t of tracks.values()) pts.push([t.cx, t.cy]);
  heat.addPoints(pts);
  hctx.clearRect(0, 0, W, H);
  heat.render(hctx, 0.5);

  drawAnnotations(tracks);
  updatePanel(tracks, events.length > 0);

  const dt = now - last; last = now;
  if (dt > 0) fps = 0.9 * fps + 0.1 * (1000 / dt);
  document.getElementById('fpsBadge').textContent = fps.toFixed(0) + ' fps';
  requestAnimationFrame(frame);
}

function drawAnnotations(tracks) {
  // Boxes, ids and trajectories go on the feed canvas (over the source).
  fctx.lineWidth = 2;
  fctx.font = '12px ui-monospace, monospace';
  for (const t of tracks.values()) {
    const col = PALETTE[t.id % PALETTE.length];
    const b = t.bbox;
    fctx.strokeStyle = col;
    fctx.strokeRect(b.x, b.y, b.w, b.h);
    fctx.fillStyle = col;
    fctx.fillText('#' + t.id, b.x, Math.max(11, b.y - 4));
    if (t.traj.length >= 2) {
      fctx.beginPath();
      t.traj.forEach((p, i) => (i ? fctx.lineTo(p[0], p[1]) : fctx.moveTo(p[0], p[1])));
      fctx.strokeStyle = col; fctx.lineWidth = 1.5; fctx.stroke(); fctx.lineWidth = 2;
    }
  }
  // Zones (translucent) then lines, drawn on the feed.
  for (const z of zones) {
    fctx.beginPath();
    z.polygon.forEach((p, i) => (i ? fctx.lineTo(p[0], p[1]) : fctx.moveTo(p[0], p[1])));
    fctx.closePath();
    fctx.fillStyle = 'rgba(168,120,240,.18)'; fctx.fill();
    fctx.strokeStyle = '#a878f0'; fctx.lineWidth = 2; fctx.stroke();
  }
  for (const line of counter.lines) {
    const col = colorFor(line.name);
    fctx.strokeStyle = col; fctx.lineWidth = 3;
    fctx.beginPath(); fctx.moveTo(line.start[0], line.start[1]);
    fctx.lineTo(line.end[0], line.end[1]); fctx.stroke();
    fctx.fillStyle = col; fctx.font = '11px ui-monospace, monospace';
    fctx.fillText(`${line.name} ${line.in}/${line.out}`,
      Math.min(W - 90, line.start[0] + 4), Math.max(12, line.start[1] + 12));
  }
}

// ---- side panel -----------------------------------------------------------

function updatePanel(tracks, crossed) {
  document.getElementById('grand').textContent = totalIn + totalOut;
  document.getElementById('mIn').textContent = totalIn;
  document.getElementById('mOut').textContent = totalOut;
  document.getElementById('mTracks').textContent = tracks.size;

  const lb = document.querySelector('#lines tbody'); lb.innerHTML = '';
  for (const line of counter.lines) {
    const col = colorFor(line.name);
    const tr = document.createElement('tr');
    tr.innerHTML =
      `<td><span class="sw" style="background:${col}"></span>${line.name}</td>` +
      `<td class="n">${line.in}</td><td class="n">${line.out}</td><td class="n">${line.net}</td>` +
      `<td class="n"><span class="rm" title="remove">×</span></td>`;
    tr.querySelector('.rm').onclick = () => { counter.remove(line.name); };
    lb.appendChild(tr);
  }

  const zb = document.querySelector('#zones tbody');
  if (!zones.length) {
    zb.innerHTML = '<tr><td colspan="4" class="empty">Draw a zone on the feed.</td></tr>';
  } else {
    zb.innerHTML = '';
    for (const z of zones) {
      const tr = document.createElement('tr');
      tr.innerHTML =
        `<td><span class="sw" style="background:#a878f0"></span>${z.name}</td>` +
        `<td class="n">${z.occupancy}</td><td class="n">${z.peak}</td><td class="n">${z.entries}</td>`;
      zb.appendChild(tr);
    }
  }

  if (crossed || series.length === 0 || series[series.length - 1] !== totalIn + totalOut) {
    series.push(totalIn + totalOut);
    if (series.length > 160) series.shift();
    drawSpark();
  }
}

function drawSpark() {
  const svg = document.getElementById('spark');
  if (series.length < 2) { svg.innerHTML = ''; return; }
  const w = 320, h = 74, max = Math.max(1, ...series);
  const pts = series.map((v, i) => [
    (i / (series.length - 1)) * w,
    h - (v / max) * (h - 8) - 4,
  ]);
  const path = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
  const end = pts[pts.length - 1];
  svg.innerHTML =
    `<path d="${path} L${w} ${h} L0 ${h} Z" fill="rgba(56,214,196,.12)"/>` +
    `<path d="${path}" fill="none" stroke="#38d6c4" stroke-width="2"/>` +
    `<circle cx="${end[0].toFixed(1)}" cy="${end[1].toFixed(1)}" r="3" fill="#38d6c4"/>`;
}

// ---- interaction: draw lines & zones -------------------------------------

let mode = 'line';
let drag = null;
let zonePts = [];
const hint = document.getElementById('hint');

function toCanvas(e) {
  const r = overlay.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H];
}

function setMode(m) {
  mode = m; drag = null; zonePts = [];
  document.getElementById('tLine').setAttribute('aria-pressed', m === 'line');
  document.getElementById('tZone').setAttribute('aria-pressed', m === 'zone');
  document.getElementById('tFinish').style.display = m === 'zone' ? '' : 'none';
  hint.textContent = m === 'line'
    ? 'Drag across the video to place a counting line.'
    : 'Click to add polygon points, then Finish zone.';
  redrawOverlay();
}
document.getElementById('tLine').onclick = () => setMode('line');
document.getElementById('tZone').onclick = () => setMode('zone');
document.getElementById('tClear').onclick = () => {
  counter.lines = []; zones = []; hint.textContent = 'Cleared. Draw a new line or zone.';
};
document.getElementById('tFinish').onclick = () => {
  if (zonePts.length < 3) { hint.textContent = 'A zone needs at least 3 points.'; return; }
  zones.push(new Zone('zone' + (zones.length + 1), zonePts.slice()));
  zonePts = []; redrawOverlay();
};

overlay.addEventListener('mousedown', (e) => {
  if (mode !== 'line') return;
  const p = toCanvas(e); drag = { x0: p[0], y0: p[1], x1: p[0], y1: p[1] };
});
overlay.addEventListener('mousemove', (e) => {
  if (!drag) return; const p = toCanvas(e); drag.x1 = p[0]; drag.y1 = p[1]; redrawOverlay();
});
window.addEventListener('mouseup', () => {
  if (mode === 'line' && drag) {
    const d = drag; drag = null;
    if (Math.hypot(d.x1 - d.x0, d.y1 - d.y0) > 10) {
      counter.add(new CountingLine('line' + (counter.lines.length + 1),
        [d.x0, d.y0], [d.x1, d.y1]));
    }
    redrawOverlay();
  }
});
overlay.addEventListener('click', (e) => {
  if (mode !== 'zone') return;
  zonePts.push(toCanvas(e)); redrawOverlay();
});

function redrawOverlay() {
  octx.clearRect(0, 0, W, H);
  if (drag) {
    octx.strokeStyle = '#ffd34d'; octx.lineWidth = 3;
    octx.beginPath(); octx.moveTo(drag.x0, drag.y0); octx.lineTo(drag.x1, drag.y1); octx.stroke();
  }
  if (zonePts.length) {
    octx.strokeStyle = '#a878f0'; octx.fillStyle = 'rgba(168,120,240,.2)'; octx.lineWidth = 2;
    octx.beginPath();
    zonePts.forEach((p, i) => (i ? octx.lineTo(p[0], p[1]) : octx.moveTo(p[0], p[1])));
    if (zonePts.length > 2) octx.closePath();
    octx.fill(); octx.stroke();
    for (const p of zonePts) { octx.fillStyle = '#a878f0'; octx.beginPath(); octx.arc(p[0], p[1], 4, 0, 7); octx.fill(); }
  }
}

// ---- source switching -----------------------------------------------------

const btnSyn = document.getElementById('srcSynthetic');
const btnCam = document.getElementById('srcWebcam');

btnSyn.onclick = () => switchSource(false);
btnCam.onclick = () => switchSource(true);

async function switchSource(webcam) {
  if (webcam && !webcamVideo) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      webcamVideo = document.createElement('video');
      webcamVideo.srcObject = stream; webcamVideo.playsInline = true;
      await webcamVideo.play();
    } catch (err) {
      hint.textContent = 'Webcam unavailable (' + (err.name || 'blocked') + '). Staying on synthetic.';
      return;
    }
  }
  usingWebcam = webcam;
  btnSyn.setAttribute('aria-pressed', String(!webcam));
  btnCam.setAttribute('aria-pressed', String(webcam));
  // Reset the pipeline so counts start fresh for the new source.
  detector = new MotionDetector(W, H, { minArea: 200, threshold: webcam ? 22 : 26 });
  hint.textContent = webcam ? 'Live webcam — frames stay on your device.'
    : 'Synthetic feed. Drag to place a counting line.';
}

setMode('line');
requestAnimationFrame(frame);
