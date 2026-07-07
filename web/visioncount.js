/*
 * visioncount.js — a faithful, dependency-free browser port of the VisionCount
 * core pipeline (detector → centroid tracker → line counter → heatmap → zones).
 *
 * The Python package runs OpenCV MOG2 + a NumPy centroid tracker on a server and
 * streams MJPEG. That architecture cannot run in a browser tab. This module is a
 * clean re-implementation of the *same algorithms* in pure JavaScript so the exact
 * behaviour — cross-product line direction, greedy nearest-centroid association
 * with velocity prediction, decaying motion heatmap, point-in-polygon occupancy —
 * runs fully client-side on a <canvas>, with zero network and zero dependencies.
 *
 * Detection here uses frame differencing (a lightweight cousin of background
 * subtraction) which is a natural fit for canvas ImageData; the tracking, counting
 * and heatmap logic below mirrors visioncount/tracker.py, counter.py, heatmap.py
 * line for line.
 */

// ---- geometry (mirrors counter.py) ---------------------------------------

function crossSign(a, b, p) {
  return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
}

function onSegment(a, b, c) {
  return (
    Math.min(a[0], b[0]) <= c[0] && c[0] <= Math.max(a[0], b[0]) &&
    Math.min(a[1], b[1]) <= c[1] && c[1] <= Math.max(a[1], b[1])
  );
}

export function segmentsIntersect(p1, p2, p3, p4) {
  const d1 = crossSign(p3, p4, p1);
  const d2 = crossSign(p3, p4, p2);
  const d3 = crossSign(p1, p2, p3);
  const d4 = crossSign(p1, p2, p4);
  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
      ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;
  if (d1 === 0 && onSegment(p3, p4, p1)) return true;
  if (d2 === 0 && onSegment(p3, p4, p2)) return true;
  if (d3 === 0 && onSegment(p1, p2, p3)) return true;
  if (d4 === 0 && onSegment(p1, p2, p4)) return true;
  return false;
}

export function pointInPolygon(pt, poly) {
  const n = poly.length;
  if (n < 3) return false;
  let inside = false;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > pt[1]) !== (yj > pt[1])) {
      const xCross = ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi;
      if (pt[0] < xCross) inside = !inside;
    }
  }
  return inside;
}

// ---- detector: frame differencing ----------------------------------------

export class MotionDetector {
  constructor(width, height, { minArea = 220, threshold = 28 } = {}) {
    this.w = width;
    this.h = height;
    this.minArea = minArea;
    this.threshold = threshold;
    this.prevGray = null;
    this._labels = new Int32Array(width * height);
  }

  // Returns [{x,y,w,h,cx,cy,area}] for connected moving regions.
  detect(imageData) {
    const { w, h } = this;
    const data = imageData.data;
    const gray = new Uint8ClampedArray(w * h);
    for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
      gray[i] = (data[p] * 0.114 + data[p + 1] * 0.587 + data[p + 2] * 0.299) | 0;
    }
    if (!this.prevGray) { this.prevGray = gray; return []; }

    // Binary motion mask.
    const mask = new Uint8Array(w * h);
    const th = this.threshold;
    for (let i = 0; i < gray.length; i++) {
      if (Math.abs(gray[i] - this.prevGray[i]) > th) mask[i] = 1;
    }
    // Blend the reference so slow lighting drift does not trigger motion.
    for (let i = 0; i < gray.length; i++) {
      this.prevGray[i] = (this.prevGray[i] * 0.6 + gray[i] * 0.4) | 0;
    }

    return this._connectedComponents(mask);
  }

  // Iterative flood-fill labelling (no recursion, so big blobs never overflow).
  _connectedComponents(mask) {
    const { w, h } = this;
    const labels = this._labels;
    labels.fill(0);
    const out = [];
    const stack = [];
    let next = 1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        if (!mask[idx] || labels[idx]) continue;
        const id = next++;
        let minX = x, maxX = x, minY = y, maxY = y, area = 0, sx = 0, sy = 0;
        stack.length = 0; stack.push(idx); labels[idx] = id;
        while (stack.length) {
          const cur = stack.pop();
          const cy = (cur / w) | 0, cx = cur - cy * w;
          area++; sx += cx; sy += cy;
          if (cx < minX) minX = cx; if (cx > maxX) maxX = cx;
          if (cy < minY) minY = cy; if (cy > maxY) maxY = cy;
          if (cx > 0 && mask[cur - 1] && !labels[cur - 1]) { labels[cur - 1] = id; stack.push(cur - 1); }
          if (cx < w - 1 && mask[cur + 1] && !labels[cur + 1]) { labels[cur + 1] = id; stack.push(cur + 1); }
          if (cy > 0 && mask[cur - w] && !labels[cur - w]) { labels[cur - w] = id; stack.push(cur - w); }
          if (cy < h - 1 && mask[cur + w] && !labels[cur + w]) { labels[cur + w] = id; stack.push(cur + w); }
        }
        if (area >= this.minArea) {
          out.push({
            x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1,
            cx: Math.round(sx / area), cy: Math.round(sy / area), area,
          });
        }
      }
    }
    return out;
  }
}

// ---- centroid tracker (mirrors tracker.py, incl. velocity prediction) -----

export class CentroidTracker {
  constructor({ maxDisappeared = 12, maxDistance = 70, velSmooth = 0.5 } = {}) {
    this.maxDisappeared = maxDisappeared;
    this.maxDistance = maxDistance;
    this.velSmooth = velSmooth;
    this.nextId = 0;
    this.tracks = new Map(); // id -> {id, cx, cy, vx, vy, missing, traj}
  }

  update(dets) {
    if (dets.length === 0) {
      for (const [id, t] of this.tracks) {
        if (++t.missing > this.maxDisappeared) this.tracks.delete(id);
      }
      return this.tracks;
    }
    if (this.tracks.size === 0) {
      for (const d of dets) this._register(d);
      return this.tracks;
    }

    const ids = [...this.tracks.keys()];
    // Cost = distance from each detection to each track's *predicted* position.
    const pairs = [];
    for (let r = 0; r < ids.length; r++) {
      const t = this.tracks.get(ids[r]);
      const px = t.cx + t.vx, py = t.cy + t.vy;
      for (let c = 0; c < dets.length; c++) {
        const dx = px - dets[c].cx, dy = py - dets[c].cy;
        pairs.push([Math.hypot(dx, dy), r, c]);
      }
    }
    pairs.sort((a, b) => a[0] - b[0]);
    const usedR = new Set(), usedC = new Set();
    for (const [dist, r, c] of pairs) {
      if (usedR.has(r) || usedC.has(c) || dist > this.maxDistance) continue;
      const t = this.tracks.get(ids[r]), d = dets[c];
      const vx = d.cx - t.cx, vy = d.cy - t.cy;
      t.vx = this.velSmooth * t.vx + (1 - this.velSmooth) * vx;
      t.vy = this.velSmooth * t.vy + (1 - this.velSmooth) * vy;
      t.prevCx = t.cx; t.prevCy = t.cy;
      t.cx = d.cx; t.cy = d.cy; t.bbox = d; t.missing = 0;
      t.traj.push([d.cx, d.cy]); if (t.traj.length > 24) t.traj.shift();
      usedR.add(r); usedC.add(c);
    }
    for (let r = 0; r < ids.length; r++) {
      if (!usedR.has(r)) {
        const t = this.tracks.get(ids[r]);
        if (++t.missing > this.maxDisappeared) this.tracks.delete(ids[r]);
      }
    }
    for (let c = 0; c < dets.length; c++) if (!usedC.has(c)) this._register(dets[c]);
    return this.tracks;
  }

  _register(d) {
    const id = this.nextId++;
    this.tracks.set(id, {
      id, cx: d.cx, cy: d.cy, prevCx: d.cx, prevCy: d.cy,
      vx: 0, vy: 0, missing: 0, bbox: d, traj: [[d.cx, d.cy]],
    });
  }
}

// ---- line counter (mirrors counter.py) -----------------------------------

export class CountingLine {
  constructor(name, start, end) {
    this.name = name; this.start = start; this.end = end;
    this.in = 0; this.out = 0; this.counted = new Set();
  }
  get total() { return this.in + this.out; }
  get net() { return this.in - this.out; }
  sideOf(p) { return crossSign(this.start, this.end, p); }
}

export class LineCounter {
  constructor(lines = []) { this.lines = lines; }
  add(line) { this.lines.push(line); return line; }
  remove(name) { this.lines = this.lines.filter((l) => l.name !== name); }
  update(tracks) {
    const events = [];
    for (const t of tracks.values()) {
      if (t.prevCx === undefined) continue;
      const prev = [t.prevCx, t.prevCy], curr = [t.cx, t.cy];
      for (const line of this.lines) {
        if (line.counted.has(t.id)) continue;
        if (!segmentsIntersect(prev, curr, line.start, line.end)) continue;
        const sp = line.sideOf(prev), sc = line.sideOf(curr);
        if (sp === sc) continue;
        const direction = sp < 0 ? 'in' : 'out';
        if (direction === 'in') line.in++; else line.out++;
        line.counted.add(t.id);
        events.push({ line: line.name, trackId: t.id, direction, point: curr });
      }
    }
    return events;
  }
}

// ---- heatmap (mirrors heatmap.py) ----------------------------------------

export class Heatmap {
  constructor(width, height, { decay = 0.985, radius = 14 } = {}) {
    this.w = width; this.h = height; this.decay = decay; this.radius = radius;
    this.buf = new Float32Array(width * height);
    this.stamp = this._makeStamp(radius);
  }
  _makeStamp(r) {
    const size = 2 * r + 1, s = new Float32Array(size * size);
    const sigma = Math.max(1, r / 2);
    for (let y = -r; y <= r; y++)
      for (let x = -r; x <= r; x++)
        s[(y + r) * size + (x + r)] = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
    return s;
  }
  addPoints(pts) {
    if (this.decay < 1) for (let i = 0; i < this.buf.length; i++) this.buf[i] *= this.decay;
    const r = this.radius, size = 2 * r + 1;
    for (const [px, py] of pts) {
      for (let y = -r; y <= r; y++) {
        const yy = py + y; if (yy < 0 || yy >= this.h) continue;
        for (let x = -r; x <= r; x++) {
          const xx = px + x; if (xx < 0 || xx >= this.w) continue;
          this.buf[yy * this.w + xx] += this.stamp[(y + r) * size + (x + r)];
        }
      }
    }
  }
  // Paint the heatmap onto a canvas 2D context using a JET-like colour ramp.
  render(ctx, alpha = 0.45) {
    let peak = 0;
    for (let i = 0; i < this.buf.length; i++) if (this.buf[i] > peak) peak = this.buf[i];
    if (peak < 1e-6) return;
    const img = ctx.createImageData(this.w, this.h);
    const d = img.data;
    for (let i = 0, p = 0; i < this.buf.length; i++, p += 4) {
      const v = Math.min(1, this.buf[i] / peak);
      if (v < 0.05) { d[p + 3] = 0; continue; }
      const [r, g, b] = jet(v);
      d[p] = r; d[p + 1] = g; d[p + 2] = b; d[p + 3] = (alpha * 255 * v) | 0;
    }
    ctx.putImageData(img, 0, 0);
  }
}

function jet(v) {
  // Compact JET colormap approximation in [0,1] -> [r,g,b].
  const four = 4 * v;
  const r = Math.min(four - 1.5, -four + 4.5);
  const g = Math.min(four - 0.5, -four + 3.5);
  const b = Math.min(four + 0.5, -four + 2.5);
  const clamp = (x) => Math.round(255 * Math.max(0, Math.min(1, x)));
  return [clamp(r), clamp(g), clamp(b)];
}

// ---- zone (mirrors zone.py) ----------------------------------------------

export class Zone {
  constructor(name, polygon) {
    this.name = name; this.polygon = polygon;
    this.peak = 0; this.entries = 0; this.occupancy = 0;
    this._inside = new Set();
  }
  update(tracks) {
    const now = new Set();
    for (const t of tracks.values()) {
      if (pointInPolygon([t.cx, t.cy], this.polygon)) {
        now.add(t.id);
        if (!this._inside.has(t.id)) this.entries++;
      }
    }
    this._inside = now;
    this.occupancy = now.size;
    if (this.occupancy > this.peak) this.peak = this.occupancy;
    return this.occupancy;
  }
}

// ---- palette shared with the dashboard -----------------------------------

export const PALETTE = ['#42a5f5', '#66bb6a', '#ef5350', '#ffca28', '#ab47bc', '#26c6da', '#ff7043'];
