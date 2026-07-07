/*
 * scene.js — a browser synthetic feed, mirroring visioncount/synthetic.py.
 *
 * Draws moving blobs on a mildly textured background so the frame-differencing
 * detector has real motion to find. Deterministic given a seed, so the demo looks
 * the same on every load (until the viewer switches to their webcam).
 */

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class SyntheticScene {
  constructor(width, height, seed = 7) {
    this.w = width; this.h = height;
    const rng = mulberry32(seed);
    const cy = height / 2;
    this.objects = [
      { x: -20, y: cy - 55, vx: 3.1, vy: 0.15, r: 20, color: '#3ec85a' },
      { x: -20, y: cy + 40, vx: 2.4, vy: -0.1, r: 26, color: '#c8783c' },
      { x: width + 20, y: cy, vx: -2.7, vy: 0.05, r: 18, color: '#5060dc' },
      { x: width * 0.4, y: -20, vx: 0.1, vy: 2.2, r: 16, color: '#d84c8c' },
    ];
    // Pre-render a static textured background once.
    this.bg = document.createElement('canvas');
    this.bg.width = width; this.bg.height = height;
    const bctx = this.bg.getContext('2d');
    const img = bctx.createImageData(width, height);
    for (let i = 0, p = 0; i < width * height; i++, p += 4) {
      const base = 34 + rng() * 16;
      img.data[p] = base; img.data[p + 1] = base + 4; img.data[p + 2] = base + 8;
      img.data[p + 3] = 255;
    }
    bctx.putImageData(img, 0, 0);
    this._rng = rng;
  }

  step(ctx) {
    ctx.drawImage(this.bg, 0, 0);
    // Light per-frame noise so differencing has to work a little.
    for (const o of this.objects) {
      o.x += o.vx; o.y += o.vy;
      if (o.x < -o.r) o.x = this.w + o.r;
      else if (o.x > this.w + o.r) o.x = -o.r;
      if (o.y < -o.r) o.y = this.h + o.r;
      else if (o.y > this.h + o.r) o.y = -o.r;
      ctx.fillStyle = o.color;
      ctx.beginPath(); ctx.arc(o.x, o.y, o.r, 0, Math.PI * 2); ctx.fill();
    }
  }
}
