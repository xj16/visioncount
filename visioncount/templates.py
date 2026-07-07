"""HTML template for the VisionCount dashboard.

Kept as a module-level constant (rendered with Flask's ``render_template_string``)
so the app ships as a single importable package with no template-folder wiring.

The dashboard is fully interactive and dependency-free: an overlay canvas lets you
click-drag a counting line or click a polygon zone directly on the live feed, and
the crossings chart is drawn as inline SVG -- no chart library, no CDN.
"""

from __future__ import annotations

INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>VisionCount Dashboard</title>
<style>
  :root {
    --bg:#0b1f2e; --panel:#123141; --card:#ffffff; --ink:#0f2230; --muted:#5f7488;
    --accent:#1f7a8c; --accent2:#f2994a; --line:#e6edf1; --in:#1f7a8c; --out:#e07a3f;
    --good:#2ea36b; --warn:#e0a83f; --bad:#d0492f; --zone:#8a5cf0;
  }
  * { box-sizing: border-box; }
  body { margin:0; font-family:"Segoe UI", system-ui, -apple-system, sans-serif;
         color:var(--ink); background:#eef3f6; }
  header { background:linear-gradient(120deg,#0b1f2e,#164058); color:#fff;
           padding:18px 26px; display:flex; align-items:center; gap:16px; flex-wrap:wrap;}
  header h1 { margin:0; font-size:21px; letter-spacing:.3px; font-weight:700; }
  header .tag { opacity:.82; font-size:13px; }
  header code { background:rgba(255,255,255,.14); padding:1px 7px; border-radius:5px; }
  #healthPill { margin-left:auto; font-size:12px; font-weight:600; padding:5px 12px;
                border-radius:999px; background:#1f7a8c; color:#fff; letter-spacing:.3px;}
  #healthPill.ok{background:var(--good);} #healthPill.starting{background:#4a7a9c;}
  #healthPill.stale{background:var(--warn);} #healthPill.degraded{background:var(--bad);}
  #errBanner{display:none; background:#fbecea; color:#8a2c1a; border-left:4px solid var(--bad);
             padding:10px 26px; font-size:13px;}
  .wrap { max-width:1180px; margin:20px auto; padding:0 18px;
          display:grid; grid-template-columns:1.65fr 1fr; gap:18px; }
  @media (max-width:900px){ .wrap{ grid-template-columns:1fr; } }
  .card { background:var(--card); border-radius:14px; box-shadow:0 3px 14px rgba(11,31,46,.09);
          padding:16px; }
  .card h2 { margin:0 0 12px; font-size:12px; text-transform:uppercase; letter-spacing:.7px;
             color:var(--muted); display:flex; align-items:center; gap:8px;}
  .feedwrap { position:relative; border-radius:10px; overflow:hidden; background:#000;
              aspect-ratio: 4 / 3; }
  img.feed { width:100%; display:block; }
  #overlay { position:absolute; inset:0; width:100%; height:100%; cursor:crosshair; }
  .toolbar { display:flex; gap:8px; margin-top:12px; flex-wrap:wrap; align-items:center; }
  .toolbar button { font:inherit; font-size:13px; border:1px solid var(--line);
    background:#fff; color:var(--ink); padding:7px 13px; border-radius:8px; cursor:pointer;
    transition:none; }
  .toolbar button.active { background:var(--accent); color:#fff; border-color:var(--accent); }
  .toolbar button.ghost { color:var(--muted); }
  .toolbar .hint { color:var(--muted); font-size:12px; margin-left:auto; }
  .stat { display:flex; align-items:baseline; gap:10px; margin-bottom:6px; }
  .stat .num { font-size:36px; font-weight:800; color:var(--accent);
               font-variant-numeric:tabular-nums; }
  .stat .lbl { color:var(--muted); font-size:13px; }
  table { width:100%; border-collapse:collapse; font-size:13.5px;
          font-variant-numeric:tabular-nums; }
  th,td { text-align:left; padding:7px 6px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase;
       letter-spacing:.4px;}
  td.n{ text-align:right; }
  .swatch{ display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:7px;
           vertical-align:middle; }
  .pill { display:inline-block; padding:2px 9px; border-radius:999px; font-size:12px;
          background:#e6f2f4; color:var(--accent); }
  .x { color:var(--muted); cursor:pointer; font-weight:700; padding:0 4px; }
  .x:hover{ color:var(--bad); }
  #chart { width:100%; height:120px; display:block; }
  .mut{ color:var(--muted); font-size:12px; }
  #events { max-height:190px; overflow:auto; font-size:12.5px; }
  #events .ev { display:flex; gap:8px; padding:4px 2px; border-bottom:1px solid var(--line);
                font-variant-numeric:tabular-nums; }
  #events .dir-in{ color:var(--in); font-weight:700; } #events .dir-out{ color:var(--out); font-weight:700; }
  .grid2{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  @media (max-width:520px){ .grid2{ grid-template-columns:1fr; } }
  a.dl{ font-size:12px; color:var(--accent); text-decoration:none; margin-left:auto; }
  a.dl:hover{ text-decoration:underline; }
  footer { text-align:center; color:var(--muted); font-size:12px; padding:20px; }
</style>
</head>
<body>
<header>
  <h1>VisionCount</h1>
  <span class="tag">Real-time counting &amp; line-crossing analytics &middot; source: <code>{{ feed_source }}</code></span>
  <span id="healthPill">…</span>
</header>
<div id="errBanner"></div>
<div class="wrap">
  <div class="card">
    <h2>Live feed</h2>
    <div class="feedwrap">
      <img class="feed" src="/video_feed" alt="live annotated feed"/>
      <canvas id="overlay"></canvas>
    </div>
    <div class="toolbar">
      <button id="mLine" class="active" data-mode="line">Draw line</button>
      <button id="mZone" data-mode="zone">Draw zone</button>
      <button id="mNone" data-mode="none" class="ghost">Off</button>
      <button id="finishZone" class="ghost" style="display:none;">Finish zone</button>
      <span class="hint" id="hint">Drag on the video to add a counting line.</span>
    </div>
  </div>
  <div>
    <div class="card" style="margin-bottom:18px;">
      <h2>Totals</h2>
      <div class="stat"><span class="num" id="grand">0</span><span class="lbl">total crossings</span></div>
      <div class="stat"><span class="num" id="fps" style="font-size:19px;color:var(--muted);">0</span><span class="lbl">fps</span></div>
      <table id="lines"><thead><tr><th>Line</th><th class="n">In</th><th class="n">Out</th><th class="n">Net</th><th></th></tr></thead><tbody></tbody></table>
    </div>
    <div class="card" style="margin-bottom:18px;">
      <h2>Zones <span class="mut" id="zoneEmpty">— draw one on the feed</span></h2>
      <table id="zones"><thead><tr><th>Zone</th><th class="n">Now</th><th class="n">Peak</th><th class="n">Entries</th></tr></thead><tbody></tbody></table>
    </div>
    <div class="card" style="margin-bottom:18px;">
      <h2>Crossings over time</h2>
      <svg id="chart" viewBox="0 0 320 120" preserveAspectRatio="none"></svg>
    </div>
    <div class="card">
      <h2>Recent crossings <a class="dl" href="/export.csv" download>Export CSV ↓</a></h2>
      <div id="events"><div class="mut">waiting for crossings…</div></div>
    </div>
  </div>
</div>
<footer>VisionCount &middot; OpenCV background-subtraction pipeline &middot; MIT licensed</footer>
<script>
const $ = s => document.querySelector(s);
const LINE_COLORS = ['#1f7a8c','#e07a3f','#8a5cf0','#2ea36b','#d0492f','#e0a83f'];
let mode = 'line';
let drag = null;            // {x0,y0,x1,y1} while drawing a line
let zonePts = [];           // polygon-in-progress
let lineColors = {};        // name -> color
let colorIdx = 0;
const overlay = $('#overlay'), octx = overlay.getContext('2d');

function fit(){ const r = overlay.getBoundingClientRect();
  overlay.width = r.width; overlay.height = r.height; }
addEventListener('resize', fit);

// Map an overlay-pixel coord to feed pixels. The feed is 640x480 by default;
// we read the natural size off the <img> once it loads.
const img = $('.feed');
function feedDims(){ return [img.naturalWidth||640, img.naturalHeight||480]; }
function toFeed(px, py){
  const [fw,fh] = feedDims();
  return [Math.round(px/overlay.width*fw), Math.round(py/overlay.height*fh)];
}

function setMode(m){
  mode = m; zonePts = []; drag = null;
  document.querySelectorAll('.toolbar button[data-mode]').forEach(b=>
    b.classList.toggle('active', b.dataset.mode===m));
  $('#finishZone').style.display = (m==='zone') ? '' : 'none';
  $('#hint').textContent = m==='line' ? 'Drag on the video to add a counting line.'
    : m==='zone' ? 'Click to add polygon points, then Finish zone.'
    : 'Drawing off.';
  redraw();
}
document.querySelectorAll('.toolbar button[data-mode]').forEach(b=>
  b.addEventListener('click', ()=>setMode(b.dataset.mode)));

overlay.addEventListener('mousedown', e=>{
  if(mode!=='line') return;
  const r = overlay.getBoundingClientRect();
  drag = {x0:e.clientX-r.left, y0:e.clientY-r.top, x1:e.clientX-r.left, y1:e.clientY-r.top};
});
overlay.addEventListener('mousemove', e=>{
  if(!drag) return;
  const r = overlay.getBoundingClientRect();
  drag.x1 = e.clientX-r.left; drag.y1 = e.clientY-r.top; redraw();
});
overlay.addEventListener('mouseup', async e=>{
  if(mode!=='line' || !drag){ drag=null; return; }
  const {x0,y0,x1,y1} = drag; drag=null;
  if(Math.hypot(x1-x0,y1-y0) < 8){ redraw(); return; }  // ignore tiny drags
  const s = toFeed(x0,y0), en = toFeed(x1,y1);
  const name = 'line'+Date.now().toString().slice(-4);
  await postJSON('/api/lines', {name, start:s, end:en});
  refresh();
});
overlay.addEventListener('click', e=>{
  if(mode!=='zone') return;
  const r = overlay.getBoundingClientRect();
  zonePts.push([e.clientX-r.left, e.clientY-r.top]); redraw();
});
$('#finishZone').addEventListener('click', async ()=>{
  if(zonePts.length < 3){ $('#hint').textContent='Need at least 3 points.'; return; }
  const poly = zonePts.map(p=>toFeed(p[0],p[1]));
  const name = 'zone'+Date.now().toString().slice(-4);
  await postJSON('/api/zones', {name, polygon:poly});
  zonePts=[]; refresh();
});

async function postJSON(url, body){
  try{
    const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)});
    if(!r.ok){ const d = await r.json().catch(()=>({})); $('#hint').textContent = d.error||('error '+r.status); }
  }catch(e){ $('#hint').textContent = 'network error'; }
}
async function delLine(name){
  await fetch('/api/lines?name='+encodeURIComponent(name), {method:'DELETE'}); refresh();
}

function colorFor(name){
  if(!(name in lineColors)) lineColors[name] = LINE_COLORS[(colorIdx++)%LINE_COLORS.length];
  return lineColors[name];
}

function redraw(){
  octx.clearRect(0,0,overlay.width,overlay.height);
  // in-progress line
  if(drag){ octx.strokeStyle='#ffd34d'; octx.lineWidth=3; octx.beginPath();
    octx.moveTo(drag.x0,drag.y0); octx.lineTo(drag.x1,drag.y1); octx.stroke(); }
  // in-progress zone
  if(zonePts.length){ octx.strokeStyle='#8a5cf0'; octx.fillStyle='rgba(138,92,240,.18)';
    octx.lineWidth=2; octx.beginPath();
    zonePts.forEach((p,i)=> i?octx.lineTo(p[0],p[1]):octx.moveTo(p[0],p[1]));
    if(zonePts.length>2) octx.closePath();
    octx.fill(); octx.stroke();
    zonePts.forEach(p=>{ octx.fillStyle='#8a5cf0'; octx.beginPath();
      octx.arc(p[0],p[1],4,0,7); octx.fill(); });
  }
}

function drawChart(series){
  const svg = $('#chart'); const W=320,H=120;
  if(!series.length){ svg.innerHTML=''; return; }
  const vals = series.map(p=>p.total); const max = Math.max(1,...vals);
  const pts = series.map((p,i)=>{
    const x=(i/(series.length-1||1))*W;
    const y=H-(p.total/max)*(H-10)-5; return [x,y];
  });
  const line = pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' ');
  const area = line+` L${W} ${H} L0 ${H} Z`;
  svg.innerHTML =
    `<path d="${area}" fill="rgba(31,122,140,.12)"/>`+
    `<path d="${line}" fill="none" stroke="#1f7a8c" stroke-width="2"/>`+
    `<circle cx="${pts[pts.length-1][0].toFixed(1)}" cy="${pts[pts.length-1][1].toFixed(1)}" r="3" fill="#1f7a8c"/>`;
}

async function refresh(){
  try{
    const r = await fetch('/api/counts'); const d = await r.json();
    $('#grand').textContent = d.grand_total;
    $('#fps').textContent = d.fps;
    const tb = $('#lines tbody'); tb.innerHTML='';
    for(const [name,c] of Object.entries(d.lines)){
      const col = colorFor(name);
      const tr = document.createElement('tr');
      tr.innerHTML = `<td><span class="swatch" style="background:${col}"></span>${name}</td>`+
        `<td class="n">${c.in}</td><td class="n">${c.out}</td><td class="n">${c.net}</td>`+
        `<td class="n"><span class="x" title="remove">×</span></td>`;
      tr.querySelector('.x').addEventListener('click',()=>delLine(name));
      tb.appendChild(tr);
    }
    const zb = $('#zones tbody'); zb.innerHTML='';
    const zEntries = Object.entries(d.zones||{});
    $('#zoneEmpty').style.display = zEntries.length ? 'none' : '';
    for(const [name,z] of zEntries){
      const tr=document.createElement('tr');
      tr.innerHTML=`<td><span class="swatch" style="background:var(--zone)"></span>${name}</td>`+
        `<td class="n">${z.occupancy}</td><td class="n">${z.peak}</td><td class="n">${z.entries}</td>`;
      zb.appendChild(tr);
    }
    drawChart(d.series);
  }catch(e){}
}
async function refreshEvents(){
  try{
    const r = await fetch('/api/events?limit=12'); const d = await r.json();
    const el = $('#events');
    if(!d.events.length){ el.innerHTML='<div class="mut">waiting for crossings…</div>'; return; }
    el.innerHTML = d.events.map(e=>
      `<div class="ev"><span class="mut">${e.iso.slice(11,19)}</span>`+
      `<span class="dir-${e.direction}">${e.direction.toUpperCase()}</span>`+
      `<span>${e.line}</span><span class="mut">#${e.track_id}</span></div>`).join('');
  }catch(e){}
}
async function refreshHealth(){
  try{
    const r = await fetch('/api/health'); const h = await r.json();
    const pill = $('#healthPill'); pill.className=''; pill.classList.add(h.status);
    pill.textContent = h.status.toUpperCase();
    const b = $('#errBanner');
    if(h.status==='degraded' && h.error){ b.style.display='block';
      b.textContent = 'Video worker error: '+h.error; }
    else if(h.status==='stale'){ b.style.display='block';
      b.textContent = 'Feed is stale — no new frame in over 5s.'; }
    else b.style.display='none';
  }catch(e){}
}

img.addEventListener('load', fit);
fit(); setMode('line');
setInterval(refresh, 1000); refresh();
setInterval(refreshEvents, 1500); refreshEvents();
setInterval(refreshHealth, 2000); refreshHealth();
</script>
</body>
</html>
"""
