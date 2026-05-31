"""
Premium 3D data center visualization — Three.js / WebGL.

Self-contained HTML embedded via st.components.v1.html().
Uses Three.js r128 from CDN (global THREE object, no ES modules).
Manual orbit controls — no OrbitControls import needed.

Features:
- Temperature-gradient coloured racks (blue → teal → green → amber → red)
- Rack height ∝ power draw
- Flat CRAC units on room perimeters with glow lights
- Hot / cold aisle floor strips
- Pulsing alert edges on hot racks, breathing on critical racks
- Raycaster hover tooltip
- LIVE data polling every 5 s with hash-based diff
- Auto-rotate (pauses on mouse interaction, resumes after 3 s)
- Touch support (drag=rotate, pinch=zoom)
- "API unavailable" overlay on fetch error
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────
# Layout constants (must match viz3d.py)
# ─────────────────────────────────────────
RACK_PITCH = 1.20
ROW_PITCH  = 3.00
ROOM_B_X   = 18.0
N_ROWS     = 4
N_RACKS    = 10
MIN_H      = 0.30
MAX_H      = 3.00

# Three.js specific dimensions
RACK_W_JS  = 0.70
RACK_D_JS  = 0.70
CRAC_W_JS  = 1.80
CRAC_D_JS  = 1.20
CRAC_H_JS  = 0.30

# Fixed CRAC centres (x, z) — matches viz3d.py _CRAC_POS
_CRAC_FIXED: List[Tuple[str, float, float]] = [
    ("C-A1",  5.50,  -1.25),
    ("C-A2",  5.50,  11.25),
    ("C-A3", -1.25,   4.90),
    ("C-A4", 12.25,   4.90),
    ("C-B1",  ROOM_B_X + 5.50,  -1.25),
    ("C-B2",  ROOM_B_X + 5.50,  11.25),
    ("C-B3",  ROOM_B_X - 1.25,   4.90),
    ("C-B4",  ROOM_B_X + 12.25,  4.90),
]

_THREE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"


def _rack_pos(room: str, row: str, position: int) -> Dict:
    room_idx = 0 if "A" in room else 1
    row_idx  = int(row.split()[-1]) - 1
    pos_idx  = int(position) - 1
    return {
        "x": (ROOM_B_X if room_idx else 0.0) + pos_idx * RACK_PITCH,
        "z": row_idx * ROW_PITCH,
    }


def _build_data(telemetry: Optional[Dict]) -> Tuple[list, list]:
    racks_js: list = []
    cracs_js: list = []
    t = telemetry or {}

    for r in t.get("racks", []):
        pos = _rack_pos(r.get("room", "Room A"),
                        r.get("row", "Row 1"),
                        r.get("position", 1))
        racks_js.append({
            "id":     r.get("asset_id", ""),
            "x":      pos["x"],
            "z":      pos["z"],
            "temp":   float(r.get("inlet_temp_c", 22)),
            "outlet": float(r.get("outlet_temp_c", 32)),
            "power":  float(r.get("power_kw", 0)),
            "humid":  float(r.get("humidity_pct", 50)),
            "ashrae": str(r.get("ashrae_class", "A1")),
            "room":   r.get("room", ""),
            "row":    r.get("row", ""),
        })

    # Use fixed CRAC positions; enrich with API data when available
    api_a = [c for c in t.get("cracs", []) if "A" in c.get("room", "")]
    api_b = [c for c in t.get("cracs", []) if "B" in c.get("room", "")]
    api_map: Dict[str, Dict] = {}
    for i, c in enumerate(api_a[:4]):
        api_map[f"C-A{i+1}"] = c
    for i, c in enumerate(api_b[:4]):
        api_map[f"C-B{i+1}"] = c

    for label, cx, cz in _CRAC_FIXED:
        api_c = api_map.get(label, {})
        cracs_js.append({
            "id":      label,
            "real_id": api_c.get("asset_id", label),
            "x":       cx,
            "z":       cz,
            "supply":  float(api_c.get("supply_air_temp_c", 18)),
            "ret":     float(api_c.get("return_air_temp_c", 28)),
            "dt":      float(api_c.get("delta_t_c", 10)),
            "cooling": float(api_c.get("cooling_load_kw", 60)),
            "room":    api_c.get("room", ""),
        })

    return racks_js, cracs_js


def make_three_html(
    telemetry: Optional[Dict],
    width: int = 1200,
    height: int = 720,
) -> str:
    """Return a self-contained HTML page embedding the Three.js 3D scene."""

    racks_data, cracs_data = _build_data(telemetry)
    racks_json = json.dumps(racks_data)
    cracs_json = json.dumps(cracs_data)

    # Floor bounds
    floor_x0 = -2.5
    floor_x1 = ROOM_B_X + (N_RACKS - 1) * RACK_PITCH + RACK_W_JS + 2.5
    floor_z0 = -2.5
    floor_z1 = (N_ROWS - 1) * ROW_PITCH + RACK_D_JS + 3.5

    # Camera target = centre of whole floor plan
    cam_x = (floor_x0 + floor_x1) / 2
    cam_z = (floor_z0 + floor_z1) / 2

    # Room label world positions (above centre of each room)
    room_a_cx = (N_RACKS - 1) * RACK_PITCH / 2
    room_b_cx = ROOM_B_X + (N_RACKS - 1) * RACK_PITCH / 2
    room_cz   = (N_ROWS - 1) * ROW_PITCH / 2

    fw = floor_x1 - floor_x0
    fd = floor_z1 - floor_z0
    fx_mid = (floor_x0 + floor_x1) / 2
    fz_mid = (floor_z0 + floor_z1) / 2

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<script src="{_THREE_CDN}"></script>
<style>
  *{{ margin:0;padding:0;box-sizing:border-box; }}
  body{{background:#0a0f1a;overflow:hidden;width:{width}px;height:{height}px;}}
  canvas{{display:block;}}
  #container{{position:relative;width:{width}px;height:{height}px;}}

  /* ── Overlays ── */
  #dc-title{{
    position:absolute;top:10px;left:10px;
    color:#00aaff;font-family:'Courier New',monospace;
    font-size:13px;font-weight:bold;pointer-events:none;z-index:10;
    text-shadow:0 0 8px rgba(0,170,255,.45);
  }}
  #live-badge{{
    position:absolute;top:10px;right:10px;
    color:#c8d8e8;font-family:'Courier New',monospace;
    font-size:11px;z-index:10;pointer-events:none;
    display:flex;align-items:center;gap:5px;
  }}
  .live-dot{{color:#00ff44;font-size:15px;animation:blink 1.4s ease-in-out infinite;}}
  @keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.15}}}}

  #legend{{
    position:absolute;bottom:12px;left:10px;
    background:rgba(10,18,28,.88);border:1px solid #1a3050;border-radius:5px;
    padding:8px 12px;font-family:'Courier New',monospace;pointer-events:none;z-index:10;
    min-width:135px;
  }}
  #legend .ltitle{{color:#3a5a7a;font-size:8px;text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px;}}
  .lr{{display:flex;align-items:center;gap:6px;margin-top:4px;}}
  .sw{{width:11px;height:11px;border-radius:2px;flex-shrink:0;}}
  .lr span{{color:#6a8aaa;font-size:9px;}}

  #hint{{
    position:absolute;bottom:12px;right:10px;
    color:#2a4a6a;font-family:'Courier New',monospace;font-size:9px;
    pointer-events:none;z-index:10;text-align:right;line-height:1.65;
  }}
  #updated{{
    position:absolute;top:30px;right:10px;
    color:#2a4a6a;font-family:'Courier New',monospace;font-size:9px;
    pointer-events:none;z-index:10;
  }}

  #tooltip{{
    position:absolute;display:none;
    background:rgba(10,21,32,.96);border:1px solid #00aaff;border-radius:6px;
    padding:10px 14px;font-size:11px;line-height:1.65;color:#c8d8e8;
    pointer-events:none;max-width:230px;z-index:20;
    box-shadow:0 4px 16px rgba(0,0,0,.6);font-family:'Courier New',monospace;
  }}
  #tooltip b{{color:#e0f0ff;}}
  .t-hot{{color:#ff6666;}}.t-cool{{color:#00cc88;}}

  #api-error{{
    position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
    background:rgba(20,8,8,.92);border:1px solid #ff4444;border-radius:8px;
    padding:18px 24px;font-size:12px;color:#ff8888;
    font-family:'Courier New',monospace;z-index:20;text-align:center;display:none;
  }}
  .rl{{
    position:absolute;color:#00aaff;font-family:'Courier New',monospace;
    font-size:13px;font-weight:bold;pointer-events:none;z-index:10;
    text-shadow:0 0 6px rgba(0,170,255,.4);display:none;
    transform:translate(-50%,-50%);
  }}
</style>
</head>
<body>
<div id="container">
  <div id="dc-title">DC TWIN — 3D LIVE VIEW</div>
  <div id="live-badge"><span class="live-dot">&#9679;</span>&nbsp;LIVE</div>
  <div id="legend">
    <div class="ltitle">Rack Inlet Temp</div>
    <div class="lr"><div class="sw" style="background:#0066ff"></div><span>&lt;20°C</span></div>
    <div class="lr"><div class="sw" style="background:#00ccaa"></div><span>20–24°C</span></div>
    <div class="lr"><div class="sw" style="background:#00cc44"></div><span>24–27°C</span></div>
    <div class="lr"><div class="sw" style="background:#ffaa00"></div><span>27–30°C warm</span></div>
    <div class="lr"><div class="sw" style="background:#ff6600"></div><span>30–33°C hot</span></div>
    <div class="lr"><div class="sw" style="background:#ff2200"></div><span>&gt;33°C crit</span></div>
    <div class="lr"><div class="sw" style="background:#00ccaa;height:5px"></div><span>CRAC unit</span></div>
    <div class="lr"><div class="sw" style="background:#002266;opacity:.7"></div><span>Cold aisle</span></div>
    <div class="lr"><div class="sw" style="background:#550000;opacity:.7"></div><span>Hot aisle</span></div>
  </div>
  <div id="hint">Drag to rotate<br>Scroll to zoom<br>Hover for details</div>
  <div id="updated"></div>
  <div id="tooltip"></div>
  <div id="api-error">&#9888; API unavailable<br><small style="color:#aa6666">Start: uvicorn backend.main:app --port 8000</small></div>
  <div id="rl-a" class="rl">ROOM A</div>
  <div id="rl-b" class="rl">ROOM B</div>
</div>

<script>
if (typeof THREE === 'undefined') {{
  var e = document.getElementById('api-error');
  e.innerHTML = '&#9888; Three.js CDN unavailable<br><small style="color:#aa6666">Check network</small>';
  e.style.display = 'block';
  throw new Error('THREE not loaded');
}}

// ── Embedded data ─────────────────────────────────────────────────────────────
const RACKS = {racks_json};
const CRACS = {cracs_json};

// ── Constants ─────────────────────────────────────────────────────────────────
const RACK_W  = {RACK_W_JS};
const RACK_D  = {RACK_D_JS};
const CRAC_W  = {CRAC_W_JS};
const CRAC_D  = {CRAC_D_JS};
const CRAC_H  = {CRAC_H_JS};
const MIN_H   = {MIN_H};
const MAX_H   = {MAX_H};
const W       = {width};
const H_VIEW  = {height};
const ROW_PITCH  = {ROW_PITCH};
const ROOM_B_X   = {ROOM_B_X};
const N_ROWS_C   = {N_ROWS};
const N_RACKS_C  = {N_RACKS};
const RACK_PITCH = {RACK_PITCH};

// ── Scene ─────────────────────────────────────────────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0f1a);
scene.fog = new THREE.FogExp2(0x0a0f1a, 0.007);

const camera = new THREE.PerspectiveCamera(60, W / H_VIEW, 0.1, 1000);

const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
renderer.setSize(W, H_VIEW);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.getElementById('container').insertBefore(renderer.domElement, document.getElementById('dc-title'));

// ── Lighting ──────────────────────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0xffffff, 0.4));
var sun = new THREE.DirectionalLight(0xffffff, 0.8);
sun.position.set(22, 35, 18);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = sun.shadow.camera.bottom = -55;
sun.shadow.camera.right = sun.shadow.camera.top   =  55;
scene.add(sun);
scene.add(new THREE.HemisphereLight(0x223355, 0x0a0f1a, 0.35));

// ── Manual orbit controls ─────────────────────────────────────────────────────
var lookAtPt = new THREE.Vector3({cam_x:.2f}, 0, {cam_z:.2f});
var sph = {{ theta: 0.9, phi: 0.85, radius: 52 }};
var isDragging = false;
var lastMX = 0, lastMY = 0;
var autoRot = true;
var lastActivity = Date.now();

function updateCam() {{
  var sinP = Math.sin(sph.phi), cosP = Math.cos(sph.phi);
  camera.position.set(
    lookAtPt.x + sph.radius * sinP * Math.sin(sph.theta),
    lookAtPt.y + sph.radius * cosP,
    lookAtPt.z + sph.radius * sinP * Math.cos(sph.theta)
  );
  camera.lookAt(lookAtPt);
}}
updateCam();

renderer.domElement.addEventListener('mousedown', function(e) {{
  isDragging = true; autoRot = false; lastActivity = Date.now();
  lastMX = e.clientX; lastMY = e.clientY;
}});
document.addEventListener('mouseup', function() {{ isDragging = false; }});
document.addEventListener('mousemove', function(e) {{
  if (!isDragging) return;
  sph.theta -= (e.clientX - lastMX) * 0.008;
  sph.phi    = Math.max(0.12, Math.min(Math.PI * 0.47, sph.phi + (e.clientY - lastMY) * 0.008));
  lastMX = e.clientX; lastMY = e.clientY;
  lastActivity = Date.now();
  updateCam();
}});
renderer.domElement.addEventListener('wheel', function(e) {{
  e.preventDefault();
  sph.radius = Math.max(10, Math.min(130, sph.radius + e.deltaY * 0.05));
  autoRot = false; lastActivity = Date.now();
  updateCam();
}}, {{ passive: false }});

// Touch
var lastTouches = [];
renderer.domElement.addEventListener('touchstart', function(e) {{
  isDragging = true; autoRot = false; lastActivity = Date.now();
  lastTouches = Array.from(e.touches).map(function(t) {{ return {{x:t.clientX, y:t.clientY}}; }});
}});
renderer.domElement.addEventListener('touchend', function() {{ isDragging = false; }});
renderer.domElement.addEventListener('touchmove', function(e) {{
  e.preventDefault();
  var touches = Array.from(e.touches).map(function(t) {{ return {{x:t.clientX, y:t.clientY}}; }});
  if (touches.length === 1 && lastTouches.length >= 1) {{
    sph.theta -= (touches[0].x - lastTouches[0].x) * 0.009;
    sph.phi = Math.max(0.12, Math.min(Math.PI*0.47, sph.phi + (touches[0].y - lastTouches[0].y) * 0.009));
  }} else if (touches.length === 2 && lastTouches.length === 2) {{
    var d0 = Math.hypot(lastTouches[1].x-lastTouches[0].x, lastTouches[1].y-lastTouches[0].y);
    var d1 = Math.hypot(touches[1].x-touches[0].x, touches[1].y-touches[0].y);
    if (d1 > 0) sph.radius = Math.max(10, Math.min(130, sph.radius * d0 / d1));
  }}
  lastTouches = touches;
  lastActivity = Date.now();
  updateCam();
}}, {{ passive: false }});

// ── Floor ─────────────────────────────────────────────────────────────────────
var floorMesh = new THREE.Mesh(
  new THREE.PlaneGeometry({fw:.2f}, {fd:.2f}),
  new THREE.MeshLambertMaterial({{ color: 0x1a2332, transparent: true, opacity: 0.85 }})
);
floorMesh.rotation.x = -Math.PI / 2;
floorMesh.position.set({fx_mid:.2f}, 0, {fz_mid:.2f});
floorMesh.receiveShadow = true;
scene.add(floorMesh);

var grid = new THREE.GridHelper(Math.max({fw:.1f}, {fd:.1f}) * 1.1, 35, 0x1e3a5f, 0x0e1e2e);
grid.position.set({fx_mid:.2f}, 0.006, {fz_mid:.2f});
scene.add(grid);

// ── Aisle strips ──────────────────────────────────────────────────────────────
function aisleStrip(x0, z0, x1, z1, hexColor, opacity) {{
  var m = new THREE.Mesh(
    new THREE.PlaneGeometry(x1-x0, z1-z0),
    new THREE.MeshBasicMaterial({{ color: hexColor, transparent: true, opacity: opacity, depthWrite: false }})
  );
  m.rotation.x = -Math.PI / 2;
  m.position.set((x0+x1)/2, 0.012, (z0+z1)/2);
  scene.add(m);
}}
for (var rm = 0; rm < 2; rm++) {{
  var xOff = rm === 0 ? 0 : ROOM_B_X;
  var xa = xOff - 0.3, xb = xOff + (N_RACKS_C-1)*RACK_PITCH + RACK_W + 0.3;
  for (var ro = 0; ro < N_ROWS_C - 1; ro++) {{
    var za = ro*ROW_PITCH + RACK_D + 0.05;
    var zb = (ro+1)*ROW_PITCH - 0.05;
    aisleStrip(xa, za, xb, zb, ro%2===0 ? 0x002266 : 0x550000, 0.12);
  }}
}}

// ── Temperature → colour ──────────────────────────────────────────────────────
function tempColor(t) {{
  var stops = [
    [16, [0,102,255]],[20,[0,204,170]],[24,[0,204,68]],
    [27,[255,170,0]],[30,[255,102,0]],[33,[255,34,0]],[40,[255,0,0]]
  ];
  for (var i = 0; i < stops.length-1; i++) {{
    var t0=stops[i][0], c0=stops[i][1], t1=stops[i+1][0], c1=stops[i+1][1];
    if (t <= t1) {{
      var f = (t-t0)/(t1-t0);
      return new THREE.Color((c0[0]+f*(c1[0]-c0[0]))/255,(c0[1]+f*(c1[1]-c0[1]))/255,(c0[2]+f*(c1[2]-c0[2]))/255);
    }}
  }}
  return new THREE.Color(1,0,0);
}}

// ── Rack meshes ───────────────────────────────────────────────────────────────
var sharedRackGeo  = new THREE.BoxGeometry(RACK_W, 1.0, RACK_D);
var sharedEdgesGeo = new THREE.EdgesGeometry(sharedRackGeo);
var rackMap = {{}};
var rayTargets = [];
var hotRacks = [], critRacks = [];

RACKS.forEach(function(r) {{
  var h = Math.max(MIN_H, Math.min(r.power,20)/20 * MAX_H);
  var col  = tempColor(r.temp);
  var emis = r.temp > 28 ? col.clone().multiplyScalar(0.18) : new THREE.Color(0x020508);
  var mat  = new THREE.MeshLambertMaterial({{ color: col, emissive: emis }});
  var mesh = new THREE.Mesh(sharedRackGeo, mat);
  mesh.scale.y = h;
  mesh.position.set(r.x + RACK_W/2, h/2, r.z + RACK_D/2);
  mesh.castShadow = mesh.receiveShadow = true;
  mesh.userData = r;
  mesh.userData.origH = h;
  scene.add(mesh);
  rayTargets.push(mesh);
  rackMap[r.id] = {{ mesh: mesh }};

  // Normal edge wireframe
  mesh.add(new THREE.LineSegments(
    sharedEdgesGeo,
    new THREE.LineBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.15 }})
  ));

  // Alert edge overlay
  if (r.temp > 28 || r.power > 17) {{
    var ac = r.temp > 32 ? 0xff0000 : r.power > 17 && r.temp <= 28 ? 0xffaa00 : 0xff4444;
    var alertMat = new THREE.LineBasicMaterial({{ color: ac }});
    var alertEdge = new THREE.LineSegments(sharedEdgesGeo, alertMat);
    mesh.add(alertEdge);
    rackMap[r.id].alertEdge = alertEdge;
  }}
  if (r.temp > 28) hotRacks.push(rackMap[r.id]);
  if (r.temp > 32) critRacks.push(mesh);
}});

// ── CRAC boxes ────────────────────────────────────────────────────────────────
var cracGeo      = new THREE.BoxGeometry(CRAC_W, 1.0, CRAC_D);
var cracEdgesGeo = new THREE.EdgesGeometry(cracGeo);

CRACS.forEach(function(c) {{
  var mat  = new THREE.MeshLambertMaterial({{ color: 0x00ccaa, emissive: 0x003322 }});
  var mesh = new THREE.Mesh(cracGeo, mat);
  mesh.scale.y = CRAC_H;
  mesh.position.set(c.x, CRAC_H/2, c.z);
  mesh.castShadow = true;
  mesh.userData = c;
  scene.add(mesh);
  rayTargets.push(mesh);

  mesh.add(new THREE.LineSegments(
    cracEdgesGeo,
    new THREE.LineBasicMaterial({{ color: 0x00aa88, transparent: true, opacity: 0.7 }})
  ));

  var pl = new THREE.PointLight(0x00ccaa, 0.28, 8);
  pl.position.set(c.x, 1.6, c.z);
  scene.add(pl);
}});

// ── Raycaster / tooltip ───────────────────────────────────────────────────────
var raycaster = new THREE.Raycaster();
var mouseVec  = new THREE.Vector2();
var tooltip   = document.getElementById('tooltip');

renderer.domElement.addEventListener('mousemove', function(e) {{
  var rect = renderer.domElement.getBoundingClientRect();
  mouseVec.x = ((e.clientX - rect.left) / W) * 2 - 1;
  mouseVec.y = -((e.clientY - rect.top) / H_VIEW) * 2 + 1;
  raycaster.setFromCamera(mouseVec, camera);
  var hits = raycaster.intersectObjects(rayTargets);
  if (hits.length > 0) {{
    var d = hits[0].object.userData;
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX - rect.left + 14) + 'px';
    tooltip.style.top  = (e.clientY - rect.top  - 10) + 'px';
    if (d.supply !== undefined) {{
      tooltip.innerHTML = '<b>' + d.id + '</b>' + (d.real_id && d.real_id !== d.id ? ' (' + d.real_id + ')' : '') + '<br>' +
        'Supply: ' + d.supply.toFixed(1) + '&deg;C &nbsp; Return: ' + d.ret.toFixed(1) + '&deg;C<br>' +
        '&Delta;T: ' + d.dt.toFixed(1) + '&deg;C &nbsp; Cooling: ' + d.cooling.toFixed(1) + ' kW';
    }} else {{
      var cls = d.temp > 30 ? 't-hot' : 't-cool';
      tooltip.innerHTML =
        '<b>' + d.id + '</b><br>' +
        'Inlet: <span class="' + cls + '">' + d.temp.toFixed(1) + '&deg;C</span> &nbsp; Outlet: ' + d.outlet.toFixed(1) + '&deg;C<br>' +
        'Power: ' + d.power.toFixed(1) + ' kW<br>' +
        'ASHRAE: ' + (d.ashrae||'—') + '<br>' +
        d.room + ' &middot; ' + d.row;
    }}
  }} else {{
    tooltip.style.display = 'none';
  }}
}});
renderer.domElement.addEventListener('mouseleave', function() {{ tooltip.style.display = 'none'; }});

// ── Live polling ──────────────────────────────────────────────────────────────
var lastHash  = '';
var apiErrDiv = document.getElementById('api-error');
var updDiv    = document.getElementById('updated');

function hashData(data) {{
  if (!data || !data.racks) return '';
  return data.racks.map(function(r) {{
    return r.asset_id + ':' + r.inlet_temp_c.toFixed(1) + ':' + r.power_kw.toFixed(1);
  }}).join('|');
}}

function applyUpdate(racks) {{
  racks.forEach(function(r) {{
    var entry = rackMap[r.asset_id];
    if (!entry) return;
    var mesh = entry.mesh;
    var newH = Math.max(MIN_H, Math.min(r.power_kw,20)/20 * MAX_H);
    mesh.scale.y = newH;
    mesh.position.y = newH / 2;
    mesh.userData.temp  = r.inlet_temp_c;
    mesh.userData.power = r.power_kw;
    mesh.userData.origH = newH;
    mesh.material.color = tempColor(r.inlet_temp_c);
  }});
  updDiv.textContent = 'Last updated: ' + new Date().toLocaleTimeString();
}}

function poll() {{
  var ctrl = typeof AbortController !== 'undefined' ? new AbortController() : null;
  var tid  = ctrl ? setTimeout(function(){{ ctrl.abort(); }}, 4000) : null;
  var opts = ctrl ? {{ signal: ctrl.signal }} : {{}};
  fetch('http://localhost:8000/telemetry/live', opts)
    .then(function(r) {{
      if (tid) clearTimeout(tid);
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }})
    .then(function(data) {{
      var h = hashData(data);
      if (h !== lastHash) {{ lastHash = h; applyUpdate(data.racks || []); }}
      apiErrDiv.style.display = 'none';
    }})
    .catch(function() {{
      if (tid) clearTimeout(tid);
      apiErrDiv.style.display = 'block';
    }});
}}
setInterval(poll, 5000);

// ── World → screen ────────────────────────────────────────────────────────────
function w2s(wx, wy, wz) {{
  var v = new THREE.Vector3(wx, wy, wz);
  v.project(camera);
  return {{ left: (v.x+1)/2*W, top: (-v.y+1)/2*H_VIEW, behind: v.z > 1 }};
}}

// ── Room labels ───────────────────────────────────────────────────────────────
var rlA = document.getElementById('rl-a');
var rlB = document.getElementById('rl-b');
setTimeout(function() {{ rlA.style.display = 'block'; rlB.style.display = 'block'; }}, 200);

// ── Animation loop ────────────────────────────────────────────────────────────
var clock = new THREE.Clock();

function animate() {{
  requestAnimationFrame(animate);
  var t = clock.getElapsedTime();

  // Auto-rotate management
  if (autoRot) {{
    sph.theta += 0.003;
    updateCam();
  }} else if (Date.now() - lastActivity > 3000) {{
    autoRot = true;
  }}

  // Pulse hot rack edges
  hotRacks.forEach(function(item) {{
    if (!item.alertEdge) return;
    var isC = item.mesh.userData.temp > 32;
    var p = 0.5 + 0.5 * Math.sin(t * 3.0 + item.mesh.position.x);
    item.alertEdge.material.color = isC
      ? new THREE.Color(1, p * 0.15, 0)
      : new THREE.Color(1, p * 0.35, 0);
  }});

  // Breathing on critical racks
  critRacks.forEach(function(mesh) {{
    var b = 1 + 0.03 * Math.sin(t * 2.6 + mesh.position.z);
    mesh.scale.y = mesh.userData.origH * b;
    mesh.position.y = mesh.userData.origH * b / 2;
  }});

  // Update room label screen positions
  var pA = w2s({room_a_cx:.2f}, 4.5, {room_cz:.2f});
  var pB = w2s({room_b_cx:.2f}, 4.5, {room_cz:.2f});
  rlA.style.left = pA.left + 'px'; rlA.style.top = pA.top + 'px';
  rlB.style.left = pB.left + 'px'; rlB.style.top = pB.top + 'px';
  rlA.style.opacity = pA.behind ? '0' : '1';
  rlB.style.opacity = pB.behind ? '0' : '1';

  renderer.render(scene, camera);
}}
animate();

// Initial data timestamp
if (RACKS.length > 0) updDiv.textContent = 'Last updated: ' + new Date().toLocaleTimeString();
</script>
</body>
</html>"""
