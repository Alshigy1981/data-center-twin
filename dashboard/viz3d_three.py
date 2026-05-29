"""
Premium 3D data center visualization using Three.js (WebGL).

Generates a self-contained HTML page that is embedded in the Streamlit
dashboard via st.components.v1.html().

Features
--------
- Physically-based lighting (ambient + directional + hemisphere)
- Temperature-gradient coloured racks (blue → green → amber → red)
- Rack height proportional to power draw
- CRAC units rendered as tall cyan boxes
- Hot / cold aisle strips on the floor
- Grid overlay on the floor
- Hover raycasting tooltip (rack ID, temp, power)
- Animated glow pulse on racks above 28 °C
- Orbit controls (drag to rotate, scroll to zoom, right-drag to pan)
- Legend and live stats overlay
- Fully offline-capable (Three.js loaded from CDN; swap for local path if needed)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────
# Layout constants (must match viz3d.py)
# ─────────────────────────────────────────
RACK_PITCH  = 1.20
ROW_PITCH   = 4.80
ROOM_GAP    = 6.00
N_ROWS      = 4
N_RACKS     = 10
MIN_H       = 0.40
MAX_H       = 2.80


def _rack_pos(room: str, row: str, position: int) -> dict:
    room_idx = 0 if "A" in room else 1
    row_idx  = int(row.split()[-1]) - 1
    pos_idx  = int(position) - 1
    return {
        "x": pos_idx * RACK_PITCH,
        "z": row_idx * ROW_PITCH + room_idx * (N_ROWS * ROW_PITCH + ROOM_GAP),
    }


def _crac_pos(asset_id: str, room: str) -> dict:
    try:
        num = int(asset_id.split("-")[-1]) - 1
    except ValueError:
        return {"x": 0, "z": 0}
    room_idx = 0 if "A" in room else 1
    x = (-1.80) if num % 2 == 0 else (N_RACKS * RACK_PITCH + 0.60)
    z = (num // 2) * 2 * ROW_PITCH + ROW_PITCH
    z += room_idx * (N_ROWS * ROW_PITCH + ROOM_GAP)
    return {"x": x, "z": z}


def _build_data(telemetry: Optional[Dict]) -> tuple[list, list]:
    """Extract rack and CRAC data arrays for embedding in HTML."""
    racks_js, cracs_js = [], []
    t = telemetry or {}

    for r in t.get("racks", []):
        pos = _rack_pos(r.get("room", "Room A"),
                        r.get("row", "Row 1"),
                        r.get("position", 1))
        racks_js.append({
            "id":      r.get("asset_id", ""),
            "x":       pos["x"],
            "z":       pos["z"],
            "temp":    float(r.get("inlet_temp_c", 22)),
            "outlet":  float(r.get("outlet_temp_c", 32)),
            "power":   float(r.get("power_kw", 0)),
            "humid":   float(r.get("humidity_pct", 50)),
            "ashrae":  str(r.get("ashrae_class", "A1")),
            "room":    r.get("room", ""),
            "row":     r.get("row", ""),
        })

    for c in t.get("cracs", []):
        pos = _crac_pos(c.get("asset_id", ""), c.get("room", "Room A"))
        cracs_js.append({
            "id":      c.get("asset_id", ""),
            "x":       pos["x"],
            "z":       pos["z"],
            "supply":  float(c.get("supply_air_temp_c", 18)),
            "ret":     float(c.get("return_air_temp_c", 28)),
            "dt":      float(c.get("delta_t_c", 10)),
            "cooling": float(c.get("cooling_load_kw", 60)),
            "room":    c.get("room", ""),
        })

    return racks_js, cracs_js


_THREE_CDN  = "https://unpkg.com/three@0.160.0/build/three.module.js"
_ORBIT_CDN  = "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js"


def make_three_html(telemetry: Optional[Dict],
                    width: int = 1200,
                    height: int = 720) -> str:
    """Return a self-contained HTML page with the Three.js 3D scene."""

    racks_data, cracs_data = _build_data(telemetry)
    racks_json = json.dumps(racks_data)
    cracs_json = json.dumps(cracs_data)

    min_h      = MIN_H
    max_h      = MAX_H
    rack_w     = 0.65
    rack_d     = 0.65
    crac_w     = 1.20
    crac_d     = 0.80
    crac_h     = 2.50
    floor_x0   = -3.0
    floor_x1   = N_RACKS * RACK_PITCH + 2.5
    floor_z0   = -2.0
    floor_z1   = (N_ROWS * ROW_PITCH) * 2 + ROOM_GAP + 3.0
    cam_target_x = (N_RACKS - 1) * RACK_PITCH / 2
    cam_target_z = floor_z1 / 2

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background:#0a1520;
    overflow:hidden;
    font-family:'Inter','Segoe UI',system-ui,sans-serif;
    color:#c8d8e8;
    width:{width}px;
    height:{height}px;
  }}
  canvas {{ display:block; }}

  #tooltip {{
    position:absolute;
    display:none;
    background:rgba(10,21,32,0.96);
    border:1px solid #00aaff;
    border-radius:6px;
    padding:10px 14px;
    font-size:11.5px;
    line-height:1.65;
    color:#c8d8e8;
    pointer-events:none;
    max-width:210px;
    z-index:10;
    box-shadow:0 4px 16px rgba(0,0,0,0.6);
  }}
  #tooltip b {{ color:#e0f0ff; }}
  #tooltip .th {{ color:#ff6666; }}
  #tooltip .tc {{ color:#00cc88; }}

  #legend {{
    position:absolute;
    top:10px; right:10px;
    background:rgba(10,18,28,0.88);
    border:1px solid #1a3050;
    border-radius:6px;
    padding:10px 14px;
    font-size:10.5px;
    z-index:10;
    min-width:140px;
  }}
  #legend .title {{
    color:#7090b0;
    text-transform:uppercase;
    letter-spacing:.08em;
    font-size:9px;
    margin-bottom:7px;
  }}
  .grad-bar {{
    width:100%;
    height:10px;
    border-radius:3px;
    background:linear-gradient(to right,#0044bb,#00cc88,#ffaa00,#ff1111);
    margin:5px 0 3px;
  }}
  .grad-labels {{
    display:flex;
    justify-content:space-between;
    font-size:9px;
    color:#4a6a8a;
  }}
  .legend-row {{
    display:flex; align-items:center; gap:7px; margin-top:6px;
  }}
  .swatch {{
    width:14px; height:14px; border-radius:2px; flex-shrink:0;
  }}

  #stats {{
    position:absolute;
    bottom:10px; left:10px; right:10px;
    background:rgba(10,18,28,0.82);
    border:1px solid #1a3050;
    border-radius:5px;
    padding:7px 14px;
    font-size:10px;
    display:flex; gap:22px; align-items:center;
    z-index:10;
  }}
  #stats .lbl {{ color:#3a5a7a; }}
  #stats .val {{ color:#00aaff; font-weight:600; }}
  #stats .hot {{ color:#ff5544; font-weight:600; }}
  #stats .cool{{ color:#00cc88; font-weight:600; }}

  #hint {{
    position:absolute;
    top:10px; left:10px;
    color:#2a4a6a;
    font-size:9.5px;
    z-index:10;
    letter-spacing:.04em;
  }}
</style>

<script type="importmap">
{{
  "imports": {{
    "three":            "{_THREE_CDN}",
    "three/addons/":    "https://unpkg.com/three@0.160.0/examples/jsm/"
  }}
}}
</script>
</head>

<body>
<div id="tooltip"></div>
<div id="hint">drag · scroll · right-drag pan</div>

<div id="legend">
  <div class="title">Rack Inlet Temperature</div>
  <div class="grad-bar"></div>
  <div class="grad-labels"><span>16°C</span><span>27°C</span><span>38°C</span></div>
  <div class="legend-row"><div class="swatch" style="background:#005577"></div><span style="font-size:10px;color:#6a8aaa">CRAC unit</span></div>
  <div class="legend-row"><div class="swatch" style="background:#001f66;opacity:.7"></div><span style="font-size:10px;color:#6a8aaa">Cold aisle</span></div>
  <div class="legend-row"><div class="swatch" style="background:#440000;opacity:.7"></div><span style="font-size:10px;color:#6a8aaa">Hot aisle</span></div>
</div>

<div id="stats">
  <span><span class="lbl">IT Load </span><span class="val" id="s-it">—</span></span>
  <span><span class="lbl">Avg Inlet </span><span class="val" id="s-avg">—</span></span>
  <span><span class="lbl">Max Inlet </span><span class="hot" id="s-max">—</span></span>
  <span><span class="lbl">Min Inlet </span><span class="cool" id="s-min">—</span></span>
  <span><span class="lbl">Hottest rack </span><span class="hot" id="s-hid">—</span></span>
</div>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

// ── Data ─────────────────────────────────────────────────────────────────────
const RACKS = {racks_json};
const CRACS = {cracs_json};
const MIN_H = {min_h}, MAX_H = {max_h};
const RACK_W = {rack_w}, RACK_D = {rack_d};
const CRAC_W = {crac_w}, CRAC_D = {crac_d}, CRAC_H = {crac_h};

// ── Scene ─────────────────────────────────────────────────────────────────────
const W = {width}, H = {height};
const scene    = new THREE.Scene();
scene.background = new THREE.Color(0x0a1520);
scene.fog        = new THREE.FogExp2(0x0a1520, 0.009);

const camera = new THREE.PerspectiveCamera(42, W / H, 0.1, 500);
camera.position.set({cam_target_x} + 8, 14, {cam_target_z} + 14);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(W, H);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type    = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

// ── Controls ──────────────────────────────────────────────────────────────────
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping  = true;
controls.dampingFactor  = 0.07;
controls.target.set({cam_target_x}, 0.5, {cam_target_z});
controls.minDistance    = 4;
controls.maxDistance    = 80;
controls.maxPolarAngle  = Math.PI / 2.05;
controls.update();

// ── Lights ────────────────────────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0x334466, 0.7));

const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(20, 30, 20);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.near = 1;
sun.shadow.camera.far  = 120;
sun.shadow.camera.left = sun.shadow.camera.bottom = -40;
sun.shadow.camera.right = sun.shadow.camera.top   =  40;
scene.add(sun);

scene.add(new THREE.HemisphereLight(0x334466, 0x0a1520, 0.5));

// ── Floor ─────────────────────────────────────────────────────────────────────
const floorGeo = new THREE.PlaneGeometry({floor_x1 - floor_x0:.1f}, {floor_z1 - floor_z0:.1f});
const floorMat = new THREE.MeshLambertMaterial({{ color: 0x0d1a26 }});
const floor    = new THREE.Mesh(floorGeo, floorMat);
floor.rotation.x = -Math.PI / 2;
floor.position.set(
  ({floor_x0:.2f} + {floor_x1:.2f}) / 2,
  0,
  ({floor_z0:.2f} + {floor_z1:.2f}) / 2
);
floor.receiveShadow = true;
scene.add(floor);

// Grid
const grid = new THREE.GridHelper(
  Math.max({floor_x1 - floor_x0:.1f}, {floor_z1 - floor_z0:.1f}),
  40, 0x1a2a3a, 0x0f1f2e
);
grid.position.set(
  ({floor_x0:.2f} + {floor_x1:.2f}) / 2,
  0.005,
  ({floor_z0:.2f} + {floor_z1:.2f}) / 2
);
scene.add(grid);

// ── Aisle strips ──────────────────────────────────────────────────────────────
function aisleStrip(x0, z0, x1, z1, hex, opacity) {{
  const geo = new THREE.PlaneGeometry(x1 - x0, z1 - z0);
  const mat = new THREE.MeshBasicMaterial({{
    color: hex,
    transparent: true,
    opacity,
    depthWrite: false,
  }});
  const m = new THREE.Mesh(geo, mat);
  m.rotation.x = -Math.PI / 2;
  m.position.set((x0 + x1) / 2, 0.01, (z0 + z1) / 2);
  scene.add(m);
}}

const ROW_PITCH = {ROW_PITCH};
const N_ROWS    = {N_ROWS};
const ROOM_GAP  = {ROOM_GAP};
const N_RACKS   = {N_RACKS};
const RACK_PITCH= {RACK_PITCH};

for (let room = 0; room < 2; room++) {{
  const zBase = room * (N_ROWS * ROW_PITCH + ROOM_GAP);
  for (let row = 0; row < N_ROWS - 1; row++) {{
    const za = zBase + row * ROW_PITCH + RACK_D + 0.05;
    const zb = zBase + (row + 1) * ROW_PITCH - 0.05;
    const x0 = -1.2, x1 = (N_RACKS - 1) * RACK_PITCH + RACK_W + 1.2;
    const isCold = (row % 2 === 0);
    aisleStrip(x0, za, x1, zb, isCold ? 0x002266 : 0x440000, 0.22);
  }}
}}

// ── Temperature → colour ──────────────────────────────────────────────────────
function tempColor(t) {{
  const stops = [
    [16, [0,  68, 187]],
    [22, [0, 153, 204]],
    [27, [0, 204, 136]],
    [31, [255,170,  0]],
    [35, [255, 85,  0]],
    [38, [255, 17, 17]],
  ];
  for (let i = 0; i < stops.length - 1; i++) {{
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (t <= t1) {{
      const f = (t - t0) / (t1 - t0);
      return new THREE.Color(
        (c0[0] + f * (c1[0] - c0[0])) / 255,
        (c0[1] + f * (c1[1] - c0[1])) / 255,
        (c0[2] + f * (c1[2] - c0[2])) / 255,
      );
    }}
  }}
  return new THREE.Color(1, 0.07, 0.07);
}}

// ── Build rack boxes ──────────────────────────────────────────────────────────
const rackMeshes  = [];
const hotRacks    = [];   // racks above 28°C — will pulse
const GEO_CACHE   = {{}};  // reuse geometry for same height bucket

RACKS.forEach(r => {{
  const h    = MIN_H + Math.max(0, Math.min(r.power, 20)) / 20 * (MAX_H - MIN_H);
  const hKey = Math.round(h * 10);
  if (!GEO_CACHE[hKey])
    GEO_CACHE[hKey] = new THREE.BoxGeometry(RACK_W, h, RACK_D);

  const geo  = GEO_CACHE[hKey];
  const col  = tempColor(r.temp);
  const emis = col.clone().multiplyScalar(r.temp > 28 ? 0.25 : 0.06);

  const mat  = new THREE.MeshLambertMaterial({{
    color:    col,
    emissive: emis,
  }});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(r.x + RACK_W / 2, h / 2, r.z + RACK_D / 2);
  mesh.castShadow = mesh.receiveShadow = true;
  mesh.userData   = r;
  scene.add(mesh);
  rackMeshes.push(mesh);

  // Wire edges
  const edges = new THREE.EdgesGeometry(geo);
  const wire  = new THREE.LineSegments(
    edges,
    new THREE.LineBasicMaterial({{ color: 0x1a3050 }})
  );
  mesh.add(wire);

  if (r.temp > 28) hotRacks.push(mesh);
}});

// ── Build CRAC boxes ──────────────────────────────────────────────────────────
const cracGeo = new THREE.BoxGeometry(CRAC_W, CRAC_H, CRAC_D);
CRACS.forEach(c => {{
  const mat  = new THREE.MeshLambertMaterial({{
    color:    0x005577,
    emissive: 0x002233,
  }});
  const mesh = new THREE.Mesh(cracGeo, mat);
  mesh.position.set(c.x + CRAC_W / 2, CRAC_H / 2, c.z + CRAC_D / 2);
  mesh.castShadow = mesh.receiveShadow = true;
  mesh.userData   = c;
  scene.add(mesh);
  rackMeshes.push(mesh);   // include in raycasting

  const edges = new THREE.EdgesGeometry(cracGeo);
  mesh.add(new THREE.LineSegments(
    edges,
    new THREE.LineBasicMaterial({{ color: 0x006688 }})
  ));
}});

// ── Stats overlay ─────────────────────────────────────────────────────────────
if (RACKS.length > 0) {{
  const temps  = RACKS.map(r => r.temp);
  const powers = RACKS.map(r => r.power);
  const itLoad = powers.reduce((a,b) => a+b, 0);
  const avg    = temps.reduce((a,b)=>a+b,0) / temps.length;
  const mx     = Math.max(...temps);
  const mn     = Math.min(...temps);
  const hotR   = RACKS.reduce((a,b) => b.temp > a.temp ? b : a, RACKS[0]);
  document.getElementById('s-it').textContent  = itLoad.toFixed(1) + ' kW';
  document.getElementById('s-avg').textContent = avg.toFixed(1) + '°C';
  document.getElementById('s-max').textContent = mx.toFixed(1) + '°C';
  document.getElementById('s-min').textContent = mn.toFixed(1) + '°C';
  document.getElementById('s-hid').textContent = hotR.id;
}}

// ── Tooltip / raycasting ──────────────────────────────────────────────────────
const raycaster = new THREE.Raycaster();
const mouse     = new THREE.Vector2();
const tooltip   = document.getElementById('tooltip');
let   hovered   = null;

function onMouseMove(e) {{
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x =  ((e.clientX - rect.left)  / W) * 2 - 1;
  mouse.y = -((e.clientY - rect.top)   / H) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(rackMeshes);

  if (hits.length > 0) {{
    const d = hits[0].object.userData;
    if (d.id) {{
      const isCrac = !!d.supply;
      tooltip.style.display = 'block';
      tooltip.style.left    = (e.clientX - rect.left + 14) + 'px';
      tooltip.style.top     = (e.clientY - rect.top  - 10) + 'px';
      if (isCrac) {{
        tooltip.innerHTML =
          `<b>${{d.id}}</b><br>` +
          `Supply: ${{d.supply.toFixed(1)}}°C &nbsp; Return: ${{d.ret.toFixed(1)}}°C<br>` +
          `ΔT: <span class="tc">${{d.dt.toFixed(1)}}°C</span> &nbsp; Cooling: ${{d.cooling.toFixed(1)}} kW<br>` +
          `${{d.room}}`;
      }} else {{
        const cls = d.temp > 30 ? 'th' : 'tc';
        tooltip.innerHTML =
          `<b>${{d.id}}</b><br>` +
          `Inlet: <span class="${{cls}}">${{d.temp.toFixed(1)}}°C</span> &nbsp;` +
          `Outlet: ${{d.outlet.toFixed(1)}}°C<br>` +
          `Power: ${{d.power.toFixed(1)}} kW &nbsp; Humid: ${{d.humid.toFixed(0)}}%<br>` +
          `ASHRAE: ${{d.ashrae}}<br>` +
          `${{d.room}} · ${{d.row}}`;
      }}
    }}
  }} else {{
    tooltip.style.display = 'none';
  }}
}}
renderer.domElement.addEventListener('mousemove', onMouseMove);

// ── Animation loop ────────────────────────────────────────────────────────────
const clock = new THREE.Clock();
function animate() {{
  requestAnimationFrame(animate);
  controls.update();

  // Pulse glow on hot racks
  const t = clock.getElapsedTime();
  hotRacks.forEach(mesh => {{
    const base  = tempColor(mesh.userData.temp);
    const pulse = 0.18 + 0.18 * Math.sin(t * 2.6 + mesh.position.x);
    mesh.material.emissive = base.clone().multiplyScalar(pulse);
  }});

  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>"""
