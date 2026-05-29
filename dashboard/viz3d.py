"""
3D data center floor visualization using Plotly Mesh3d.

All 80 racks are rendered as temperature-coloured boxes with height
proportional to power draw.  CRAC units appear as distinct cyan boxes.
Hot and cold aisles are marked with translucent floor strips.

Accepts the raw API JSON dict returned by GET /telemetry/live.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go

# ─────────────────────────────────────────
# Layout constants (metres)
# ─────────────────────────────────────────
RACK_W     = 0.65
RACK_D     = 0.65
RACK_PITCH = 1.20   # centre-to-centre along a row
ROW_PITCH  = 4.80   # centre-to-centre between rows
ROOM_GAP   = 6.00   # extra z-gap between Room A and Room B

MIN_RACK_H = 0.40   # height at 0 kW
MAX_RACK_H = 2.80   # height at 20 kW

CRAC_W     = 1.20
CRAC_D     = 0.80
CRAC_H     = 2.50

N_ROOMS    = 2
N_ROWS     = 4
N_RACKS    = 10

# ─────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────

def _rack_origin(room: str, row: str, position: int) -> tuple[float, float]:
    """Return bottom-left corner (x, z) for a rack."""
    room_idx = 0 if "A" in room else 1
    row_idx  = int(row.split()[-1]) - 1   # "Row 1" → 0
    pos_idx  = int(position) - 1

    x = pos_idx * RACK_PITCH
    z = row_idx * ROW_PITCH + room_idx * (N_ROWS * ROW_PITCH + ROOM_GAP)
    return float(x), float(z)


def _crac_origin(asset_id: str, room: str) -> tuple[float, float]:
    """Place CRAC units on the long sides of each room."""
    try:
        num = int(asset_id.split("-")[-1]) - 1   # 0-based
    except ValueError:
        return 0.0, 0.0

    room_idx = 0 if "A" in room else 1
    # Even-numbered CRACs on left, odd on right
    x = (-CRAC_W - 0.4) if num % 2 == 0 else (N_RACKS * RACK_PITCH + 0.4)
    # Split across top half / bottom half of room
    z_mid = (num // 2) * 2 * ROW_PITCH + ROW_PITCH
    z = z_mid + room_idx * (N_ROWS * ROW_PITCH + ROOM_GAP)
    return float(x), float(z)


def _power_to_height(kw: float) -> float:
    return MIN_RACK_H + max(0.0, min(float(kw), 20.0)) / 20.0 * (MAX_RACK_H - MIN_RACK_H)


# ─────────────────────────────────────────
# Box geometry
# ─────────────────────────────────────────

def _box(x0: float, z0: float, h: float,
         w: float = RACK_W, d: float = RACK_D):
    """
    Return (vx, vy, vz, i, j, k) for a closed box.
    Origin at (x0, 0, z0); size (w, h, d); y is the vertical axis.
    """
    x1, y1, z1 = x0 + w, h, z0 + d

    vx = [x0, x1, x1, x0,  x0, x1, x1, x0]
    vy = [0,  0,  0,  0,   y1, y1, y1, y1]
    vz = [z0, z0, z1, z1,  z0, z0, z1, z1]

    # 12 triangles for 6 faces (CCW winding)
    i = [0, 0,  4, 4,  0, 1,  3, 2,  0, 3,  1, 2]
    j = [1, 2,  5, 6,  1, 5,  2, 6,  4, 7,  5, 6]
    k = [2, 3,  6, 7,  5, 4,  6, 7,  7, 4,  2, 5]

    return vx, vy, vz, i, j, k


# ─────────────────────────────────────────
# Trace builders
# ─────────────────────────────────────────

def _rack_trace(racks: List[Dict]) -> go.Mesh3d:
    """Combine all racks into one Mesh3d, vertex-coloured by inlet temperature."""
    ax, ay, az, ai, aj, ak, intensities, texts = [], [], [], [], [], [], [], []

    for r in racks:
        room     = r.get("room", "Room A")
        row      = r.get("row", "Row 1")
        pos      = r.get("position", 1)
        power    = float(r.get("power_kw", 0))
        temp     = float(r.get("inlet_temp_c", 22))
        outlet   = float(r.get("outlet_temp_c", 32))
        humid    = float(r.get("humidity_pct", 50))
        ashrae   = str(r.get("ashrae_class", "A1"))
        aid      = r.get("asset_id", "")

        x0, z0  = _rack_origin(room, row, pos)
        h       = _power_to_height(power)
        vx, vy, vz, fi, fj, fk = _box(x0, z0, h)

        off = len(ax)
        ax.extend(vx); ay.extend(vy); az.extend(vz)
        ai.extend(v + off for v in fi)
        aj.extend(v + off for v in fj)
        ak.extend(v + off for v in fk)
        intensities.extend([temp] * 8)

        tip = (f"<b>{aid}</b><br>"
               f"Inlet: {temp:.1f}°C  Outlet: {outlet:.1f}°C<br>"
               f"Power: {power:.1f} kW<br>"
               f"Humidity: {humid:.0f}%  ASHRAE: {ashrae}<br>"
               f"{room} · {row} · Position {pos}")
        texts.extend([tip] * 8)

    return go.Mesh3d(
        x=ax, y=ay, z=az,
        i=ai, j=aj, k=ak,
        intensity=intensities,
        intensitymode="vertex",
        colorscale=[
            [0.00, "#0044bb"],
            [0.20, "#0099cc"],
            [0.45, "#00cc88"],
            [0.65, "#ffaa00"],
            [0.85, "#ff5500"],
            [1.00, "#ff1111"],
        ],
        cmin=16, cmax=38,
        colorbar=dict(
            title=dict(text="Inlet °C", font=dict(color="#9ab8d0", size=11)),
            tickfont=dict(color="#9ab8d0"),
            bgcolor="#0a1520",
            outlinecolor="#1a3050",
            thickness=12,
            x=1.01,
        ),
        flatshading=True,
        lighting=dict(ambient=0.55, diffuse=0.90, specular=0.30, roughness=0.40),
        lightposition=dict(x=600, y=900, z=500),
        hovertemplate="%{text}<extra></extra>",
        text=texts,
        name="Racks",
        showscale=True,
    )


def _crac_trace(cracs: List[Dict]) -> Optional[go.Mesh3d]:
    """Render CRAC units as opaque cyan boxes."""
    if not cracs:
        return None

    ax, ay, az, ai, aj, ak, texts = [], [], [], [], [], [], []

    for c in cracs:
        room    = c.get("room", "Room A")
        aid     = c.get("asset_id", "")
        supply  = float(c.get("supply_air_temp_c", 18))
        ret     = float(c.get("return_air_temp_c", 28))
        dt      = float(c.get("delta_t_c", 10))
        cooling = float(c.get("cooling_load_kw", 60))

        x0, z0 = _crac_origin(aid, room)
        vx, vy, vz, fi, fj, fk = _box(x0, z0, CRAC_H, CRAC_W, CRAC_D)

        off = len(ax)
        ax.extend(vx); ay.extend(vy); az.extend(vz)
        ai.extend(v + off for v in fi)
        aj.extend(v + off for v in fj)
        ak.extend(v + off for v in fk)

        tip = (f"<b>{aid}</b><br>"
               f"Supply: {supply:.1f}°C  Return: {ret:.1f}°C<br>"
               f"ΔT: {dt:.1f}°C  Cooling: {cooling:.1f} kW<br>"
               f"{room}")
        texts.extend([tip] * 8)

    return go.Mesh3d(
        x=ax, y=ay, z=az,
        i=ai, j=aj, k=ak,
        color="#005577",
        opacity=0.90,
        flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.85, specular=0.5),
        lightposition=dict(x=600, y=900, z=500),
        hovertemplate="%{text}<extra></extra>",
        text=texts,
        name="CRACs",
        showscale=False,
    )


def _floor_trace(racks: List[Dict]) -> go.Mesh3d:
    """Flat dark floor plane sized to the rack footprint."""
    if racks:
        origins = [_rack_origin(
            r.get("room", "Room A"),
            r.get("row", "Row 1"),
            r.get("position", 1),
        ) for r in racks]
        xs = [o[0] for o in origins]
        zs = [o[1] for o in origins]
        x0, x1 = min(xs) - 2.5, max(xs) + RACK_W + 2.5
        z0, z1 = min(zs) - 2.0, max(zs) + RACK_D + 2.5
    else:
        x0, x1, z0, z1 = -3.0, 15.0, -2.0, 35.0

    return go.Mesh3d(
        x=[x0, x1, x1, x0],
        y=[0, 0, 0, 0],
        z=[z0, z0, z1, z1],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="#0d1a26",
        flatshading=True,
        lighting=dict(ambient=0.85),
        showscale=False,
        hoverinfo="none",
        name="Floor",
    )


def _aisle_strip(x0: float, z0: float, x1: float, z1: float,
                 color: str, opacity: float = 0.18) -> go.Mesh3d:
    return go.Mesh3d(
        x=[x0, x1, x1, x0],
        y=[0.01, 0.01, 0.01, 0.01],
        z=[z0, z0, z1, z1],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color=color,
        opacity=opacity,
        flatshading=True,
        showscale=False,
        hoverinfo="none",
        name="Aisle",
        showlegend=False,
    )


def _aisle_traces() -> List[go.Mesh3d]:
    """Hot (red) and cold (blue) aisle strips between every pair of rows."""
    strips = []
    x0 = -1.0
    x1 = (N_RACKS - 1) * RACK_PITCH + RACK_W + 1.0

    for room_idx in range(N_ROOMS):
        z_base = room_idx * (N_ROWS * ROW_PITCH + ROOM_GAP)
        for row in range(N_ROWS - 1):
            za = z_base + row * ROW_PITCH + RACK_D + 0.05
            zb = z_base + (row + 1) * ROW_PITCH - 0.05
            color = "#001f66" if row % 2 == 0 else "#440000"
            strips.append(_aisle_strip(x0, za, x1, zb, color))

    return strips


def _label_trace() -> go.Scatter3d:
    """Floating row labels to orient the viewer."""
    lx, ly, lz, texts = [], [], [], []
    for room_idx, room_ch in enumerate(["A", "B"]):
        z_base = room_idx * (N_ROWS * ROW_PITCH + ROOM_GAP)
        for row in range(N_ROWS):
            lx.append(-3.2)
            ly.append(0.4)
            lz.append(z_base + row * ROW_PITCH + RACK_D / 2)
            texts.append(f"Room {room_ch} R{row+1}")

    return go.Scatter3d(
        x=lx, y=ly, z=lz,
        text=texts, mode="text",
        textfont=dict(color="#2a5a8a", size=9),
        hoverinfo="none",
        showlegend=False,
    )


# ─────────────────────────────────────────
# Public API
# ─────────────────────────────────────────

def make_3d_figure(telemetry: Optional[Dict]) -> go.Figure:
    """
    Build a Plotly 3D figure of the data center floor plan.

    Args:
        telemetry: Raw dict from GET /telemetry/live, or None.

    Returns:
        Plotly Figure for use with st.plotly_chart().
    """
    racks = (telemetry or {}).get("racks", [])
    cracs = (telemetry or {}).get("cracs", [])

    traces: list = [_floor_trace(racks)]
    traces.extend(_aisle_traces())
    if racks:
        traces.append(_rack_trace(racks))
    crac_t = _crac_trace(cracs)
    if crac_t:
        traces.append(crac_t)
    traces.append(_label_trace())

    # Centre camera on the full floor
    total_z = N_ROOMS * N_ROWS * ROW_PITCH + (N_ROOMS - 1) * ROOM_GAP
    cx = (N_RACKS - 1) * RACK_PITCH / 2

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#0f1923",
        margin=dict(l=0, r=0, t=38, b=0),
        title=dict(
            text="3D Data Center Floor — Rack Thermal View (Plotly)",
            font=dict(color="#c0d8f0", size=13),
            x=0.01,
        ),
        scene=dict(
            bgcolor="#0a1520",
            xaxis=dict(visible=False, backgroundcolor="#0a1520"),
            yaxis=dict(visible=False, backgroundcolor="#0a1520"),
            zaxis=dict(visible=False, backgroundcolor="#0a1520"),
            camera=dict(
                eye=dict(x=1.6, y=1.2, z=1.8),
                center=dict(x=0, y=-0.1, z=0),
                up=dict(x=0, y=1, z=0),
            ),
            aspectmode="data",
        ),
        showlegend=False,
        height=680,
    )

    return fig
