"""
3D data center floor visualization — Plotly Mesh3d, polished.

80 racks rendered as temperature-coloured boxes; height = power draw.
8 CRAC units as flat teal perimeter boxes.
Hot / cold aisle floor strips with labels.
Alert wireframe outlines for over-temperature / high-power racks.
ASHRAE A1/A2 reference planes.
Room side-by-side layout: Room A at x=0-12, Room B at x=18-30.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go

# ─────────────────────────────────────────
# Layout constants
# ─────────────────────────────────────────
RACK_W     = 0.80
RACK_D     = 0.80
RACK_PITCH = 1.20   # rack centre-to-centre along a row
ROW_PITCH  = 3.00   # row centre-to-centre (z direction)
ROOM_B_X   = 18.0   # Room B x-offset; aisle gap = x12 → x18 = 6 units

MIN_RACK_H = 0.50
MAX_RACK_H = 3.00   # height at 20 kW

CRAC_W     = 2.00
CRAC_D     = 1.50
CRAC_H     = 0.40   # flat, low profile

N_ROOMS    = 2
N_ROWS     = 4
N_RACKS    = 10

COLORSCALE = [
    [0.00, "#0066ff"],
    [0.30, "#00ccaa"],
    [0.55, "#00cc44"],
    [0.75, "#ffaa00"],
    [0.90, "#ff6600"],
    [1.00, "#ff0000"],
]
CMIN, CMAX = 15.0, 40.0

# Fixed CRAC positions: (label, cx, cz, room_idx)
# 2 on front/back walls (z axis), 2 on left/right walls (x axis) per room
_CRAC_POS: List[Tuple[str, float, float, int]] = [
    ("C-A1",  5.50,  -1.25, 0),
    ("C-A2",  5.50,  11.25, 0),
    ("C-A3", -1.25,   4.90, 0),
    ("C-A4", 12.25,   4.90, 0),
    ("C-B1",  ROOM_B_X + 5.50,  -1.25, 1),
    ("C-B2",  ROOM_B_X + 5.50,  11.25, 1),
    ("C-B3",  ROOM_B_X - 1.25,   4.90, 1),
    ("C-B4",  ROOM_B_X + 12.25,  4.90, 1),
]

# ─────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────

def _rack_origin(room: str, row: str, position: int) -> Tuple[float, float]:
    """Return bottom-left corner (x, z) for a rack box."""
    room_idx = 0 if "A" in room else 1
    row_idx  = int(row.split()[-1]) - 1
    pos_idx  = int(position) - 1
    x = (ROOM_B_X if room_idx else 0.0) + pos_idx * RACK_PITCH
    z = row_idx * ROW_PITCH
    return float(x), float(z)


def _power_to_height(kw: float) -> float:
    return max(MIN_RACK_H, min(float(kw), 20.0) / 20.0 * MAX_RACK_H)


# ─────────────────────────────────────────
# Box geometry
# ─────────────────────────────────────────

def _box(x0: float, z0: float, h: float,
         w: float = RACK_W, d: float = RACK_D) -> Tuple:
    """Return (vx, vy, vz, i, j, k) for a closed box."""
    x1, y1, z1 = x0 + w, h, z0 + d
    vx = [x0, x1, x1, x0, x0, x1, x1, x0]
    vy = [0,   0,  0,  0, y1, y1, y1, y1]
    vz = [z0, z0, z1, z1, z0, z0, z1, z1]
    i  = [0, 0, 4, 4, 0, 1, 3, 2, 0, 3, 1, 2]
    j  = [1, 2, 5, 6, 1, 5, 2, 6, 4, 7, 5, 6]
    k  = [2, 3, 6, 7, 5, 4, 6, 7, 7, 4, 2, 5]
    return vx, vy, vz, i, j, k


def _wireframe(x0: float, z0: float, h: float,
               w: float = RACK_W, d: float = RACK_D,
               e: float = 0.04) -> Tuple[list, list, list]:
    """12-edge wireframe lines for an alert box (expands by e on all sides)."""
    xe0, ze0 = x0 - e, z0 - e
    xe1, ye1, ze1 = x0 + w + e, h + e, z0 + d + e
    c = [
        (xe0, 0,   ze0), (xe1, 0,   ze0), (xe1, 0,   ze1), (xe0, 0,   ze1),
        (xe0, ye1, ze0), (xe1, ye1, ze0), (xe1, ye1, ze1), (xe0, ye1, ze1),
    ]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    px: list = []
    py: list = []
    pz: list = []
    for a, b in edges:
        px += [c[a][0], c[b][0], None]
        py += [c[a][1], c[b][1], None]
        pz += [c[a][2], c[b][2], None]
    return px, py, pz


# ─────────────────────────────────────────
# Trace builders
# ─────────────────────────────────────────

def _floor_trace() -> go.Mesh3d:
    x0 = -2.5
    x1 = ROOM_B_X + (N_RACKS - 1) * RACK_PITCH + RACK_W + 2.5
    z0 = -2.5
    z1 = (N_ROWS - 1) * ROW_PITCH + RACK_D + 3.0
    return go.Mesh3d(
        x=[x0, x1, x1, x0],
        y=[0,   0,  0,  0],
        z=[z0, z0, z1, z1],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="#0d1a26",
        flatshading=True,
        lighting=dict(ambient=0.85),
        showscale=False,
        hoverinfo="none",
        name="Floor",
        showlegend=False,
    )


def _aisle_strip_trace(
    x0: float, z0: float, x1: float, z1: float,
    color: str, opacity: float = 0.15,
) -> go.Mesh3d:
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
    traces = []
    for room_idx in range(N_ROOMS):
        x_off = ROOM_B_X if room_idx else 0.0
        xa = x_off - 0.3
        xb = x_off + (N_RACKS - 1) * RACK_PITCH + RACK_W + 0.3
        for row in range(N_ROWS - 1):
            za = row * ROW_PITCH + RACK_D + 0.05
            zb = (row + 1) * ROW_PITCH - 0.05
            color = "#001a55" if row % 2 == 0 else "#550000"
            traces.append(_aisle_strip_trace(xa, za, xb, zb, color))
    return traces


def _ashrae_planes() -> List[go.Mesh3d]:
    """Semi-transparent horizontal planes as ASHRAE A1/A2 reference levels."""
    x0 = -2.0
    x1 = ROOM_B_X + (N_RACKS - 1) * RACK_PITCH + RACK_W + 2.0
    z0 = -2.0
    z1 = (N_ROWS - 1) * ROW_PITCH + RACK_D + 2.0
    planes = []
    for y_h, color, opacity in [
        (MAX_RACK_H * 0.75, "#ffaa00", 0.05),   # A1 32°C visual limit
        (MAX_RACK_H * 0.92, "#ff4444", 0.05),   # A2 35°C visual limit
    ]:
        planes.append(go.Mesh3d(
            x=[x0, x1, x1, x0],
            y=[y_h, y_h, y_h, y_h],
            z=[z0, z0, z1, z1],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color,
            opacity=opacity,
            flatshading=True,
            showscale=False,
            hoverinfo="none",
            showlegend=False,
        ))
    return planes


def _rack_trace(racks: List[Dict]) -> go.Mesh3d:
    ax: list = []; ay: list = []; az: list = []
    ai: list = []; aj: list = []; ak: list = []
    intensities: list = []
    texts: list = []

    for r in racks:
        room   = r.get("room", "Room A")
        row    = r.get("row", "Row 1")
        pos    = r.get("position", 1)
        power  = float(r.get("power_kw", 0))
        temp   = float(r.get("inlet_temp_c", 22))
        outlet = float(r.get("outlet_temp_c", 32))
        ashrae = str(r.get("ashrae_class", "A1"))
        aid    = r.get("asset_id", "")
        pdu    = r.get("pdu_id", "—")

        if temp > 32:
            alert = "⚠ CRITICAL"
        elif temp > 28:
            alert = "⚠ WARNING"
        elif power > 17:
            alert = "⚡ HIGH POWER"
        else:
            alert = "✓ None"

        ashrae_ok = "✓" if ashrae in ("A1", "A2") else "⚠"

        x0, z0 = _rack_origin(room, row, pos)
        h = _power_to_height(power)
        vx, vy, vz, fi, fj, fk = _box(x0, z0, h)

        off = len(ax)
        ax.extend(vx); ay.extend(vy); az.extend(vz)
        ai.extend(v + off for v in fi)
        aj.extend(v + off for v in fj)
        ak.extend(v + off for v in fk)
        intensities.extend([temp] * 8)

        tip = (
            f"<b>Rack ID:</b> {aid}<br>"
            f"<b>Inlet Temp:</b> {temp:.1f}°C<br>"
            f"<b>Outlet Temp:</b> {outlet:.1f}°C<br>"
            f"<b>Power Draw:</b> {power:.1f} kW<br>"
            f"<b>ASHRAE Class:</b> {ashrae} {ashrae_ok}<br>"
            f"<b>Row:</b> {room} / {row}<br>"
            f"<b>PDU:</b> {pdu}<br>"
            f"<b>Alert:</b> {alert}"
        )
        texts.extend([tip] * 8)

    return go.Mesh3d(
        x=ax, y=ay, z=az,
        i=ai, j=aj, k=ak,
        intensity=intensities,
        intensitymode="vertex",
        colorscale=COLORSCALE,
        cmin=CMIN, cmax=CMAX,
        colorbar=dict(
            title=dict(text="Inlet °C", font=dict(color="#9ab8d0", size=11)),
            tickvals=[15, 20, 24, 28, 32, 35, 40],
            ticktext=["15", "20", "24 A1✓", "28 warm", "32 A1 lim", "35 A2 lim", "40"],
            tickfont=dict(color="#9ab8d0", size=9),
            bgcolor="#0a1520",
            outlinecolor="#1a3050",
            thickness=14,
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


def _crac_trace(cracs_api: List[Dict]) -> go.Mesh3d:
    """8 flat CRAC boxes at fixed perimeter positions, teal colour."""
    # Group API CRACs by room for tooltip enrichment
    a_api = [c for c in cracs_api if "A" in c.get("room", "")]
    b_api = [c for c in cracs_api if "B" in c.get("room", "")]
    api_lookup: Dict[str, Dict] = {}
    for i, c in enumerate(a_api[:4]):
        api_lookup[f"C-A{i+1}"] = c
    for i, c in enumerate(b_api[:4]):
        api_lookup[f"C-B{i+1}"] = c

    ax: list = []; ay: list = []; az: list = []
    ai: list = []; aj: list = []; ak: list = []
    texts: list = []

    for label, cx, cz, _ in _CRAC_POS:
        x0 = cx - CRAC_W / 2
        z0 = cz - CRAC_D / 2
        vx, vy, vz, fi, fj, fk = _box(x0, z0, CRAC_H, CRAC_W, CRAC_D)
        off = len(ax)
        ax.extend(vx); ay.extend(vy); az.extend(vz)
        ai.extend(v + off for v in fi)
        aj.extend(v + off for v in fj)
        ak.extend(v + off for v in fk)

        api_c = api_lookup.get(label)
        if api_c:
            supply  = float(api_c.get("supply_air_temp_c", 18))
            ret     = float(api_c.get("return_air_temp_c", 28))
            dt      = float(api_c.get("delta_t_c", 10))
            cooling = float(api_c.get("cooling_load_kw", 60))
            real_id = api_c.get("asset_id", label)
            tip = (
                f"<b>{label}</b> ({real_id})<br>"
                f"Supply: {supply:.1f}°C  Return: {ret:.1f}°C<br>"
                f"ΔT: {dt:.1f}°C  Cooling: {cooling:.1f} kW"
            )
        else:
            tip = f"<b>{label}</b><br>CRAC Cooling Unit"
        texts.extend([tip] * 8)

    return go.Mesh3d(
        x=ax, y=ay, z=az,
        i=ai, j=aj, k=ak,
        color="#00ccaa",
        opacity=0.90,
        flatshading=True,
        lighting=dict(ambient=0.6, diffuse=0.85, specular=0.5),
        lightposition=dict(x=600, y=900, z=500),
        hovertemplate="%{text}<extra></extra>",
        text=texts,
        name="CRACs",
        showscale=False,
    )


def _alert_traces(racks: List[Dict]) -> List[go.Scatter3d]:
    """Wireframe outlines for over-temperature or high-power racks."""
    warn: Tuple[list, list, list] = ([], [], [])
    crit: Tuple[list, list, list] = ([], [], [])
    pwr:  Tuple[list, list, list] = ([], [], [])

    for r in racks:
        temp  = float(r.get("inlet_temp_c", 0))
        power = float(r.get("power_kw", 0))
        if temp <= 28 and power <= 17:
            continue
        x0, z0 = _rack_origin(
            r.get("room", "Room A"),
            r.get("row", "Row 1"),
            r.get("position", 1),
        )
        h = _power_to_height(power)
        px, py, pz = _wireframe(x0, z0, h)

        if temp > 32:
            crit[0].extend(px); crit[1].extend(py); crit[2].extend(pz)
        elif temp > 28:
            warn[0].extend(px); warn[1].extend(py); warn[2].extend(pz)
        elif power > 17:
            pwr[0].extend(px); pwr[1].extend(py); pwr[2].extend(pz)

    out: List[go.Scatter3d] = []
    for (px, py, pz), color, width in [
        (warn, "#ff4444", 2),
        (crit, "#ff0000", 4),
        (pwr,  "#ffaa00", 2),
    ]:
        if px:
            out.append(go.Scatter3d(
                x=px, y=py, z=pz,
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="none",
                showlegend=False,
            ))
    return out


def _label_trace() -> List[go.Scatter3d]:
    """Room labels and CRAC ID labels as separate traces."""
    # Room labels — larger, blue
    room_x, room_y, room_z, room_t = [], [], [], []
    for room_idx, room_ch in enumerate(["A", "B"]):
        x_off = ROOM_B_X if room_idx else 0.0
        room_x.append(x_off + (N_RACKS - 1) * RACK_PITCH / 2)
        room_y.append(MAX_RACK_H + 0.9)
        room_z.append((N_ROWS - 1) * ROW_PITCH / 2)
        room_t.append(f"ROOM {room_ch}")

    room_trace = go.Scatter3d(
        x=room_x, y=room_y, z=room_z,
        text=room_t, mode="text",
        textfont=dict(color="#00aaff", size=13),
        hoverinfo="none",
        showlegend=False,
    )

    # CRAC labels — smaller, teal
    crac_x, crac_y, crac_z, crac_t = [], [], [], []
    for label, cx, cz, _ in _CRAC_POS:
        crac_x.append(cx)
        crac_y.append(CRAC_H + 0.28)
        crac_z.append(cz)
        crac_t.append(label)

    crac_trace = go.Scatter3d(
        x=crac_x, y=crac_y, z=crac_z,
        text=crac_t, mode="text",
        textfont=dict(color="#00cc88", size=8),
        hoverinfo="none",
        showlegend=False,
    )

    return [room_trace, crac_trace]


def _aisle_label_trace() -> go.Scatter3d:
    """Text labels centred on each hot/cold aisle strip."""
    lx, ly, lz, texts, colors = [], [], [], [], []
    for room_idx in range(N_ROOMS):
        x_off = ROOM_B_X if room_idx else 0.0
        xmid  = x_off + (N_RACKS - 1) * RACK_PITCH / 2
        for row in range(N_ROWS - 1):
            za = row * ROW_PITCH + RACK_D + 0.05
            zb = (row + 1) * ROW_PITCH - 0.05
            lx.append(xmid)
            ly.append(0.18)
            lz.append((za + zb) / 2)
            is_cold = row % 2 == 0
            texts.append("COLD AISLE" if is_cold else "HOT AISLE")
            colors.append("#0055cc" if is_cold else "#cc3300")

    return go.Scatter3d(
        x=lx, y=ly, z=lz,
        text=texts, mode="text",
        textfont=dict(color=colors, size=7),
        hoverinfo="none",
        showlegend=False,
    )


# ─────────────────────────────────────────
# Public API
# ─────────────────────────────────────────

def make_3d_figure(telemetry: Optional[Dict]) -> go.Figure:
    """
    Build a polished Plotly 3D figure of the data center floor plan.

    Args:
        telemetry: Raw dict from GET /telemetry/live, or None.

    Returns:
        Plotly Figure for use with st.plotly_chart().
    """
    racks = (telemetry or {}).get("racks", [])
    cracs = (telemetry or {}).get("cracs", [])

    traces: list = [_floor_trace()]
    traces.extend(_aisle_traces())
    traces.extend(_ashrae_planes())
    if racks:
        traces.append(_rack_trace(racks))
    traces.extend(_alert_traces(racks))
    traces.append(_crac_trace(cracs))
    traces.extend(_label_trace())
    traces.append(_aisle_label_trace())

    now = datetime.now().strftime("%H:%M:%S")

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#2e2e2e",
        margin=dict(l=0, r=0, t=44, b=0),
        title=dict(
            text="Data Center Floor — Live Thermal View",
            font=dict(color="#c0d8f0", size=14),
            x=0.01,
        ),
        scene=dict(
            bgcolor="#2e2e2e",
            xaxis=dict(visible=False, backgroundcolor="#2e2e2e"),
            yaxis=dict(visible=False, backgroundcolor="#2e2e2e"),
            zaxis=dict(visible=False, backgroundcolor="#2e2e2e"),
            camera=dict(
                eye=dict(x=1.5, y=1.8, z=1.2),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=1, z=0),
            ),
            aspectmode="data",
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.0, x=1.0,
                xanchor="right",
                yanchor="top",
                buttons=[dict(
                    label="⟳ Reset Camera",
                    method="relayout",
                    args=["scene.camera", dict(
                        eye=dict(x=1.5, y=1.8, z=1.2),
                        center=dict(x=0, y=0, z=0),
                        up=dict(x=0, y=1, z=0),
                    )],
                )],
                bgcolor="#0e1f30",
                bordercolor="#1a3050",
                font=dict(color="#9ab8d0", size=10),
            )
        ],
        annotations=[
            dict(
                text="Rack height = power draw · Color = inlet temperature · Drag to rotate",
                x=0.5, y=1.01,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="#4a6a8a", size=10),
            ),
            dict(
                text=f"Last updated: {now}",
                x=1.0, y=0.0,
                xref="paper", yref="paper",
                showarrow=False,
                xanchor="right",
                font=dict(color="#3a5a7a", size=9),
            ),
        ],
        showlegend=False,
        height=680,
    )

    return fig
