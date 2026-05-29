"""
Data Center Digital Twin — Streamlit Operations Dashboard

Professional industrial-grade SCADA/Grafana-style interface for real-time
monitoring of cooling performance, power consumption, thermal conditions,
efficiency metrics, anomaly alerts, and optimization recommendations.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

from viz3d import make_3d_figure
from viz3d_three import make_three_html

# ─────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="DC Twin | Operations Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
APP_VERSION = "1.0.0"

# ─────────────────────────────────────────
# 2. CUSTOM CSS — Dark industrial theme
# ─────────────────────────────────────────

st.markdown(
    """
<style>
/* ── Base ──────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: #c8d8e8;
}
.stApp {
    background-color: #0f1923;
}
section[data-testid="stSidebar"] {
    background-color: #0a1420;
    border-right: 1px solid #1a2f45;
}
section[data-testid="stSidebar"] * {
    color: #a0b8cc !important;
}
/* ── Hide default Streamlit chrome ────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

/* ── Section headers ───────────────────── */
.section-header {
    display: flex;
    align-items: center;
    border-left: 4px solid #00aaff;
    padding-left: 12px;
    margin: 18px 0 10px 0;
    color: #e0ecf8;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

/* ── KPI cards ──────────────────────────── */
.kpi-card {
    background: #112030;
    border: 1px solid #1a3050;
    border-left: 4px solid #00aaff;
    border-radius: 6px;
    padding: 14px 16px;
    margin: 4px 0;
    min-height: 90px;
}
.kpi-card.amber  { border-left-color: #ffaa00; }
.kpi-card.red    { border-left-color: #ff4444; }
.kpi-card.green  { border-left-color: #00cc88; }
.kpi-card.purple { border-left-color: #aa66ff; }

.kpi-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: #6a8aaa;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e0f0ff;
    line-height: 1.1;
}
.kpi-delta {
    font-size: 0.78rem;
    margin-top: 4px;
}
.kpi-delta.up   { color: #ff6666; }
.kpi-delta.down { color: #00cc88; }
.kpi-delta.neutral { color: #7090b0; }
.kpi-sub {
    font-size: 0.72rem;
    color: #5a7a9a;
    margin-top: 2px;
}

/* ── Alert rows ─────────────────────────── */
.alert-row {
    display: flex;
    align-items: flex-start;
    border-radius: 5px;
    padding: 10px 12px;
    margin: 5px 0;
    background: #0e1f30;
    border-left: 4px solid #7090b0;
}
.alert-row.critical { border-left-color: #ff4444; background: #1a0e0e; }
.alert-row.warning  { border-left-color: #ffaa00; background: #1a1500; }
.alert-row.info     { border-left-color: #00aaff; background: #0d1f30; }

.alert-severity-badge {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 2px 7px;
    border-radius: 3px;
    margin-right: 8px;
    flex-shrink: 0;
    margin-top: 2px;
}
.badge-critical { background: #ff4444; color: #fff; }
.badge-warning  { background: #ffaa00; color: #000; }
.badge-info     { background: #00aaff; color: #000; }

.alert-content { flex: 1; }
.alert-message { font-size: 0.85rem; color: #d0e0f0; font-weight: 500; }
.alert-meta    { font-size: 0.72rem; color: #507090; margin-top: 3px; }

/* ── Confidence bar ─────────────────────── */
.conf-bar-outer {
    background: #1a2f45;
    border-radius: 3px;
    height: 6px;
    margin-top: 5px;
    width: 100%;
}
.conf-bar-inner {
    height: 6px;
    border-radius: 3px;
}
.conf-green  { background: #00cc88; }
.conf-amber  { background: #ffaa00; }
.conf-red    { background: #ff4444; }

/* ── Recommendation cards ───────────────── */
.rec-card {
    background: #0e1f30;
    border: 1px solid #1a3050;
    border-radius: 6px;
    padding: 12px 14px;
    margin: 6px 0;
}
.priority-badge {
    display: inline-block;
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 2px 8px;
    border-radius: 3px;
    margin-right: 8px;
}
.priority-high   { background: #ff4444; color: #fff; }
.priority-medium { background: #ffaa00; color: #000; }
.priority-low    { background: #00aaff; color: #000; }

.rec-action { font-size: 0.85rem; color: #c0d8f0; margin-top: 6px; line-height: 1.45; }
.rec-savings { font-size: 0.75rem; color: #00cc88; margin-top: 4px; }
.rec-rationale { font-size: 0.72rem; color: #4a6a8a; margin-top: 4px; line-height: 1.4; }

/* ── Event table ────────────────────────── */
.event-row { padding: 6px 0; border-bottom: 1px solid #1a2f45; font-size: 0.82rem; }

/* ── Sidebar ────────────────────────────── */
.sidebar-logo {
    font-family: monospace;
    font-size: 0.7rem;
    color: #2a6090;
    line-height: 1.3;
    white-space: pre;
    margin-bottom: 8px;
}
.sidebar-version {
    font-size: 0.68rem;
    color: #2a5070;
    margin-bottom: 12px;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid #1a2f45;
    margin: 12px 0;
}
/* ── Footer ─────────────────────────────── */
.footer {
    font-size: 0.68rem;
    color: #2a4a6a;
    text-align: center;
    padding: 12px 0 4px 0;
    border-top: 1px solid #1a2f45;
    margin-top: 24px;
}
/* ── General inputs ─────────────────────── */
.stSelectbox > div, .stSlider > div { color: #a0b8cc; }
div[data-testid="stMetricValue"] { color: #e0f0ff !important; }
div[data-testid="stExpander"] { background: #0e1f30; border: 1px solid #1a3050; border-radius: 6px; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────

@st.cache_data(ttl=5)
def fetch(endpoint: str, params: Optional[Dict] = None) -> Optional[Any]:
    """Fetch JSON from the API, returning None on error."""
    try:
        resp = httpx.get(f"{API_BASE_URL}{endpoint}", params=params or {}, timeout=8.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        return None
    except Exception:
        return None


def post_api(endpoint: str, body: Dict) -> Optional[Any]:
    try:
        resp = httpx.post(f"{API_BASE_URL}{endpoint}", json=body, timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ─────────────────────────────────────────
# Plotly dark theme helper
# ─────────────────────────────────────────

PLOTLY_DARK = dict(
    paper_bgcolor="#0f1923",
    plot_bgcolor="#0a1520",
    font=dict(color="#9ab8d0", family="Inter, Segoe UI, sans-serif", size=11),
    xaxis=dict(gridcolor="#1a3050", linecolor="#1a3050", zerolinecolor="#1a3050"),
    yaxis=dict(gridcolor="#1a3050", linecolor="#1a3050", zerolinecolor="#1a3050"),
    margin=dict(l=40, r=20, t=36, b=36),
)


def apply_dark_layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(**PLOTLY_DARK, title=dict(text=title, font=dict(size=13, color="#c0d8f0")))
    return fig


# ─────────────────────────────────────────
# KPI card renderer
# ─────────────────────────────────────────

def kpi_card(
    label: str,
    value: str,
    delta: str = "",
    delta_dir: str = "neutral",
    sub: str = "",
    accent: str = "",
) -> str:
    delta_html = f'<div class="kpi-delta {delta_dir}">{delta}</div>' if delta else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    cls = f"kpi-card {accent}" if accent else "kpi-card"
    return (
        f'<div class="{cls}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"{delta_html}{sub_html}"
        f"</div>"
    )


def section_header(title: str, icon: str = "") -> None:
    st.markdown(
        f'<div class="section-header">{icon}&nbsp;&nbsp;{title}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────
# 3. SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">'
        "╔════════════════╗\n"
        "║  DC TWIN v1.0  ║\n"
        "║  ◈ LIVE        ║\n"
        "╚════════════════╝"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sidebar-version">Operations Center · v{APP_VERSION}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="sidebar-divider"/>', unsafe_allow_html=True)

    refresh_interval = st.slider(
        "Refresh interval (s)", min_value=5, max_value=60, value=15, step=5
    )
    st.markdown('<hr class="sidebar-divider"/>', unsafe_allow_html=True)

    st.markdown("**Carbon Configuration**")
    carbon_factor = st.number_input(
        "Grid carbon intensity (kg CO₂/kWh)",
        min_value=0.01,
        max_value=1.0,
        value=0.233,
        step=0.01,
        format="%.3f",
    )
    st.markdown('<hr class="sidebar-divider"/>', unsafe_allow_html=True)

    st.markdown("**What-If Scenario**")
    scenario_type = st.selectbox(
        "Scenario",
        options=[
            "add_gpu_rack",
            "cooling_unit_failure",
            "higher_outside_temp",
            "ups_overload",
            "increase_it_load",
        ],
        format_func=lambda x: {
            "add_gpu_rack": "Add GPU Racks",
            "cooling_unit_failure": "CRAC Unit Failure",
            "higher_outside_temp": "Higher Outside Temp",
            "ups_overload": "UPS Overload",
            "increase_it_load": "Increase IT Load",
        }[x],
    )

    params: Dict[str, Any] = {}
    if scenario_type == "add_gpu_rack":
        params["n_racks"] = st.number_input("Number of racks", 1, 20, 4)
        params["kw_per_rack"] = st.number_input("kW per rack", 5.0, 30.0, 12.0, step=0.5)
    elif scenario_type == "higher_outside_temp":
        params["delta_c"] = st.number_input("Temperature increase (°C)", 1, 20, 8)
    elif scenario_type in ("ups_overload", "increase_it_load"):
        params["increase_pct"] = st.number_input("Increase (%)", 5, 100, 25)
    elif scenario_type == "cooling_unit_failure":
        params["crac_id"] = "A-CRAC-1"

    run_sim = st.button("▶  Run Simulation", use_container_width=True)

    st.markdown('<hr class="sidebar-divider"/>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sidebar-version">Last refresh: {datetime.now().strftime("%H:%M:%S")}</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────
# Load data
# ─────────────────────────────────────────

health = fetch("/health")
telemetry = fetch("/telemetry/live")
pue_data = fetch("/metrics/pue")
alerts_data = fetch("/alerts") or []
recs_data = fetch("/recommendations") or []
stranded_data = fetch("/metrics/stranded-capacity")
carbon_data = fetch("/metrics/carbon", {"carbon_factor": carbon_factor})
events_data = fetch("/events/log", {"hours": 6, "page_size": 100})

api_ok = health is not None

if not api_ok:
    st.warning(
        "⚠️  Cannot reach API at **" + API_BASE_URL + "**.  "
        "Start the backend with `uvicorn backend.main:app --reload --port 8000`."
    )

# ─────────────────────────────────────────
# Page title bar
# ─────────────────────────────────────────

col_title, col_status = st.columns([5, 1])
with col_title:
    st.markdown(
        '<h1 style="color:#e0f0ff;font-size:1.6rem;margin:0;padding:0;font-weight:700;">'
        "⚡ Data Center Digital Twin — Operations Center"
        "</h1>",
        unsafe_allow_html=True,
    )
with col_status:
    status_color = "#00cc88" if api_ok else "#ff4444"
    status_text = "LIVE" if api_ok else "OFFLINE"
    st.markdown(
        f'<div style="text-align:right;padding-top:6px;">'
        f'<span style="background:{status_color};color:#000;font-size:0.7rem;'
        f'font-weight:700;padding:3px 10px;border-radius:3px;letter-spacing:0.08em;">'
        f"{status_text}</span></div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────
# 4. TOP KPI ROW
# ─────────────────────────────────────────

section_header("Key Performance Indicators", "📊")
k1, k2, k3, k4, k5, k6 = st.columns(6)

# IT Load
it_load = 0.0
if telemetry and telemetry.get("racks"):
    it_load = sum(r["power_kw"] for r in telemetry["racks"])

with k1:
    st.markdown(
        kpi_card("Total IT Load", f"{it_load:.1f} kW", sub="80 racks aggregated"),
        unsafe_allow_html=True,
    )

# PUE
live_pue = pue_data["live_pue"] if pue_data else 0.0
pue_avg = pue_data["rolling_avg_pue"] if pue_data else 0.0
pue_accent = "green" if live_pue < 1.4 else ("amber" if live_pue < 1.6 else "red")
pue_delta_dir = "up" if live_pue > pue_avg else "down"
with k2:
    st.markdown(
        kpi_card(
            "PUE — Live",
            f"{live_pue:.3f}" if live_pue else "—",
            delta=f"avg {pue_avg:.3f}" if pue_avg else "",
            delta_dir=pue_delta_dir,
            sub="Green Grid target < 1.2",
            accent=pue_accent,
        ),
        unsafe_allow_html=True,
    )

# Projected PUE
proj_pue = pue_data.get("projected_pue_24h") if pue_data else None
trend = pue_data.get("trend_direction", "stable") if pue_data else "stable"
trend_arrow = "▲" if trend == "degrading" else ("▼" if trend == "improving" else "→")
trend_dir = "up" if trend == "degrading" else ("down" if trend == "improving" else "neutral")
with k3:
    st.markdown(
        kpi_card(
            "Projected PUE (24h)",
            f"{proj_pue:.3f}" if proj_pue else "—",
            delta=f"{trend_arrow} {trend}",
            delta_dir=trend_dir,
            sub="Linear regression forecast",
            accent="amber" if trend == "degrading" else "green",
        ),
        unsafe_allow_html=True,
    )

# Stranded Capacity
stranded_kw = stranded_data["stranded_capacity_kw"] if stranded_data else 0.0
constraint = stranded_data["binding_constraint"] if stranded_data else "—"
sc_accent = "red" if stranded_kw < 50 else ("amber" if stranded_kw < 150 else "green")
with k4:
    st.markdown(
        kpi_card(
            "Stranded Capacity",
            f"{stranded_kw:.0f} kW",
            sub=constraint.replace("-", " ").title() if constraint else "",
            accent=sc_accent,
        ),
        unsafe_allow_html=True,
    )

# Carbon
carbon_kghr = carbon_data["carbon_kg_per_hr"] if carbon_data else 0.0
carbon_tyr = carbon_data["carbon_tonnes_per_year"] if carbon_data else 0.0
with k5:
    st.markdown(
        kpi_card(
            "Carbon Intensity",
            f"{carbon_kghr:.1f} kg/hr",
            sub=f"{carbon_tyr:.0f} t CO₂/yr",
            accent="amber",
        ),
        unsafe_allow_html=True,
    )

# Alerts
critical_count = sum(1 for a in alerts_data if a.get("severity") == "critical")
warning_count = sum(1 for a in alerts_data if a.get("severity") == "warning")
alert_accent = "red" if critical_count > 0 else ("amber" if warning_count > 0 else "green")
with k6:
    st.markdown(
        kpi_card(
            "Active Alerts",
            str(len(alerts_data)),
            delta=f"🔴 {critical_count} crit  🟡 {warning_count} warn",
            delta_dir="up" if critical_count > 0 else "neutral",
            accent=alert_accent,
        ),
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 5. VISUALIZATION — 2D Heatmap / 3D Plotly / 3D WebGL
# ─────────────────────────────────────────

section_header("Data Center Floor Visualization", "🏗️")

tab_2d, tab_3d_plotly, tab_3d_webgl = st.tabs([
    "📊  2D Heatmap",
    "📦  3D Plotly",
    "✨  3D WebGL (Three.js)",
])

# ── Tab 1: classic 2D heatmap ────────────────────────────────────────────────
with tab_2d:
    if telemetry and telemetry.get("racks"):
        racks = telemetry["racks"]
        rooms = ["Room A", "Room B"]
        rows_labels, z_data, text_data = [], [], []

        for room in rooms:
            for row_num in range(1, 5):
                row_label = f"{room} R{row_num}"
                rows_labels.append(row_label)
                row_racks = sorted(
                    [r for r in racks
                     if r.get("room") == room and r.get("row") == f"Row {row_num}"],
                    key=lambda r: r.get("position", 0),
                )
                temps     = [r["inlet_temp_c"]  for r in row_racks[:10]]
                powers    = [r["power_kw"]       for r in row_racks[:10]]
                outlets   = [r["outlet_temp_c"]  for r in row_racks[:10]]
                ashraes   = [r.get("ashrae_class", "A1") for r in row_racks[:10]]
                asset_ids = [r["asset_id"]       for r in row_racks[:10]]
                while len(temps) < 10:
                    temps.append(None); powers.append(None)
                    outlets.append(None); ashraes.append("—"); asset_ids.append("—")
                z_data.append(temps)
                text_data.append([
                    f"<b>{aid}</b><br>Inlet: {t:.1f}°C<br>Outlet: {o:.1f}°C"
                    f"<br>Power: {p:.1f} kW<br>ASHRAE: {a}"
                    if t is not None else "—"
                    for aid, t, o, p, a in zip(asset_ids, temps, outlets, powers, ashraes)
                ])

        fig_heat = go.Figure(go.Heatmap(
            z=z_data,
            x=[f"Rack {i}" for i in range(1, 11)],
            y=rows_labels,
            text=text_data,
            hovertemplate="%{text}<extra></extra>",
            colorscale=[[0.0,"#0088cc"],[0.3,"#00cc88"],
                        [0.6,"#ffaa00"],[0.8,"#ff6600"],[1.0,"#ff2222"]],
            zmin=16, zmax=38,
            colorbar=dict(
                title=dict(text="°C", font=dict(color="#9ab8d0")),
                tickfont=dict(color="#9ab8d0"),
                bgcolor="#0a1520", outlinecolor="#1a3050",
            ),
            xgap=2, ygap=2,
        ))
        apply_dark_layout(fig_heat, "Floor Thermal Map — Rack Inlet Temperature (°C)")
        fig_heat.update_layout(height=340)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Waiting for thermal data…")

# ── Tab 2: Plotly 3D mesh ─────────────────────────────────────────────────────
with tab_3d_plotly:
    st.caption(
        "Racks coloured by inlet temperature · height = power draw · "
        "drag to rotate · scroll to zoom"
    )
    try:
        fig_3d = make_3d_figure(telemetry)
        st.plotly_chart(fig_3d, use_container_width=True)
    except Exception as e:
        st.warning(f"3D Plotly view unavailable: {e}")

# ── Tab 3: Three.js WebGL ─────────────────────────────────────────────────────
with tab_3d_webgl:
    st.caption(
        "WebGL · physically-based lighting · hot racks pulse · "
        "drag to rotate · scroll to zoom · hover for details"
    )
    try:
        html_src = make_three_html(telemetry, width=1180, height=700)
        components.html(html_src, height=720, scrolling=False)
    except Exception as e:
        st.warning(f"3D WebGL view unavailable: {e}")

# ─────────────────────────────────────────
# 6. TWO-COLUMN ROW: Temperature + Power trends
# ─────────────────────────────────────────

col_left, col_right = st.columns(2)

# ── Left: Temperature trend with ASHRAE overlay + PUE forecast ──

with col_left:
    section_header("Temperature Trends & PUE Forecast", "📈")

    if pue_data and pue_data.get("pue_history"):
        history = pue_data["pue_history"]
        df_hist = pd.DataFrame(history)
        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])

        fig_temp = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.55, 0.45],
            vertical_spacing=0.06,
        )

        # PUE trend + forecast
        fig_temp.add_trace(
            go.Scatter(
                x=df_hist["timestamp"],
                y=df_hist["pue"],
                mode="lines",
                name="PUE",
                line=dict(color="#00aaff", width=2),
            ),
            row=1, col=1,
        )

        # ASHRAE A1 and A2 reference lines on temperature chart
        if "outside_air_temp_c" in df_hist.columns:
            fig_temp.add_trace(
                go.Scatter(
                    x=df_hist["timestamp"],
                    y=df_hist["outside_air_temp_c"],
                    mode="lines",
                    name="Outside Air °C",
                    line=dict(color="#aa66ff", width=1.5, dash="dot"),
                    yaxis="y3",
                ),
                row=1, col=1,
            )

        # IT load trend
        if "it_load_kw" in df_hist.columns:
            fig_temp.add_trace(
                go.Scatter(
                    x=df_hist["timestamp"],
                    y=df_hist["it_load_kw"],
                    mode="lines",
                    name="IT Load kW",
                    line=dict(color="#00cc88", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0,204,136,0.06)",
                ),
                row=2, col=1,
            )

        # ASHRAE reference zones on PUE chart
        if len(df_hist) > 1:
            x_range = [df_hist["timestamp"].min(), df_hist["timestamp"].max()]
            # A1 recommended PUE target band
            fig_temp.add_hrect(
                y0=1.0, y1=1.4,
                fillcolor="rgba(0,204,136,0.07)",
                line_width=0,
                annotation_text="Target PUE",
                annotation_font_size=9,
                annotation_font_color="#00cc88",
                row=1, col=1,
            )
            fig_temp.add_hrect(
                y0=1.4, y1=1.8,
                fillcolor="rgba(255,170,0,0.06)",
                line_width=0,
                row=1, col=1,
            )
            fig_temp.add_hline(
                y=1.8,
                line_dash="dash",
                line_color="#ff4444",
                line_width=1,
                annotation_text="PUE 1.8 Alert",
                annotation_font_size=9,
                annotation_font_color="#ff4444",
                row=1, col=1,
            )

        apply_dark_layout(fig_temp, "")
        fig_temp.update_layout(
            height=380,
            legend=dict(
                font=dict(size=10, color="#7090b0"),
                bgcolor="rgba(0,0,0,0)",
                orientation="h",
                y=1.02,
            ),
            paper_bgcolor="#0f1923",
            plot_bgcolor="#0a1520",
        )
        fig_temp.update_yaxes(title_text="PUE", title_font=dict(size=10), row=1, col=1)
        fig_temp.update_yaxes(title_text="IT Load (kW)", title_font=dict(size=10), row=2, col=1)
        st.plotly_chart(fig_temp, use_container_width=True)
    else:
        st.info("Accumulating PUE history...")

    # Rack inlet temperature chart with ASHRAE bands
    if telemetry and telemetry.get("racks"):
        racks = telemetry["racks"]
        sorted_racks = sorted(racks, key=lambda r: r.get("inlet_temp_c", 0), reverse=True)[:20]
        rack_ids = [r["asset_id"] for r in sorted_racks]
        rack_temps = [r["inlet_temp_c"] for r in sorted_racks]

        colors = [
            "#ff4444" if t > 32 else ("#ffaa00" if t > 28 else "#00cc88")
            for t in rack_temps
        ]

        fig_rack = go.Figure()
        fig_rack.add_trace(
            go.Bar(
                x=rack_ids,
                y=rack_temps,
                marker=dict(color=colors, line=dict(width=0)),
                name="Inlet Temp °C",
                hovertemplate="<b>%{x}</b><br>%{y:.1f}°C<extra></extra>",
            )
        )
        # ASHRAE A1 limit
        fig_rack.add_hline(y=32, line_dash="dash", line_color="#ffaa00", line_width=1,
                           annotation_text="A1 32°C", annotation_font_size=9,
                           annotation_font_color="#ffaa00")
        # ASHRAE A2 limit
        fig_rack.add_hline(y=35, line_dash="dash", line_color="#ff4444", line_width=1,
                           annotation_text="A2 35°C", annotation_font_size=9,
                           annotation_font_color="#ff4444")
        # A1 band
        fig_rack.add_hrect(y0=15, y1=32, fillcolor="rgba(0,204,136,0.06)", line_width=0)
        # A2 band
        fig_rack.add_hrect(y0=32, y1=35, fillcolor="rgba(255,170,0,0.07)", line_width=0)

        apply_dark_layout(fig_rack, "Top-20 Hottest Rack Inlet Temperatures with ASHRAE Overlay")
        fig_rack.update_layout(
            height=260,
            xaxis=dict(tickfont=dict(size=8)),
            yaxis_title="Temp (°C)",
        )
        st.plotly_chart(fig_rack, use_container_width=True)

# ── Right: Power trends ────────────────────────────────────────────

with col_right:
    section_header("Power Monitoring", "⚡")

    if telemetry:
        upses = telemetry.get("upses", [])
        pdus = telemetry.get("pdus", [])
        racks_t = telemetry.get("racks", [])

        # UPS load gauge chart
        fig_ups = go.Figure()
        for ups in upses:
            fig_ups.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=ups.get("load_pct", 0),
                    title=dict(text=ups["asset_id"], font=dict(size=11, color="#9ab8d0")),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor="#1a3050"),
                        bar=dict(color="#00aaff"),
                        steps=[
                            dict(range=[0, 80], color="#0a1520"),
                            dict(range=[80, 90], color="#332200"),
                            dict(range=[90, 100], color="#330000"),
                        ],
                        threshold=dict(
                            line=dict(color="#ff4444", width=2),
                            thickness=0.8,
                            value=90,
                        ),
                        bgcolor="#0a1520",
                        bordercolor="#1a3050",
                    ),
                    number=dict(suffix=" %", font=dict(color="#c0d8f0")),
                    domain=dict(
                        x=[0, 0.5] if ups["asset_id"] == "UPS-1" else [0.5, 1.0],
                        y=[0, 1],
                    ),
                )
            )
        apply_dark_layout(fig_ups, "UPS Load (%)")
        fig_ups.update_layout(height=200)
        st.plotly_chart(fig_ups, use_container_width=True)

        # PDU load bar chart
        if pdus:
            pdu_ids = [p["asset_id"] for p in pdus]
            pdu_loads = [p["load_pct"] for p in pdus]
            pdu_colors = [
                "#ff4444" if l >= 90 else ("#ffaa00" if l >= 80 else "#00aaff")
                for l in pdu_loads
            ]
            fig_pdu = go.Figure()
            fig_pdu.add_trace(
                go.Bar(
                    x=pdu_ids,
                    y=pdu_loads,
                    marker=dict(color=pdu_colors),
                    name="PDU Load %",
                    hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
                )
            )
            fig_pdu.add_hline(y=80, line_dash="dash", line_color="#ffaa00", line_width=1,
                              annotation_text="80% warn", annotation_font_size=9)
            fig_pdu.add_hline(y=90, line_dash="dash", line_color="#ff4444", line_width=1,
                              annotation_text="90% crit", annotation_font_size=9)
            apply_dark_layout(fig_pdu, "PDU Load (%) by Unit")
            fig_pdu.update_layout(height=210, yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_pdu, use_container_width=True)

        # Power distribution: rack IT vs cooling vs overhead
        if racks_t:
            it_kw = sum(r["power_kw"] for r in racks_t)
            cooling_kw = sum(c["cooling_load_kw"] for c in telemetry.get("cracs", []))
            overhead_kw = it_kw * 0.05
            fig_pie = go.Figure(
                go.Pie(
                    labels=["IT Load", "Cooling", "Overhead"],
                    values=[it_kw, cooling_kw, overhead_kw],
                    hole=0.55,
                    marker=dict(colors=["#00aaff", "#00cc88", "#ffaa00"]),
                    textfont=dict(color="#c0d8f0", size=11),
                    hovertemplate="<b>%{label}</b><br>%{value:.1f} kW<br>%{percent}<extra></extra>",
                )
            )
            apply_dark_layout(fig_pie, "Facility Power Distribution")
            fig_pie.update_layout(
                height=230,
                showlegend=True,
                legend=dict(
                    font=dict(size=10, color="#7090b0"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                annotations=[
                    dict(
                        text=f"{it_kw:.0f}<br>kW IT",
                        x=0.5, y=0.5,
                        font=dict(size=13, color="#e0f0ff"),
                        showarrow=False,
                    )
                ],
            )
            st.plotly_chart(fig_pie, use_container_width=True)

# ─────────────────────────────────────────
# Carbon trend chart
# ─────────────────────────────────────────

section_header("Carbon Emissions Trend", "🌍")
if carbon_data and carbon_data.get("carbon_history"):
    ch = carbon_data["carbon_history"]
    df_c = pd.DataFrame(ch)
    df_c["timestamp"] = pd.to_datetime(df_c["timestamp"])

    fig_carbon = go.Figure()
    fig_carbon.add_trace(
        go.Scatter(
            x=df_c["timestamp"],
            y=df_c["carbon_kg_per_hr"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(0,204,136,0.07)",
            line=dict(color="#00cc88", width=2),
            name="kg CO₂/hr",
            hovertemplate="<b>%{x|%H:%M}</b><br>%{y:.1f} kg CO₂/hr<extra></extra>",
        )
    )
    # Target line at 80% of current value as aspirational goal
    if len(df_c) > 0:
        target = df_c["carbon_kg_per_hr"].mean() * 0.8
        fig_carbon.add_hline(
            y=target,
            line_dash="dot",
            line_color="#ffaa00",
            line_width=1,
            annotation_text=f"Target: {target:.1f} kg/hr (−20%)",
            annotation_font_size=9,
            annotation_font_color="#ffaa00",
        )
    apply_dark_layout(fig_carbon, f"Carbon Intensity (kg CO₂/hr) — Factor: {carbon_factor:.3f} kgCO₂/kWh")
    fig_carbon.update_layout(height=220, yaxis_title="kg CO₂/hr")
    st.plotly_chart(fig_carbon, use_container_width=True)

# ─────────────────────────────────────────
# 7. ALERTS + RECOMMENDATIONS
# ─────────────────────────────────────────

col_al, col_rec = st.columns(2)

# ── Alerts panel ──────────────────────────

with col_al:
    section_header("Active Alerts", "🚨")
    if not alerts_data:
        st.markdown(
            '<div style="color:#00cc88;font-size:0.85rem;padding:12px;background:#0a1f10;'
            'border:1px solid #1a3020;border-radius:5px;">'
            "✅  No active alerts — all systems nominal."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        for alert in sorted(
            alerts_data,
            key=lambda a: {"critical": 0, "warning": 1, "info": 2}.get(a.get("severity", "info"), 3),
        ):
            sev = alert.get("severity", "info")
            conf = alert.get("confidence", 0)
            conf_class = "conf-red" if conf > 70 else ("conf-amber" if conf > 40 else "conf-green")
            badge_class = f"badge-{sev}"

            st.markdown(
                f'<div class="alert-row {sev}">'
                f'  <span class="alert-severity-badge {badge_class}">{sev.upper()}</span>'
                f'  <div class="alert-content">'
                f'    <div class="alert-message">{alert.get("message", "")}</div>'
                f'    <div class="alert-meta">'
                f'      Asset: {alert.get("asset_id", "")}  •  '
                f'      Duration: {alert.get("duration_cycles", 1)} cycle(s)  •  '
                f'      Confidence: {conf}%'
                f'    </div>'
                f'    <div class="conf-bar-outer">'
                f'      <div class="conf-bar-inner {conf_class}" style="width:{conf}%"></div>'
                f'    </div>'
                f'    <div class="alert-meta" style="color:#3a5a7a;font-size:0.68rem;margin-top:3px;">'
                f'      {alert.get("explanation", "")[:120]}…'
                f'    </div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ── Recommendations panel ──────────────────

with col_rec:
    section_header("Optimisation Recommendations", "💡")
    if not recs_data:
        st.markdown(
            '<div style="color:#00aaff;font-size:0.85rem;padding:12px;background:#0a1520;'
            'border:1px solid #1a2f45;border-radius:5px;">'
            "ℹ️  No recommendations at this time — system operating within optimal parameters."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        for rec in recs_data:
            priority = rec.get("priority", "low")
            badge_cls = f"priority-{priority}"
            savings = rec.get("expected_savings_kw", 0)
            pue_imp = rec.get("pue_impact", 0)

            savings_text = (
                f"Est. savings: <b>{savings:.1f} kW</b>"
                + (f"  |  PUE −{pue_imp:.3f}" if pue_imp > 0 else "")
                if savings > 0 else ""
            )

            st.markdown(
                f'<div class="rec-card">'
                f'  <span class="priority-badge {badge_cls}">{priority.upper()}</span>'
                f'  <span style="font-size:0.72rem;color:#4a6a8a;">{rec.get("recommendation_type", "").replace("_", " ").title()}</span>'
                f'  <div class="rec-action">{rec.get("action_text", "")}</div>'
                + (f'  <div class="rec-savings">{savings_text}</div>' if savings_text else "")
                + f'  <div class="rec-rationale">{rec.get("rationale", "")[:160]}…</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────
# 8. EVENT LOG TABLE
# ─────────────────────────────────────────

section_header("Operations Audit Trail", "📋")

log_col1, log_col2, log_col3 = st.columns([2, 2, 1])
with log_col1:
    log_filter_sev = st.selectbox(
        "Filter severity", ["All", "critical", "warning", "info", "high", "medium", "low"]
    )
with log_col2:
    log_filter_hours = st.selectbox("Time range", [1, 6, 24, 48], index=1, format_func=lambda h: f"Last {h}h")
with log_col3:
    st.markdown("<div style='padding-top:24px'></div>", unsafe_allow_html=True)
    refresh_log = st.button("↻ Refresh", use_container_width=True)

params_log: Dict[str, Any] = {"hours": log_filter_hours, "page_size": 100}
if log_filter_sev != "All":
    params_log["severity"] = log_filter_sev

events_filtered = fetch("/events/log", params_log)
events_list = events_filtered.get("events", []) if events_filtered else []

if events_list:
    sev_colors = {
        "critical": "#ff4444",
        "warning": "#ffaa00",
        "info": "#00aaff",
        "high": "#ff4444",
        "medium": "#ffaa00",
        "low": "#7090b0",
    }

    df_events = pd.DataFrame(events_list)
    df_events["timestamp"] = pd.to_datetime(df_events["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Style the table
    def color_severity(val):
        color = sev_colors.get(str(val).lower(), "#7090b0")
        return f"color: {color}; font-weight: 600"

    display_cols = ["timestamp", "event_type", "asset_id", "severity", "message", "confidence"]
    available = [c for c in display_cols if c in df_events.columns]
    df_show = df_events[available].fillna("—")

    styled = df_show.style.applymap(color_severity, subset=["severity"] if "severity" in available else [])
    st.dataframe(
        styled,
        use_container_width=True,
        height=280,
    )
    st.caption(
        f"{events_filtered.get('total', len(events_list))} total events in last {log_filter_hours}h — "
        f"showing {len(events_list)}"
    )
else:
    st.info("No events in the selected time range.")

# ─────────────────────────────────────────
# 9. WHAT-IF RESULTS
# ─────────────────────────────────────────

with st.expander("🔬  What-If Simulation Results", expanded="whatif_result" in st.session_state):
    if run_sim:
        with st.spinner("Running scenario simulation..."):
            result = post_api(
                "/simulate/what-if",
                {"scenario_type": scenario_type, "parameters": params},
            )
        if result:
            st.session_state["whatif_result"] = result
        else:
            st.error("Simulation failed — is the backend running?")

    result = st.session_state.get("whatif_result")
    if result:
        r1, r2, r3, r4 = st.columns(4)
        pue_delta = result["projected_pue"] - result["current_pue"]
        sc_delta = result["projected_stranded_kw"] - result["current_stranded_kw"]

        with r1:
            st.markdown(
                kpi_card(
                    "Current PUE", f"{result['current_pue']:.3f}",
                    sub="Before scenario"
                ),
                unsafe_allow_html=True,
            )
        with r2:
            pue_dir = "up" if pue_delta > 0 else "down"
            st.markdown(
                kpi_card(
                    "Projected PUE",
                    f"{result['projected_pue']:.3f}",
                    delta=f"{'▲' if pue_delta > 0 else '▼'} {abs(pue_delta):.3f}",
                    delta_dir=pue_dir,
                    accent="red" if pue_delta > 0.1 else "amber",
                ),
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                kpi_card(
                    "Current Stranded", f"{result['current_stranded_kw']:.0f} kW",
                    sub="Before scenario"
                ),
                unsafe_allow_html=True,
            )
        with r4:
            sc_dir = "up" if sc_delta < 0 else "down"
            st.markdown(
                kpi_card(
                    "Projected Stranded",
                    f"{result['projected_stranded_kw']:.0f} kW",
                    delta=f"{'▼' if sc_delta < 0 else '▲'} {abs(sc_delta):.0f} kW",
                    delta_dir=sc_dir,
                    accent="red" if sc_delta < -100 else "amber",
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div style="background:#0e1f30;border-left:4px solid #00aaff;padding:12px 16px;'
            f'border-radius:5px;margin:12px 0;font-size:0.88rem;color:#c0d8f0;">'
            f"<b>Summary:</b> {result['summary']}"
            f"</div>",
            unsafe_allow_html=True,
        )

        if result.get("risks"):
            st.markdown("**Risk Assessment:**")
            for risk in result["risks"]:
                st.markdown(f"- {risk}")

        if result.get("new_alerts"):
            st.markdown("**Alerts that would fire:**")
            for a in result["new_alerts"]:
                sev_color = "#ff4444" if "CRITICAL" in a else "#ffaa00"
                st.markdown(
                    f'<div style="color:{sev_color};font-size:0.85rem;padding:4px 0">⚠ {a}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            '<div style="color:#4a6a8a;font-size:0.85rem;">'
            "Configure a scenario in the sidebar and click ▶ Run Simulation."
            "</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────
# 10. FOOTER
# ─────────────────────────────────────────

st.markdown(
    f'<div class="footer">'
    f"DC Twin Operations Center · v{APP_VERSION} · "
    f"Simulated Data · "
    f"Refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    f"</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────
# Auto-refresh
# ─────────────────────────────────────────

time.sleep(refresh_interval)
st.rerun()
