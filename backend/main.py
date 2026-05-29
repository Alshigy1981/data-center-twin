"""
Data Center Digital Twin Platform — FastAPI Backend

Exposes the full REST API surface for the digital twin: asset inventory,
live telemetry, efficiency metrics, alerts, recommendations, what-if
simulation, capacity planning, carbon reporting, and audit trail.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.database import (
    AlertDB,
    EventDB,
    LatestTelemetryDB,
    PUEHistoryDB,
    RecommendationDB,
    TelemetryDB,
    get_db,
    get_simulation_cycle_count,
)
from backend.metrics import (
    DEFAULT_CARBON_FACTOR,
    TOTAL_CRAC_RATED_KW,
    TOTAL_PDU_RATED_KW,
    TOTAL_UPS_RATED_KW,
    calculate_carbon,
    calculate_pue,
    calculate_pue_trend,
    calculate_stranded_capacity,
    forecast_pue,
    pue_forecast_series,
    project_scenario,
)
from backend.models import (
    Alert,
    AlertSeverity,
    AssetInventory,
    BindingConstraint,
    CarbonMetrics,
    ChillerAsset,
    ChillerTelemetry,
    CRACAsset,
    CRACTelemetry,
    EventLogEntry,
    EventLogResponse,
    HealthResponse,
    PDUAsset,
    PDUTelemetry,
    Priority,
    PUEMetrics,
    RackAsset,
    RackTelemetry,
    Recommendation,
    StrandedCapacityMetrics,
    TelemetryHistory,
    TelemetrySnapshot,
    TrendDirection,
    UPSAsset,
    UPSTelemetry,
    WhatIfRequest,
    WhatIfResponse,
)
from backend.simulator import build_asset_inventory, get_last_snapshot, start_simulation

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
SIMULATION_INTERVAL = int(os.getenv("SIMULATION_INTERVAL", "30"))
_start_time = time.time()

# ─────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────

app = FastAPI(
    title="Data Center Digital Twin API",
    description=(
        "Real-time digital twin platform for data center cooling optimisation "
        "and power monitoring.  Provides live telemetry, efficiency metrics, "
        "anomaly alerts, capacity planning, carbon reporting, and what-if simulation."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cached asset inventory (static — no need to re-query on every request)
_asset_inventory: Optional[AssetInventory] = None


def _get_inventory() -> AssetInventory:
    global _asset_inventory
    if _asset_inventory is None:
        _asset_inventory = build_asset_inventory()
    return _asset_inventory


# ─────────────────────────────────────────
# Startup / Shutdown
# ─────────────────────────────────────────

@app.on_event("startup")
async def on_startup() -> None:
    """Initialise database and start background simulation thread."""
    start_simulation(interval=SIMULATION_INTERVAL)
    logger.info("DC Twin API started.  Simulation interval: %ds", SIMULATION_INTERVAL)


# ─────────────────────────────────────────
# Helper: latest telemetry from DB
# ─────────────────────────────────────────

def _latest_rows(db: Session) -> List[LatestTelemetryDB]:
    return db.query(LatestTelemetryDB).all()


def _parse_snapshot_from_db(rows: List[LatestTelemetryDB]) -> TelemetrySnapshot:
    """Reconstruct a TelemetrySnapshot from the latest_telemetry DB rows."""
    racks, cracs, chillers, upses, pdus = [], [], [], [], []
    latest_ts = datetime.utcnow()
    cycle = 0
    outside_temp = 15.0

    for row in rows:
        data = json.loads(row.data_json)
        if row.timestamp > latest_ts or cycle == 0:
            latest_ts = row.timestamp
        cycle = max(cycle, row.cycle_id)

        if row.asset_type == "rack":
            racks.append(RackTelemetry(**data))
        elif row.asset_type == "crac":
            cracs.append(CRACTelemetry(**data))
        elif row.asset_type == "chiller":
            chillers.append(ChillerTelemetry(**data))
        elif row.asset_type == "ups":
            upses.append(UPSTelemetry(**data))
        elif row.asset_type == "pdu":
            pdus.append(PDUTelemetry(**data))

    # Try to get outside temp from last snapshot in memory
    mem = get_last_snapshot()
    if mem:
        outside_temp = mem.outside_air_temp_c

    return TelemetrySnapshot(
        timestamp=latest_ts,
        cycle_id=cycle,
        racks=racks,
        cracs=cracs,
        chillers=chillers,
        upses=upses,
        pdus=pdus,
        outside_air_temp_c=outside_temp,
    )


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns service status, uptime, version, and simulation cycle count.",
    tags=["Operations"],
)
def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    try:
        cycle_count = get_simulation_cycle_count(db)
        db_status = "ok"
    except Exception as exc:
        db_status = f"error: {exc}"
        cycle_count = 0

    last_sim: Optional[datetime] = None
    mem = get_last_snapshot()
    if mem:
        last_sim = mem.timestamp

    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        uptime_seconds=round(time.time() - _start_time, 1),
        db_status=db_status,
        simulation_cycles=cycle_count,
        last_simulation=last_sim,
    )


@app.get(
    "/assets",
    response_model=AssetInventory,
    summary="Full asset inventory",
    description="Returns the complete hierarchical asset inventory: racks, CRACs, chillers, UPS, PDUs.",
    tags=["Assets"],
)
def get_assets() -> AssetInventory:
    return _get_inventory()


@app.get(
    "/telemetry/live",
    response_model=TelemetrySnapshot,
    summary="Live telemetry snapshot",
    description="Returns the most recent telemetry reading for every asset.",
    tags=["Telemetry"],
)
def telemetry_live(db: Session = Depends(get_db)) -> TelemetrySnapshot:
    # Prefer in-memory snapshot (zero DB overhead) if available
    mem = get_last_snapshot()
    if mem and mem.racks:
        return mem

    rows = _latest_rows(db)
    if not rows:
        raise HTTPException(status_code=503, detail="No telemetry available yet — simulation starting.")
    return _parse_snapshot_from_db(rows)


@app.get(
    "/telemetry/history",
    response_model=List[Dict[str, Any]],
    summary="Historical telemetry",
    description="Returns the last N telemetry readings per asset type.",
    tags=["Telemetry"],
)
def telemetry_history(
    limit: int = Query(default=60, ge=1, le=500, description="Max readings to return per asset"),
    asset_id: Optional[str] = Query(default=None, description="Filter by asset ID"),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    query = db.query(TelemetryDB).order_by(TelemetryDB.timestamp.desc())
    if asset_id:
        query = query.filter(TelemetryDB.asset_id == asset_id)
    rows = query.limit(limit).all()

    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "cycle_id": r.cycle_id,
            "asset_id": r.asset_id,
            "asset_type": r.asset_type,
            **json.loads(r.data_json),
        }
        for r in rows
    ]


@app.get(
    "/metrics/pue",
    response_model=PUEMetrics,
    summary="Power Usage Effectiveness metrics",
    description="Returns live PUE, rolling average, trend direction, and 24-hour forecast.",
    tags=["Metrics"],
)
def metrics_pue(db: Session = Depends(get_db)) -> PUEMetrics:
    snap = get_last_snapshot()
    if not snap or not snap.racks:
        raise HTTPException(status_code=503, detail="No telemetry available.")

    it_load = sum(r.power_kw for r in snap.racks)
    cooling_load = sum(c.cooling_load_kw for c in snap.cracs)
    overhead = it_load * 0.05
    live_pue = calculate_pue(it_load, cooling_load)

    # Fetch PUE history for rolling average and forecast
    history_rows = (
        db.query(PUEHistoryDB)
        .order_by(PUEHistoryDB.timestamp.asc())
        .limit(120)
        .all()
    )
    pue_vals = [h.pue for h in history_rows]
    rolling_avg = round(sum(pue_vals[-60:]) / max(len(pue_vals[-60:]), 1), 4) if pue_vals else live_pue
    trend = calculate_pue_trend(pue_vals[-30:]) if len(pue_vals) >= 3 else TrendDirection.STABLE

    history_list = [
        {
            "timestamp": h.timestamp.isoformat(),
            "pue": h.pue,
            "it_load_kw": h.it_load_kw,
            "outside_air_temp_c": h.outside_air_temp_c,
        }
        for h in history_rows[-60:]
    ]

    pue_dicts = [{"pue": h.pue, "timestamp": h.timestamp.isoformat()} for h in history_rows]
    projected = forecast_pue(pue_dicts, horizon_steps=24) if len(pue_dicts) >= 5 else None

    return PUEMetrics(
        timestamp=datetime.utcnow(),
        live_pue=live_pue,
        rolling_avg_pue=rolling_avg,
        it_load_kw=round(it_load, 2),
        cooling_load_kw=round(cooling_load, 2),
        overhead_kw=round(overhead, 2),
        total_facility_kw=round(it_load + cooling_load + overhead, 2),
        trend_direction=trend,
        projected_pue_24h=projected,
        pue_history=history_list,
    )


@app.get(
    "/alerts",
    response_model=List[Alert],
    summary="Active alerts with confidence scores",
    description="Returns all currently active alerts including anomaly confidence scores and explanations.",
    tags=["Alerts"],
)
def get_alerts(
    severity: Optional[str] = Query(default=None, description="Filter by severity: critical/warning/info"),
    db: Session = Depends(get_db),
) -> List[Alert]:
    query = db.query(AlertDB).filter(AlertDB.is_active == True)
    if severity:
        query = query.filter(AlertDB.severity == severity.lower())
    rows = query.order_by(AlertDB.severity, AlertDB.confidence.desc()).all()

    return [
        Alert(
            id=r.id,
            timestamp=r.timestamp,
            last_seen=r.last_seen,
            asset_id=r.asset_id,
            metric=r.metric,
            severity=AlertSeverity(r.severity),
            message=r.message,
            confidence=r.confidence,
            explanation=r.explanation,
            duration_cycles=r.duration_cycles,
            deviation=r.deviation,
            baseline_value=r.baseline_value,
            current_value=r.current_value,
            is_active=r.is_active,
        )
        for r in rows
    ]


@app.get(
    "/recommendations",
    response_model=List[Recommendation],
    summary="Prioritised optimisation recommendations",
    description="Returns actionable cooling and power optimisation recommendations sorted by priority.",
    tags=["Recommendations"],
)
def get_recommendations(db: Session = Depends(get_db)) -> List[Recommendation]:
    rows = (
        db.query(RecommendationDB)
        .filter(RecommendationDB.is_active == True)
        .order_by(RecommendationDB.priority, RecommendationDB.confidence.desc())
        .all()
    )

    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_rows = sorted(rows, key=lambda r: priority_order.get(r.priority, 9))

    return [
        Recommendation(
            id=r.id,
            timestamp=r.timestamp,
            recommendation_type=r.recommendation_type,
            asset_id=r.asset_id,
            priority=Priority(r.priority),
            action_text=r.action_text,
            expected_savings_kw=r.expected_savings_kw,
            pue_impact=r.pue_impact,
            confidence=r.confidence,
            rationale=r.rationale,
        )
        for r in sorted_rows
    ]


@app.get(
    "/metrics/stranded-capacity",
    response_model=StrandedCapacityMetrics,
    summary="Stranded IT capacity",
    description="Calculates available IT capacity before hitting the binding cooling or power constraint.",
    tags=["Metrics"],
)
def metrics_stranded_capacity(db: Session = Depends(get_db)) -> StrandedCapacityMetrics:
    snap = get_last_snapshot()
    if not snap or not snap.racks:
        raise HTTPException(status_code=503, detail="No telemetry available.")

    it_load = sum(r.power_kw for r in snap.racks)
    cooling_load = sum(c.cooling_load_kw for c in snap.cracs)

    sc = calculate_stranded_capacity(it_load, cooling_load)

    return StrandedCapacityMetrics(
        timestamp=datetime.utcnow(),
        **sc,
    )


@app.get(
    "/metrics/carbon",
    response_model=CarbonMetrics,
    summary="Carbon emissions metrics",
    description="Returns live carbon emissions and annualised projection based on total facility power.",
    tags=["Metrics"],
)
def metrics_carbon(
    carbon_factor: Optional[float] = Query(
        default=None,
        description="Override grid carbon intensity (kg CO₂/kWh).  Defaults to env CARBON_INTENSITY_KG_PER_KWH.",
    ),
    db: Session = Depends(get_db),
) -> CarbonMetrics:
    snap = get_last_snapshot()
    if not snap or not snap.racks:
        raise HTTPException(status_code=503, detail="No telemetry available.")

    factor = carbon_factor if carbon_factor is not None else DEFAULT_CARBON_FACTOR
    it_load = sum(r.power_kw for r in snap.racks)
    cooling_load = sum(c.cooling_load_kw for c in snap.cracs)
    pue = calculate_pue(it_load, cooling_load)
    total_facility = it_load * pue

    c = calculate_carbon(total_facility, factor)

    # Fetch recent history for chart
    history_rows = (
        db.query(PUEHistoryDB)
        .order_by(PUEHistoryDB.timestamp.asc())
        .limit(60)
        .all()
    )
    carbon_history = [
        {
            "timestamp": h.timestamp.isoformat(),
            "carbon_kg_per_hr": round(h.total_facility_kw * factor, 2),
            "total_facility_kw": h.total_facility_kw,
        }
        for h in history_rows
    ]

    return CarbonMetrics(
        timestamp=datetime.utcnow(),
        carbon_kg_per_hr=c["carbon_kg_per_hr"],
        carbon_tonnes_per_year=c["carbon_tonnes_per_year"],
        total_facility_kw=c["total_facility_kw"],
        carbon_factor=factor,
        carbon_history=carbon_history,
    )


@app.post(
    "/simulate/what-if",
    response_model=WhatIfResponse,
    summary="What-if scenario simulation",
    description=(
        "Run a hypothetical scenario against the current facility state without modifying live data.  "
        "Returns projected PUE, stranded capacity, and risk assessment."
    ),
    tags=["Simulation"],
)
def simulate_what_if(request: WhatIfRequest, db: Session = Depends(get_db)) -> WhatIfResponse:
    snap = get_last_snapshot()
    if not snap or not snap.racks:
        raise HTTPException(status_code=503, detail="No telemetry available.")

    it_load = sum(r.power_kw for r in snap.racks)
    cooling_load = sum(c.cooling_load_kw for c in snap.cracs)
    current_pue = calculate_pue(it_load, cooling_load)
    current_stranded = calculate_stranded_capacity(it_load, cooling_load)["stranded_capacity_kw"]

    projection = project_scenario(
        current_it_kw=it_load,
        current_cooling_kw=cooling_load,
        scenario_type=request.scenario_type.value,
        params=request.parameters,
    )

    return WhatIfResponse(
        scenario_type=request.scenario_type,
        parameters=request.parameters,
        current_pue=current_pue,
        projected_pue=projection["projected_pue"],
        current_stranded_kw=current_stranded,
        projected_stranded_kw=projection["projected_stranded_kw"],
        new_alerts=projection["new_alerts"],
        risks=projection["risks"],
        summary=(
            f"Scenario '{request.scenario_type.value}': "
            f"PUE {current_pue:.3f} → {projection['projected_pue']:.3f} "
            f"({'▲' if projection['projected_pue'] > current_pue else '▼'} "
            f"{abs(projection['projected_pue'] - current_pue):.3f}).  "
            f"Stranded capacity {current_stranded:.0f} → {projection['projected_stranded_kw']:.0f} kW.  "
            f"{len(projection['new_alerts'])} new alert(s) would fire."
        ),
    )


@app.get(
    "/events/log",
    response_model=EventLogResponse,
    summary="Operations audit trail",
    description="Paginated, filterable audit log of all alert firings and recommendation generations.",
    tags=["Audit"],
)
def events_log(
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=50, ge=1, le=200, description="Events per page"),
    severity: Optional[str] = Query(default=None, description="Filter by severity"),
    asset_id: Optional[str] = Query(default=None, description="Filter by asset ID"),
    hours: int = Query(default=24, ge=1, le=168, description="Look-back window in hours"),
    db: Session = Depends(get_db),
) -> EventLogResponse:
    since = datetime.utcnow() - timedelta(hours=hours)
    query = db.query(EventDB).filter(EventDB.timestamp >= since)

    if severity:
        query = query.filter(EventDB.severity == severity.lower())
    if asset_id:
        query = query.filter(EventDB.asset_id == asset_id)

    total = query.count()
    rows = (
        query.order_by(EventDB.timestamp.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return EventLogResponse(
        total=total,
        page=page,
        page_size=page_size,
        events=[
            EventLogEntry(
                id=r.id,
                timestamp=r.timestamp,
                event_type=r.event_type,
                asset_id=r.asset_id,
                severity=r.severity,
                message=r.message,
                confidence=r.confidence,
            )
            for r in rows
        ],
    )
