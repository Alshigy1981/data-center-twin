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
    AssetRiskEntry,
    BindingConstraint,
    DemandStatus,
    HourlyRate,
    MigrationRecommendation,
    MonthlyBillProjection,
    OptimalSchedule,
    OptimizationConflict,
    ParetoFrontData,
    PlacementRecommendation,
    PlacementWeightsRequest,
    PreCoolingAction,
    PreCoolingSchedule,
    RackScore,
    SavingsSummary,
    ScheduledLoad,
    ThermalPreview,
    UnifiedOptimizationPlan,
    CarbonMetrics,
    ChillerAsset,
    ChillerTelemetry,
    CRACAsset,
    CRACTelemetry,
    EventLogEntry,
    EventLogResponse,
    HealthResponse,
    MaintenanceScheduleResponse,
    OptimizerResult,
    ParetoPoint,
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
    TrainingRequest,
    TrainingResponse,
    TrendDirection,
    UPSAsset,
    UPSTelemetry,
    WhatIfRequest,
    WhatIfResponse,
)
from backend.optimizer import compute_pareto_front, optimize_setpoints
from backend.rl_agent import get_agent
from backend.simulator import build_asset_inventory, get_last_snapshot, start_simulation
from backend.workload_placement import placement_optimizer
from backend.energy_scheduler import scheduler as energy_scheduler, DEFAULT_LOADS
from backend.grid_pricing import tariff
from backend.optimization_coordinator import coordinator

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


# ─────────────────────────────────────────
# Multi-Objective Optimizer Endpoints
# ─────────────────────────────────────────

@app.get(
    "/optimize/cooling",
    response_model=OptimizerResult,
    summary="Multi-objective cooling setpoint optimizer",
    description=(
        "Finds optimal CHW and CRAC setpoints minimizing a weighted combination of "
        "PUE, carbon emissions, and thermal risk. Includes a Pareto front approximation "
        "over the setpoint search space for trade-off visualization."
    ),
    tags=["Optimization"],
)
def optimize_cooling(
    w_pue: float = Query(default=0.5, ge=0.0, le=1.0, description="Weight on PUE objective"),
    w_carbon: float = Query(default=0.3, ge=0.0, le=1.0, description="Weight on carbon objective"),
    w_risk: float = Query(default=0.2, ge=0.0, le=1.0, description="Weight on thermal risk objective"),
) -> OptimizerResult:
    snap = get_last_snapshot()
    if not snap or not snap.racks:
        raise HTTPException(status_code=503, detail="No telemetry available.")

    result = optimize_setpoints(snap, w_pue=w_pue, w_carbon=w_carbon, w_risk=w_risk)
    pareto = compute_pareto_front(snap)

    return OptimizerResult(
        timestamp=datetime.utcnow(),
        **result,
        pareto_front=[ParetoPoint(**p) for p in pareto],
    )


@app.get(
    "/optimize/pareto",
    response_model=List[ParetoPoint],
    summary="Cooling Pareto front",
    description=(
        "Returns Pareto-efficient (chw_setpoint, crac_setpoint) pairs that trade off "
        "PUE against thermal risk. Useful for dashboard scatter plots."
    ),
    tags=["Optimization"],
)
def pareto_front() -> List[ParetoPoint]:
    snap = get_last_snapshot()
    if not snap or not snap.racks:
        raise HTTPException(status_code=503, detail="No telemetry available.")
    return [ParetoPoint(**p) for p in compute_pareto_front(snap)]


# ─────────────────────────────────────────
# PPO Maintenance Agent Endpoints
# ─────────────────────────────────────────

@app.get(
    "/rl/maintenance-schedule",
    response_model=MaintenanceScheduleResponse,
    summary="PPO maintenance schedule",
    description=(
        "Returns a maintenance schedule produced by the trained PPO policy. "
        "Falls back to a rule-based schedule (health < 0.4 or anomaly > 0.5) "
        "if the model has not been trained yet."
    ),
    tags=["RL Agent"],
)
def maintenance_schedule() -> MaintenanceScheduleResponse:
    agent = get_agent()
    result = agent.get_schedule()
    return MaintenanceScheduleResponse(
        timestamp=datetime.utcnow(),
        assets_scheduled=result["assets_scheduled"],
        n_scheduled=result["n_scheduled"],
        risk_summary=[AssetRiskEntry(**e) for e in result["risk_summary"]],
        policy=result["policy"],
        training_steps=result["training_steps"],
    )


@app.post(
    "/rl/train",
    response_model=TrainingResponse,
    summary="Train PPO maintenance agent",
    description=(
        "Triggers a synchronous PPO training run for the specified number of timesteps. "
        "Subsequent calls add to the existing training (no reset). "
        "Model is persisted to assets/ppo_maintenance.zip after training."
    ),
    tags=["RL Agent"],
)
def train_agent(request: TrainingRequest) -> TrainingResponse:
    agent = get_agent()
    result = agent.train(total_timesteps=request.total_timesteps)
    return TrainingResponse(**result)


# ─────────────────────────────────────────
# Layer 3 — Workload Placement Endpoints
# ─────────────────────────────────────────

@app.get("/optimizer/placement/recommend", response_model=PlacementRecommendation, tags=["Optimization"])
def placement_recommend(workload_kw: float = Query(..., ge=0.5, le=20.0, description="Workload size in kW")):
    """Find optimal rack assignment for a new workload using MILP."""
    snap = get_last_snapshot()
    if not snap or not snap.racks:
        raise HTTPException(503, "No telemetry available.")
    result = placement_optimizer.recommend_placement(workload_kw, snap)
    return PlacementRecommendation(
        recommended_rack_id=result.get("recommended_rack_id"),
        workload_kw=workload_kw,
        thermal_risk_score=result.get("thermal_risk_score", 0.0),
        projected_inlet_temp=result.get("projected_inlet_temp", 0.0),
        projected_outlet_temp=result.get("projected_outlet_temp", 0.0),
        ashrae_class=result.get("ashrae_class", "A1"),
        pue_impact=result.get("pue_impact", 0.0),
        phase_impact=result.get("phase_impact", 0.0),
        overall_score=result.get("overall_score", 0.0),
        rationale=result.get("rationale", ""),
        feasible=result.get("recommended_rack_id") is not None,
        alternatives=[RackScore(**a) for a in result.get("alternatives", [])],
    )


@app.get("/optimizer/placement/scores", response_model=List[RackScore], tags=["Optimization"])
def placement_scores(workload_kw: float = Query(..., ge=0.5, le=20.0, description="Workload size in kW")):
    """Score all 80 racks for a given workload size."""
    snap = get_last_snapshot()
    if not snap or not snap.racks:
        raise HTTPException(503, "No telemetry available.")
    return [RackScore(**s) for s in placement_optimizer.score_all_racks(workload_kw, snap)]


@app.get("/optimizer/placement/migrations", response_model=List[MigrationRecommendation], tags=["Optimization"])
def placement_migrations():
    """Identify the top 5 workload migration opportunities from hot to cool racks."""
    snap = get_last_snapshot()
    if not snap or not snap.racks:
        raise HTTPException(503, "No telemetry available.")
    return [MigrationRecommendation(**m) for m in placement_optimizer.find_migration_candidates(snap)]


@app.get("/optimizer/placement/preview/{rack_id}", response_model=ThermalPreview, tags=["Optimization"])
def placement_preview(rack_id: str, workload_kw: float = Query(..., ge=0.5, le=20.0)):
    """Show before/after thermal projection for placing a workload on a specific rack."""
    snap = get_last_snapshot()
    if not snap or not snap.racks:
        raise HTTPException(503, "No telemetry available.")
    preview = placement_optimizer.get_thermal_preview(rack_id, workload_kw, snap)
    if not preview:
        raise HTTPException(404, f"Rack {rack_id} not found.")
    return ThermalPreview(**preview)


@app.post("/optimizer/placement/weights", tags=["Optimization"])
def placement_weights(req: PlacementWeightsRequest):
    """Update placement objective weights (must sum to 1.0)."""
    total = req.w1 + req.w2 + req.w3 + req.w4
    if abs(total - 1.0) > 0.01:
        raise HTTPException(422, f"Weights must sum to 1.0, got {total:.3f}")
    placement_optimizer.set_weights(req.w1, req.w2, req.w3, req.w4)
    return {"status": "ok", "weights": {"w1": req.w1, "w2": req.w2, "w3": req.w3, "w4": req.w4}}


# ─────────────────────────────────────────
# Layer 4 — Energy Cost Scheduler Endpoints
# ─────────────────────────────────────────

@app.get("/optimizer/energy/rate/current", tags=["Optimization"])
def energy_rate_current():
    """Return current energy rate, period type, and carbon intensity."""
    now = datetime.utcnow()
    return {
        "rate_per_kwh": tariff.get_current_rate(now),
        "period_type": tariff.get_period_type(now),
        "carbon_intensity": tariff.get_carbon_intensity(now),
        "is_grid_peak": tariff.is_grid_peak_event(now),
        "timestamp": now.isoformat(),
    }


@app.get("/optimizer/energy/rate/forecast", response_model=List[HourlyRate], tags=["Optimization"])
def energy_rate_forecast():
    """24-hour price and carbon intensity forecast."""
    return [HourlyRate(**h) for h in tariff.get_rate_forecast_24h(datetime.utcnow())]


@app.get("/optimizer/energy/schedule", response_model=OptimalSchedule, tags=["Optimization"])
def energy_schedule():
    """Optimal deferrable load schedule for the next 24 hours."""
    now = datetime.utcnow()
    snap = get_last_snapshot()
    peak_kw = sum(r.power_kw for r in snap.racks) if snap and snap.racks else 0.0
    result = energy_scheduler.solve_schedule(DEFAULT_LOADS, now, peak_kw)
    mapped_loads = [
        ScheduledLoad(
            load_name=l["load_name"],
            load_type=l["load_type"],
            start_time=l["start_time_iso"],
            end_time=l["end_time_iso"],
            power_kw=l["power_kw"],
            energy_cost=l["energy_cost"],
            cost_vs_immediate=l["cost_vs_immediate"],
            cost_saving=l["cost_saving"],
        )
        for l in result["loads"]
    ]
    return OptimalSchedule(
        loads=mapped_loads,
        total_cost=result["total_cost"],
        total_saving_vs_asap=result["total_saving_vs_asap"],
        peak_demand_avoided_kw=result["peak_demand_avoided_kw"],
        carbon_saved_kg=result["carbon_saved_kg"],
        schedule_horizon_hrs=result["schedule_horizon_hrs"],
    )


@app.get("/optimizer/energy/bill/current-month", response_model=MonthlyBillProjection, tags=["Optimization"])
def energy_bill_current_month():
    """Projected electricity bill for the current calendar month."""
    now = datetime.utcnow()
    snap = get_last_snapshot()
    it_kw = sum(r.power_kw for r in snap.racks) if snap and snap.racks else 200.0
    result = energy_scheduler.project_monthly_bill(it_kw, now.day, 30, it_kw * 1.1)
    return MonthlyBillProjection(**result)


@app.get("/optimizer/energy/demand/current", response_model=DemandStatus, tags=["Optimization"])
def energy_demand_current():
    """Current 15-minute rolling demand and ratchet risk status."""
    snap = get_last_snapshot()
    it_kw = sum(r.power_kw for r in snap.racks) if snap and snap.racks else 200.0
    facility_kw = it_kw * 1.4
    rolling = energy_scheduler.track_15min_demand([facility_kw])
    monthly_peak = facility_kw * 1.1
    threshold = 400.0
    return DemandStatus(
        rolling_15min_kw=round(rolling, 2),
        monthly_peak_kw=round(monthly_peak, 2),
        demand_charge_to_date=round(monthly_peak * 18.50, 2),
        ratchet_risk=monthly_peak > 350.0,
        projected_demand_charge=round(monthly_peak * 18.50, 2),
        threshold_kw=threshold,
        utilization_pct=round(rolling / threshold * 100, 1),
    )


@app.get("/optimizer/energy/precooling", response_model=PreCoolingSchedule, tags=["Optimization"])
def energy_precooling():
    """Pre-cooling strategy: lower CHW setpoint before on-peak windows to coast on thermal mass."""
    now = datetime.utcnow()
    snap = get_last_snapshot()
    chw = snap.chillers[0].chw_supply_temp_c if snap and snap.chillers else 8.0
    crac = snap.cracs[0].setpoint_c if snap and snap.cracs else 18.0
    result = energy_scheduler.optimize_precooling(tariff.get_rate_forecast_24h(now), chw, crac)
    return PreCoolingSchedule(
        actions=[PreCoolingAction(**a) for a in result["actions"]],
        total_cost_saving=result["total_cost_saving"],
        peak_load_reduction_kw=result["peak_load_reduction_kw"],
        start_time=result["start_time"],
        end_time=result["end_time"],
    )


# ─────────────────────────────────────────
# Optimization Coordinator Endpoints
# ─────────────────────────────────────────

@app.get("/optimizer/unified", response_model=UnifiedOptimizationPlan, tags=["Optimization"])
def optimizer_unified():
    """Unified recommendation from all four optimization layers with conflict resolution."""
    snap = get_last_snapshot()
    if not snap or not snap.racks:
        raise HTTPException(503, "No telemetry available.")
    it_kw = sum(r.power_kw for r in snap.racks)
    cooling_kw = sum(c.cooling_load_kw for c in snap.cracs)
    pue = calculate_pue(it_kw, cooling_kw)
    result = coordinator.get_unified_recommendations(snap, pue)
    return UnifiedOptimizationPlan(
        cooling_actions=result.get("cooling_actions", []),
        placement_actions=result.get("placement_actions", []),
        schedule_actions=result.get("schedule_actions", []),
        maintenance_actions=result.get("maintenance_actions", []),
        conflicts_detected=[OptimizationConflict(**c) for c in result.get("conflicts_detected", [])],
        estimated_savings=result.get("estimated_savings", {}),
        priority_order=result.get("priority_order", []),
        generated_at=datetime.utcnow(),
    )


@app.get("/optimizer/pareto", response_model=ParetoFrontData, tags=["Optimization"])
def optimizer_4obj_pareto():
    """4-objective Pareto front: PUE, energy cost, reliability, carbon."""
    snap = get_last_snapshot()
    if not snap or not snap.racks:
        raise HTTPException(503, "No telemetry available.")
    return ParetoFrontData(**coordinator.compute_4obj_pareto(snap))


@app.get("/optimizer/conflicts", response_model=List[OptimizationConflict], tags=["Optimization"])
def optimizer_conflicts():
    """Detect active conflicts between optimization layers."""
    snap = get_last_snapshot()
    now = datetime.utcnow()
    forecast = tariff.get_rate_forecast_24h(now)
    return [OptimizationConflict(**c) for c in coordinator.detect_conflicts(snap, forecast)]


@app.get("/optimizer/savings/summary", response_model=SavingsSummary, tags=["Optimization"])
def optimizer_savings_summary():
    """Estimated annual savings across all four optimization layers."""
    snap = get_last_snapshot()
    it_kw = sum(r.power_kw for r in snap.racks) if snap and snap.racks else 200.0
    cooling_kw = sum(c.cooling_load_kw for c in snap.cracs) if snap and snap.cracs else 100.0
    facility_kw = it_kw + cooling_kw + it_kw * 0.05
    return SavingsSummary(**coordinator.get_savings_summary(snap, it_kw, facility_kw))
