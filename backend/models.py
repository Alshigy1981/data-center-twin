"""
Pydantic domain models for the Data Center Digital Twin Platform.
All models use Field() with descriptions for automatic OpenAPI documentation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────

class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TrendDirection(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"


class BindingConstraint(str, Enum):
    COOLING_LIMITED = "cooling-limited"
    POWER_LIMITED = "power-limited"
    BALANCED = "balanced"


class AshraeClass(str, Enum):
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    ABOVE_A2 = "Above A2"


class ScenarioType(str, Enum):
    ADD_GPU_RACK = "add_gpu_rack"
    COOLING_UNIT_FAILURE = "cooling_unit_failure"
    HIGHER_OUTSIDE_TEMP = "higher_outside_temp"
    UPS_OVERLOAD = "ups_overload"
    INCREASE_IT_LOAD = "increase_it_load"


# ─────────────────────────────────────────
# Asset Models
# ─────────────────────────────────────────

class RackAsset(BaseModel):
    """Physical rack asset definition."""
    id: str = Field(..., description="Unique rack identifier, e.g. A-R1-001")
    name: str = Field(..., description="Human-readable rack name")
    room: str = Field(..., description="Data center room, e.g. Room A")
    row: str = Field(..., description="Row label within room, e.g. Row 1")
    position: int = Field(..., description="Rack position (1–10) within row")
    rated_kw: float = Field(default=20.0, description="Rated maximum IT power in kW")


class CRACAsset(BaseModel):
    """CRAC/CRAH cooling unit asset definition."""
    id: str = Field(..., description="Unique CRAC identifier, e.g. A-CRAC-1")
    name: str = Field(..., description="Human-readable CRAC name")
    room: str = Field(..., description="Data center room served by this unit")
    rated_cooling_kw: float = Field(default=100.0, description="Rated cooling capacity in kW")
    setpoint_c: float = Field(default=18.0, description="Supply air temperature setpoint °C")


class ChillerAsset(BaseModel):
    """Central chiller plant asset definition."""
    id: str = Field(..., description="Unique chiller identifier, e.g. CH-1")
    name: str = Field(..., description="Human-readable chiller name")
    rated_cooling_kw: float = Field(default=500.0, description="Rated cooling capacity in kW")
    cop_design: float = Field(default=4.5, description="Design COP at full load")


class UPSAsset(BaseModel):
    """Uninterruptible Power Supply asset definition."""
    id: str = Field(..., description="Unique UPS identifier, e.g. UPS-1")
    name: str = Field(..., description="Human-readable UPS name")
    rated_kw: float = Field(default=200.0, description="Rated UPS output capacity in kW")


class PDUAsset(BaseModel):
    """Power Distribution Unit asset definition."""
    id: str = Field(..., description="Unique PDU identifier, e.g. A-PDU-1")
    name: str = Field(..., description="Human-readable PDU name")
    room: str = Field(..., description="Data center room containing this PDU")
    rated_kw: float = Field(default=100.0, description="Rated PDU capacity in kW")


class AssetInventory(BaseModel):
    """Complete asset inventory response."""
    racks: List[RackAsset] = Field(default_factory=list, description="All rack assets")
    cracs: List[CRACAsset] = Field(default_factory=list, description="All CRAC/CRAH units")
    chillers: List[ChillerAsset] = Field(default_factory=list, description="All chiller units")
    upses: List[UPSAsset] = Field(default_factory=list, description="All UPS units")
    pdus: List[PDUAsset] = Field(default_factory=list, description="All PDU units")
    total_rack_count: int = Field(default=0, description="Total number of racks")
    total_rated_it_kw: float = Field(default=0.0, description="Total rated IT power across all racks in kW")


# ─────────────────────────────────────────
# Telemetry Models
# ─────────────────────────────────────────

class RackTelemetry(BaseModel):
    """Real-time telemetry reading for a single rack."""
    asset_id: str = Field(..., description="Rack identifier")
    timestamp: datetime = Field(..., description="UTC timestamp of reading")
    inlet_temp_c: float = Field(..., description="Rack inlet air temperature °C")
    outlet_temp_c: float = Field(..., description="Rack outlet air temperature °C")
    power_kw: float = Field(..., description="Rack IT power draw in kW")
    humidity_pct: float = Field(..., description="Relative humidity percentage")
    temp_rise_c: float = Field(..., description="Outlet minus inlet temperature delta °C")
    ashrae_class: AshraeClass = Field(..., description="Current ASHRAE thermal class")
    room: str = Field(default="", description="Room containing this rack")
    row: str = Field(default="", description="Row containing this rack")
    position: int = Field(default=0, description="Position within the row")


class CRACTelemetry(BaseModel):
    """Real-time telemetry for a CRAC/CRAH unit."""
    asset_id: str = Field(..., description="CRAC unit identifier")
    timestamp: datetime = Field(..., description="UTC timestamp of reading")
    supply_air_temp_c: float = Field(..., description="Supply air temperature °C")
    return_air_temp_c: float = Field(..., description="Return air temperature °C")
    delta_t_c: float = Field(..., description="Return minus supply temperature delta °C")
    cooling_load_kw: float = Field(..., description="Estimated cooling load delivered in kW")
    setpoint_c: float = Field(..., description="Current supply air setpoint °C")
    room: str = Field(default="", description="Room served by this CRAC unit")


class ChillerTelemetry(BaseModel):
    """Real-time telemetry for a chiller unit."""
    asset_id: str = Field(..., description="Chiller identifier")
    timestamp: datetime = Field(..., description="UTC timestamp of reading")
    chw_supply_temp_c: float = Field(..., description="Chilled water supply temperature °C")
    chw_return_temp_c: float = Field(..., description="Chilled water return temperature °C")
    load_pct: float = Field(..., description="Chiller load as percentage of rated capacity")
    power_kw: float = Field(..., description="Chiller electrical power draw in kW")
    cop: float = Field(..., description="Current coefficient of performance")


class UPSTelemetry(BaseModel):
    """Real-time telemetry for a UPS unit."""
    asset_id: str = Field(..., description="UPS identifier")
    timestamp: datetime = Field(..., description="UTC timestamp of reading")
    load_pct: float = Field(..., description="UPS load as percentage of rated capacity")
    input_kw: float = Field(..., description="Input power draw in kW")
    output_kw: float = Field(..., description="Output power delivered in kW")
    battery_pct: float = Field(..., description="Battery charge percentage")


class PDUTelemetry(BaseModel):
    """Real-time telemetry for a PDU."""
    asset_id: str = Field(..., description="PDU identifier")
    timestamp: datetime = Field(..., description="UTC timestamp of reading")
    phase_a_kw: float = Field(..., description="Phase A load in kW")
    phase_b_kw: float = Field(..., description="Phase B load in kW")
    phase_c_kw: float = Field(..., description="Phase C load in kW")
    total_kw: float = Field(..., description="Total PDU load in kW")
    load_pct: float = Field(..., description="PDU load as percentage of rated capacity")
    room: str = Field(default="", description="Room containing this PDU")


class TelemetrySnapshot(BaseModel):
    """Complete telemetry snapshot across all assets."""
    timestamp: datetime = Field(..., description="Snapshot generation timestamp")
    cycle_id: int = Field(..., description="Sequential simulation cycle counter")
    racks: List[RackTelemetry] = Field(default_factory=list)
    cracs: List[CRACTelemetry] = Field(default_factory=list)
    chillers: List[ChillerTelemetry] = Field(default_factory=list)
    upses: List[UPSTelemetry] = Field(default_factory=list)
    pdus: List[PDUTelemetry] = Field(default_factory=list)
    outside_air_temp_c: float = Field(..., description="Simulated outside air temperature °C")


class TelemetryHistory(BaseModel):
    """Historical telemetry for a single asset."""
    asset_id: str = Field(..., description="Asset identifier")
    readings: List[Dict[str, Any]] = Field(default_factory=list, description="List of historical readings")


# ─────────────────────────────────────────
# Alert Models
# ─────────────────────────────────────────

class Alert(BaseModel):
    """Active alert with anomaly confidence scoring."""
    id: str = Field(..., description="Unique alert identifier")
    timestamp: datetime = Field(..., description="When alert first fired")
    last_seen: datetime = Field(..., description="When alert was last seen active")
    asset_id: str = Field(..., description="Asset that triggered the alert")
    metric: str = Field(..., description="Metric that breached threshold")
    severity: AlertSeverity = Field(..., description="Alert severity level")
    message: str = Field(..., description="Human-readable alert message")
    confidence: int = Field(..., ge=0, le=100, description="Confidence score 0–100")
    explanation: str = Field(..., description="Why this alert fired")
    duration_cycles: int = Field(default=1, description="Consecutive cycles the condition has been active")
    deviation: float = Field(..., description="Magnitude of deviation from expected baseline")
    baseline_value: float = Field(..., description="Expected normal value for this metric")
    current_value: float = Field(..., description="Current observed value")
    is_active: bool = Field(default=True, description="Whether the condition is still active")


# ─────────────────────────────────────────
# Recommendation Models
# ─────────────────────────────────────────

class Recommendation(BaseModel):
    """Optimization recommendation with priority and expected impact."""
    id: str = Field(..., description="Unique recommendation identifier")
    timestamp: datetime = Field(..., description="When this recommendation was generated")
    recommendation_type: str = Field(..., description="Category of recommendation")
    asset_id: Optional[str] = Field(None, description="Target asset (None for facility-wide)")
    priority: Priority = Field(..., description="Recommendation priority")
    action_text: str = Field(..., description="Specific action to take")
    expected_savings_kw: float = Field(default=0.0, description="Projected power savings in kW")
    pue_impact: float = Field(default=0.0, description="Estimated PUE improvement")
    confidence: int = Field(default=80, ge=0, le=100, description="Recommendation confidence 0–100")
    rationale: str = Field(..., description="Engineering rationale for this recommendation")


# ─────────────────────────────────────────
# Metrics Response Models
# ─────────────────────────────────────────

class PUEMetrics(BaseModel):
    """Power Usage Effectiveness metrics."""
    timestamp: datetime = Field(..., description="Metric calculation timestamp")
    live_pue: float = Field(..., description="Current live PUE value")
    rolling_avg_pue: float = Field(..., description="Rolling average PUE over last 60 readings")
    it_load_kw: float = Field(..., description="Total IT load in kW")
    cooling_load_kw: float = Field(..., description="Total cooling load in kW")
    overhead_kw: float = Field(..., description="Lighting and misc overhead in kW")
    total_facility_kw: float = Field(..., description="Total facility power draw in kW")
    trend_direction: TrendDirection = Field(..., description="PUE trend over recent history")
    projected_pue_24h: Optional[float] = Field(None, description="Linear regression forecast for next 24h")
    pue_history: List[Dict[str, Any]] = Field(default_factory=list, description="Recent PUE readings for charting")


class StrandedCapacityMetrics(BaseModel):
    """Available IT capacity before hitting a system constraint."""
    timestamp: datetime = Field(..., description="Calculation timestamp")
    cooling_headroom_kw: float = Field(..., description="Available cooling capacity in kW")
    power_headroom_kw: float = Field(..., description="Available power capacity in kW")
    stranded_capacity_kw: float = Field(..., description="Usable additional IT capacity in kW")
    binding_constraint: BindingConstraint = Field(..., description="Which system limits available capacity")
    total_crac_rated_kw: float = Field(..., description="Total CRAC rated cooling capacity in kW")
    total_pdu_rated_kw: float = Field(..., description="Total PDU rated capacity in kW")
    total_ups_rated_kw: float = Field(..., description="Total UPS rated capacity in kW")
    current_it_load_kw: float = Field(..., description="Current IT power draw in kW")
    current_cooling_load_kw: float = Field(..., description="Current cooling load in kW")


class CarbonMetrics(BaseModel):
    """Carbon emissions metrics."""
    timestamp: datetime = Field(..., description="Metric calculation timestamp")
    carbon_kg_per_hr: float = Field(..., description="Live carbon emissions in kg CO₂ per hour")
    carbon_tonnes_per_year: float = Field(..., description="Annualized carbon emissions in tonnes CO₂")
    total_facility_kw: float = Field(..., description="Total facility power draw in kW")
    carbon_factor: float = Field(..., description="Grid carbon intensity factor used (kg CO₂/kWh)")
    carbon_history: List[Dict[str, Any]] = Field(default_factory=list, description="Recent carbon readings for charting")


# ─────────────────────────────────────────
# What-If Simulation Models
# ─────────────────────────────────────────

class WhatIfRequest(BaseModel):
    """Request body for what-if scenario simulation."""
    scenario_type: ScenarioType = Field(..., description="Type of scenario to simulate")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Scenario parameters. Keys vary by scenario_type: "
            "add_gpu_rack: {n_racks, kw_per_rack}; "
            "cooling_unit_failure: {crac_id}; "
            "higher_outside_temp: {delta_c}; "
            "ups_overload: {increase_pct}; "
            "increase_it_load: {increase_pct}"
        ),
    )


class WhatIfResponse(BaseModel):
    """Response from a what-if scenario simulation."""
    scenario_type: ScenarioType = Field(..., description="Scenario that was simulated")
    parameters: Dict[str, Any] = Field(..., description="Parameters used in simulation")
    current_pue: float = Field(..., description="PUE before scenario")
    projected_pue: float = Field(..., description="Projected PUE after scenario")
    current_stranded_kw: float = Field(..., description="Stranded capacity before scenario in kW")
    projected_stranded_kw: float = Field(..., description="Projected stranded capacity after scenario in kW")
    new_alerts: List[str] = Field(default_factory=list, description="Alerts that would fire under this scenario")
    risks: List[str] = Field(default_factory=list, description="Risk statements for this scenario")
    summary: str = Field(..., description="Engineering summary of scenario impact")


# ─────────────────────────────────────────
# Event Log Models
# ─────────────────────────────────────────

class EventLogEntry(BaseModel):
    """Single entry in the operations audit trail."""
    id: int = Field(..., description="Sequential event ID")
    timestamp: datetime = Field(..., description="UTC event timestamp")
    event_type: str = Field(..., description="Event type: alert_fired, alert_cleared, recommendation_generated")
    asset_id: Optional[str] = Field(None, description="Asset associated with this event")
    severity: Optional[str] = Field(None, description="Severity level for alert events")
    message: str = Field(..., description="Event message text")
    confidence: Optional[int] = Field(None, description="Confidence score for alert events")


class EventLogResponse(BaseModel):
    """Paginated event log response."""
    total: int = Field(..., description="Total number of events matching filter")
    page: int = Field(..., description="Current page number (1-based)")
    page_size: int = Field(..., description="Number of events per page")
    events: List[EventLogEntry] = Field(default_factory=list, description="Events on this page")


# ─────────────────────────────────────────
# Multi-Objective Optimizer Models
# ─────────────────────────────────────────

class ParetoPoint(BaseModel):
    """Single point on the PUE vs thermal-risk Pareto front."""
    chw_setpoint_c: float = Field(..., description="Chilled water supply temperature °C")
    crac_setpoint_c: float = Field(..., description="CRAC supply air temperature °C")
    pue: float = Field(..., description="Power Usage Effectiveness at this setpoint pair")
    thermal_risk_score: float = Field(..., description="Thermal risk penalty (0 = safe, >0 = approaching ASHRAE limit)")
    carbon_kg_hr: float = Field(..., description="Carbon emissions kg CO₂/hr")
    cop: float = Field(..., description="Chiller COP at this CHW setpoint")
    crac_fan_power_kw: float = Field(..., description="CRAC fan power kW")
    chiller_power_kw: float = Field(..., description="Chiller electrical power kW")


class OptimizerResult(BaseModel):
    """Response from the multi-objective cooling setpoint optimizer."""
    timestamp: datetime = Field(..., description="Optimization timestamp")
    optimal_chw_setpoint_c: float = Field(..., description="Recommended chilled water supply setpoint °C")
    optimal_crac_setpoint_c: float = Field(..., description="Recommended CRAC supply air setpoint °C")
    expected_pue: float = Field(..., description="Expected PUE at optimal setpoints")
    expected_carbon_kg_hr: float = Field(..., description="Expected carbon emissions at optimal setpoints kg CO₂/hr")
    expected_cop: float = Field(..., description="Expected chiller COP at optimal setpoints")
    thermal_risk_score: float = Field(..., description="Thermal risk score at optimal setpoints")
    crac_fan_power_kw: float = Field(..., description="Expected CRAC fan power kW")
    chiller_power_kw: float = Field(..., description="Expected chiller power kW")
    current_pue: float = Field(..., description="Current operating PUE")
    pue_improvement: float = Field(..., description="PUE reduction achieved by switching to optimal setpoints")
    carbon_reduction_kg_hr: float = Field(..., description="Carbon reduction kg CO₂/hr vs current operating point")
    it_load_kw: float = Field(..., description="Current IT load used in optimization kW")
    avg_rack_outlet_c: float = Field(..., description="Average rack outlet temperature used in optimization °C")
    converged: bool = Field(..., description="Whether the optimizer converged")
    iterations: int = Field(..., description="Number of optimizer iterations")
    weights: Dict[str, float] = Field(..., description="Objective weights used: pue, carbon, thermal_risk")
    pareto_front: List[ParetoPoint] = Field(default_factory=list, description="Pareto-efficient setpoint pairs")


# ─────────────────────────────────────────
# RL Maintenance Agent Models
# ─────────────────────────────────────────

class AssetRiskEntry(BaseModel):
    """Per-asset health and risk metrics from the maintenance agent."""
    asset_id: str = Field(..., description="Asset identifier")
    asset_type: str = Field(..., description="Asset category: CRAC, Chiller, or UPS")
    health_score: float = Field(..., ge=0.0, le=1.0, description="Normalised health score (1.0 = perfect)")
    failure_probability_30d: float = Field(..., ge=0.0, le=1.0, description="Estimated failure probability over next 30 days")
    maintenance_recommended: bool = Field(..., description="True if the agent recommends scheduling maintenance")
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Accumulated anomaly intensity (0 = none)")


class MaintenanceScheduleResponse(BaseModel):
    """Response from the PPO maintenance scheduling agent."""
    timestamp: datetime = Field(..., description="Schedule generation timestamp")
    assets_scheduled: List[str] = Field(default_factory=list, description="Asset IDs recommended for maintenance")
    n_scheduled: int = Field(..., description="Number of assets scheduled for maintenance")
    risk_summary: List[AssetRiskEntry] = Field(default_factory=list, description="Full per-asset health snapshot")
    policy: str = Field(..., description="Policy used: ppo or rule_based")
    training_steps: int = Field(default=0, description="Total PPO training timesteps completed")


class TrainingRequest(BaseModel):
    """Request body to trigger PPO training."""
    total_timesteps: int = Field(
        default=50_000, ge=1_000, le=500_000,
        description="Timesteps to train (adds to any existing training)",
    )


class TrainingResponse(BaseModel):
    """Response after a PPO training run."""
    status: str = Field(..., description="Training outcome")
    total_timesteps: int = Field(..., description="Cumulative training timesteps")
    model_path: str = Field(..., description="Path where the model was saved")


# ─────────────────────────────────────────
# Health Check Model
# ─────────────────────────────────────────

class HealthResponse(BaseModel):
    """Service health check response."""
    status: str = Field(default="ok", description="Service health status")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Seconds since service started")
    db_status: str = Field(..., description="Database connectivity status")
    simulation_cycles: int = Field(default=0, description="Number of simulation cycles completed")
    last_simulation: Optional[datetime] = Field(None, description="Timestamp of most recent simulation cycle")
