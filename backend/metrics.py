"""
Engineering metric calculations for the Data Center Digital Twin Platform.

All functions accept plain numeric inputs so they can be tested in isolation
without any database or API dependencies.  The FastAPI layer calls these
functions after pulling values from the database.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from backend.models import (
    BindingConstraint,
    TrendDirection,
)

# ─────────────────────────────────────────
# Constants (engineering assumptions)
# ─────────────────────────────────────────

# Air properties at ~20°C, sea level
AIR_DENSITY_KG_M3 = 1.2
AIR_SPECIFIC_HEAT_KJ_KG_K = 1.006

# Lighting and misc overhead: 5 % of IT load (ASHRAE guideline)
OVERHEAD_FACTOR = 0.05

# Default CRAC, PDU, UPS rated capacities per unit (kW)
CRAC_RATED_KW = 100.0
PDU_RATED_KW = 100.0
UPS_RATED_KW = 200.0

# Number of each unit type
NUM_CRACS = 8
NUM_PDUS = 8
NUM_UPS = 2

# Total rated capacities
TOTAL_CRAC_RATED_KW = NUM_CRACS * CRAC_RATED_KW
TOTAL_PDU_RATED_KW = NUM_PDUS * PDU_RATED_KW
TOTAL_UPS_RATED_KW = NUM_UPS * UPS_RATED_KW

# Default grid carbon intensity (kg CO₂ per kWh) — UK national grid average
DEFAULT_CARBON_FACTOR = float(os.getenv("CARBON_INTENSITY_KG_PER_KWH", "0.233"))


# ─────────────────────────────────────────
# Core PUE Calculation
# ─────────────────────────────────────────

def calculate_pue(
    it_load_kw: float,
    cooling_load_kw: float,
    overhead_factor: float = OVERHEAD_FACTOR,
) -> float:
    """
    Calculate Power Usage Effectiveness (PUE).

    PUE = total facility power / IT load.
    The Green Grid defines PUE ≥ 1.0; a perfect data center has PUE = 1.0.
    Industry average is ~1.58; hyperscale targets < 1.2.

    Args:
        it_load_kw:     Sum of all rack IT power draws in kW.
        cooling_load_kw: Total cooling plant power draw in kW.
        overhead_factor: Fraction of IT load for lighting/misc (default 5 %).

    Returns:
        PUE value (float ≥ 1.0).  Returns 1.0 on zero IT load to avoid
        division-by-zero — callers should treat this as a guard value.
    """
    if it_load_kw <= 0:
        return 1.0
    overhead_kw = it_load_kw * overhead_factor
    total_facility_kw = it_load_kw + cooling_load_kw + overhead_kw
    pue = total_facility_kw / it_load_kw
    return round(max(1.0, pue), 4)


def calculate_total_facility_power(
    it_load_kw: float,
    cooling_load_kw: float,
    overhead_factor: float = OVERHEAD_FACTOR,
) -> float:
    """Return total facility power in kW (IT + cooling + overhead)."""
    return it_load_kw + cooling_load_kw + it_load_kw * overhead_factor


def calculate_cooling_overhead(it_load_kw: float, pue: float) -> float:
    """Derive cooling + overhead in kW from IT load and PUE."""
    return it_load_kw * (pue - 1)


# ─────────────────────────────────────────
# PUE Trend and Forecast
# ─────────────────────────────────────────

def calculate_pue_trend(pue_values: List[float]) -> TrendDirection:
    """
    Determine PUE trend from a recent history window.

    Uses linear regression slope: positive slope means PUE is rising
    (degrading efficiency).  A slope magnitude below 0.001 per sample
    is considered stable.
    """
    if len(pue_values) < 3:
        return TrendDirection.STABLE
    x = np.arange(len(pue_values), dtype=float)
    slope, _, _, _, _ = stats.linregress(x, pue_values)
    if slope > 0.001:
        return TrendDirection.DEGRADING
    if slope < -0.001:
        return TrendDirection.IMPROVING
    return TrendDirection.STABLE


def forecast_pue(
    pue_history: List[Dict],
    horizon_steps: int = 24,
) -> Optional[float]:
    """
    Linear-regression forecast of PUE over a specified number of future steps.

    Args:
        pue_history: List of dicts with 'pue' and 'timestamp' keys, newest last.
        horizon_steps: Number of steps (simulation cycles) to project forward.

    Returns:
        Projected PUE at horizon_steps ahead, or None if insufficient data.
    """
    if len(pue_history) < 5:
        return None
    values = [h["pue"] for h in pue_history]
    x = np.arange(len(values), dtype=float)
    slope, intercept, _, _, _ = stats.linregress(x, values)
    projected = slope * (len(values) + horizon_steps) + intercept
    return round(max(1.0, projected), 4)


def pue_forecast_series(
    pue_history: List[Dict],
    horizon_steps: int = 48,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Return (forecast values, upper CI, lower CI) for charting.

    CI is a ±1.96σ prediction interval from ordinary least squares.
    """
    if len(pue_history) < 5:
        return [], [], []

    values = np.array([h["pue"] for h in pue_history], dtype=float)
    x = np.arange(len(values), dtype=float)
    slope, intercept, _, _, std_err = stats.linregress(x, values)

    n = len(values)
    mean_x = float(np.mean(x))
    ss_x = float(np.sum((x - mean_x) ** 2))

    forecasts, uppers, lowers = [], [], []
    for i in range(1, horizon_steps + 1):
        future_x = float(n + i - 1)
        pred = slope * future_x + intercept
        # Prediction interval width
        se_pred = std_err * np.sqrt(1 + 1 / n + (future_x - mean_x) ** 2 / (ss_x + 1e-9))
        ci = 1.96 * se_pred
        forecasts.append(round(max(1.0, pred), 4))
        uppers.append(round(max(1.0, pred + ci), 4))
        lowers.append(round(max(1.0, pred - ci), 4))

    return forecasts, uppers, lowers


# ─────────────────────────────────────────
# Temperature Metrics
# ─────────────────────────────────────────

def calculate_rack_temp_rise(inlet_temp_c: float, outlet_temp_c: float) -> float:
    """Return rack temperature rise (delta-T) in °C."""
    return round(outlet_temp_c - inlet_temp_c, 2)


def calculate_crac_delta_t(
    return_air_temp_c: float, supply_air_temp_c: float
) -> float:
    """Return CRAC delta-T (return minus supply) in °C."""
    return round(return_air_temp_c - supply_air_temp_c, 2)


def chiller_cop_estimate(
    cooling_output_kw: float, electrical_input_kw: float
) -> float:
    """
    Estimate chiller COP = cooling output / electrical input.
    Returns 0 if electrical input is zero to avoid division error.
    """
    if electrical_input_kw <= 0:
        return 0.0
    return round(cooling_output_kw / electrical_input_kw, 2)


# ─────────────────────────────────────────
# Stranded Capacity
# ─────────────────────────────────────────

def calculate_stranded_capacity(
    current_it_load_kw: float,
    current_cooling_load_kw: float,
    total_crac_rated_kw: float = TOTAL_CRAC_RATED_KW,
    total_pdu_rated_kw: float = TOTAL_PDU_RATED_KW,
    total_ups_rated_kw: float = TOTAL_UPS_RATED_KW,
) -> Dict:
    """
    Calculate how much additional IT load could be deployed before hitting
    the first system constraint (cooling or power).

    The power headroom is the lesser of PDU and UPS headroom, since both
    must be satisfied simultaneously.

    Returns a dict with:
        cooling_headroom_kw, power_headroom_kw, stranded_capacity_kw,
        binding_constraint
    """
    cooling_headroom = max(0.0, total_crac_rated_kw - current_cooling_load_kw)
    pdu_headroom = max(0.0, total_pdu_rated_kw - current_it_load_kw)
    ups_headroom = max(0.0, total_ups_rated_kw - current_it_load_kw)
    power_headroom = min(pdu_headroom, ups_headroom)

    stranded = min(cooling_headroom, power_headroom)

    if cooling_headroom < power_headroom:
        constraint = BindingConstraint.COOLING_LIMITED
    elif power_headroom < cooling_headroom:
        constraint = BindingConstraint.POWER_LIMITED
    else:
        constraint = BindingConstraint.BALANCED

    return {
        "cooling_headroom_kw": round(cooling_headroom, 1),
        "power_headroom_kw": round(power_headroom, 1),
        "stranded_capacity_kw": round(stranded, 1),
        "binding_constraint": constraint,
        "total_crac_rated_kw": total_crac_rated_kw,
        "total_pdu_rated_kw": total_pdu_rated_kw,
        "total_ups_rated_kw": total_ups_rated_kw,
        "current_it_load_kw": round(current_it_load_kw, 1),
        "current_cooling_load_kw": round(current_cooling_load_kw, 1),
    }


# ─────────────────────────────────────────
# Carbon Intensity
# ─────────────────────────────────────────

def calculate_carbon(
    total_facility_kw: float,
    carbon_factor: float = DEFAULT_CARBON_FACTOR,
) -> Dict:
    """
    Calculate carbon emissions from total facility power draw.

    Args:
        total_facility_kw: Total facility power in kW.
        carbon_factor:     Grid carbon intensity in kg CO₂ per kWh.

    Returns:
        Dict with carbon_kg_per_hr, carbon_tonnes_per_year, carbon_factor.
    """
    carbon_kg_per_hr = total_facility_kw * carbon_factor
    carbon_tonnes_per_year = carbon_kg_per_hr * 8760 / 1000  # 8760 h/year

    return {
        "carbon_kg_per_hr": round(carbon_kg_per_hr, 2),
        "carbon_tonnes_per_year": round(carbon_tonnes_per_year, 2),
        "carbon_factor": carbon_factor,
        "total_facility_kw": round(total_facility_kw, 1),
    }


# ─────────────────────────────────────────
# What-If Scenario Projection
# ─────────────────────────────────────────

def project_scenario(
    current_it_kw: float,
    current_cooling_kw: float,
    scenario_type: str,
    params: Dict,
) -> Dict:
    """
    Project the impact of a what-if scenario without modifying live state.

    Returns projected PUE, stranded capacity, and a risk/alert list.
    """
    it_kw = current_it_kw
    cooling_kw = current_cooling_kw
    new_alerts: List[str] = []
    risks: List[str] = []

    if scenario_type == "add_gpu_rack":
        n_racks = int(params.get("n_racks", 1))
        kw_each = float(params.get("kw_per_rack", 10.0))
        added_kw = n_racks * kw_each
        it_kw += added_kw
        # Cooling load scales roughly proportionally
        cooling_kw += added_kw * (current_cooling_kw / max(current_it_kw, 1))
        risks.append(f"Adding {n_racks} GPU racks at {kw_each:.1f} kW each (+{added_kw:.1f} kW IT).")
        if kw_each > 15:
            risks.append("High-density racks (>15 kW) require hot-aisle containment review.")

    elif scenario_type == "cooling_unit_failure":
        # Lose one CRAC unit (100 kW capacity)
        effective_cooling_cap = TOTAL_CRAC_RATED_KW - CRAC_RATED_KW
        if current_cooling_kw > effective_cooling_cap:
            new_alerts.append("CRITICAL: Cooling capacity insufficient after CRAC failure.")
            risks.append(f"Cooling deficit of {current_cooling_kw - effective_cooling_cap:.1f} kW.")
        cooling_kw = min(cooling_kw, effective_cooling_cap)
        risks.append("Inlet temperatures may rise 3–8°C above baseline within 15 minutes.")

    elif scenario_type == "higher_outside_temp":
        delta = float(params.get("delta_c", 5.0))
        # Higher outside temp raises chiller entering water temperature
        # Approximate: 0.5 % more chiller power per °C outside temp rise
        cooling_kw *= 1 + (delta * 0.005)
        risks.append(f"Outside temp +{delta}°C increases chiller power draw by ~{delta * 0.5:.1f} %.")
        if delta > 10:
            new_alerts.append("WARNING: High ambient temperature may reduce chiller COP below design.")

    elif scenario_type == "ups_overload":
        increase_pct = float(params.get("increase_pct", 10.0))
        it_kw *= 1 + increase_pct / 100
        cooling_kw *= 1 + increase_pct / 100
        ups_load_pct = (it_kw / TOTAL_UPS_RATED_KW) * 100
        if ups_load_pct > 90:
            new_alerts.append(f"CRITICAL: UPS load would reach {ups_load_pct:.1f} % (>90 % threshold).")
        elif ups_load_pct > 80:
            new_alerts.append(f"WARNING: UPS load would reach {ups_load_pct:.1f} % (>80 % threshold).")
        risks.append(f"UPS load projected at {ups_load_pct:.1f} % of rated capacity.")

    elif scenario_type == "increase_it_load":
        increase_pct = float(params.get("increase_pct", 20.0))
        factor = 1 + increase_pct / 100
        it_kw *= factor
        cooling_kw *= factor
        risks.append(f"IT load increased by {increase_pct:.1f} %.")

    projected_pue = calculate_pue(it_kw, cooling_kw)
    stranded = calculate_stranded_capacity(it_kw, cooling_kw)

    if projected_pue > 2.0:
        new_alerts.append(f"WARNING: Projected PUE {projected_pue:.2f} indicates severe inefficiency.")
    if stranded["stranded_capacity_kw"] < 20:
        new_alerts.append(f"WARNING: Only {stranded['stranded_capacity_kw']:.0f} kW stranded capacity remaining.")

    return {
        "projected_pue": projected_pue,
        "projected_stranded_kw": stranded["stranded_capacity_kw"],
        "binding_constraint": stranded["binding_constraint"].value,
        "new_alerts": new_alerts,
        "risks": risks,
        "projected_it_kw": round(it_kw, 1),
        "projected_cooling_kw": round(cooling_kw, 1),
    }
