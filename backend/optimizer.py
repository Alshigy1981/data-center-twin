"""
Multi-objective cooling setpoint optimizer for the Data Center Digital Twin.

Decision variables:
  - chw_setpoint_c   : chilled water supply temperature (6–14 °C)
  - crac_setpoint_c  : CRAC supply air temperature (16–24 °C)

Objectives (minimized):
  1. PUE — total facility power / IT load
  2. Carbon emissions — kg CO₂ / hr
  3. Thermal risk — penalty as rack inlet approaches ASHRAE A1 limit (32 °C)

The weighted-sum objective is minimized via scipy L-BFGS-B. A Pareto front is
approximated via uniform grid sampling over the two setpoint dimensions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

from backend.models import TelemetrySnapshot

# ─── Bounds ───────────────────────────────────────────────────────────────────

CHW_MIN, CHW_MAX = 6.0, 14.0     # chilled-water supply (°C)
CRAC_MIN, CRAC_MAX = 16.0, 24.0  # CRAC supply air (°C)

# ─── Physics constants ────────────────────────────────────────────────────────

_CONDENSER_TEMP_C = 30.0      # cooling-tower condenser approach
_CARNOT_EFF = 0.55            # fraction of Carnot COP achieved
_OVERHEAD_FRAC = 0.05         # lighting + misc as fraction of IT load
_CARBON_KG_PER_KWH = 0.233   # default grid carbon intensity
_ASHRAE_A1_LIMIT_C = 32.0     # inlet temperature hard limit
_THERMAL_MARGIN_SOFT_C = 6.0  # penalty onset: 6 °C below limit
_COIL_APPROACH_C = 5.0        # minimum delta between CRAC supply and CHW supply (HEX approach)


# ─── Physics helpers ──────────────────────────────────────────────────────────

def _effective_chw(chw_setpoint_c: float, crac_setpoint_c: float) -> float:
    """
    Return the actual CHW supply temp after applying the coil approach constraint.

    A chilled-water AHU/CRAC coil can only achieve a supply air temperature
    that is at least _COIL_APPROACH_C above the entering CHW temperature.
    So: T_supply_air ≥ T_chw + _COIL_APPROACH_C → T_chw ≤ T_supply_air − 5 °C.

    If the requested CHW setpoint is already low enough for the CRAC target,
    the CHW setpoint is the binding constraint. Otherwise the coil physics limit it.
    This coupling makes higher CRAC setpoints genuinely attractive: they unlock
    higher CHW temperatures (lower chiller lift, better COP).
    """
    max_chw_for_crac = crac_setpoint_c - _COIL_APPROACH_C
    return max(CHW_MIN, min(chw_setpoint_c, max_chw_for_crac))


def _chiller_cop(chw_setpoint_c: float) -> float:
    """Carnot-derived COP. Higher CHW setpoint → smaller lift → better COP."""
    supply_k = chw_setpoint_c + 273.15
    cond_k = _CONDENSER_TEMP_C + 273.15
    if cond_k <= supply_k:
        return 2.0
    return max(1.5, min(8.0, (supply_k / (cond_k - supply_k)) * _CARNOT_EFF))


def _crac_fan_power_kw(
    crac_setpoint_c: float,
    avg_outlet_c: float,
    n_cracs: int,
    crac_rated_kw: float = 100.0,
) -> float:
    """
    CRAC fan power via cube-law affinity.

    For a fixed cooling load Q = m_dot × cp × ΔT:
      - Higher CRAC setpoint → smaller ΔT (supply closer to return) → more airflow needed
      - Higher airflow → fan power scales as flow³ → more fan power

    Reference point: ΔT = 10 °C at 80 % of maximum fan speed.
    Fans are modelled as 15 % of each CRAC's rated electrical capacity.
    """
    delta_t = max(1.0, avg_outlet_c - crac_setpoint_c)
    ref_delta_t = 10.0
    # Higher setpoint → smaller delta_t → flow_ratio > 1 (but capped at 1.0 by min())
    # Note: flow_ratio < 1 means we can run fans slower (less power); > ref means faster
    flow_ratio = min(2.0, ref_delta_t / delta_t)
    fan_fraction = min(1.0, flow_ratio ** 3) * 0.8
    return max(0.5, fan_fraction * n_cracs * crac_rated_kw * 0.15)


def _thermal_risk(crac_setpoint_c: float, avg_outlet_c: float) -> float:
    """
    Returns a dimensionless penalty [0, ∞) rising sharply as the estimated
    rack inlet temperature approaches ASHRAE A1 (32 °C).

    Inlet estimate: supply + 30 % of (outlet − supply) mixing.
    """
    inlet = crac_setpoint_c + 0.3 * max(0.0, avg_outlet_c - crac_setpoint_c)
    margin = _ASHRAE_A1_LIMIT_C - inlet
    if margin >= _THERMAL_MARGIN_SOFT_C:
        return 0.0
    if margin <= 0.0:
        return 100.0
    return math.exp(3.0 * (_THERMAL_MARGIN_SOFT_C - margin) / _THERMAL_MARGIN_SOFT_C) - 1.0


# ─── Core evaluation ──────────────────────────────────────────────────────────

def evaluate_setpoints(
    chw_setpoint_c: float,
    crac_setpoint_c: float,
    it_kw: float,
    cooling_kw: float,
    avg_outlet_c: float,
    n_cracs: int,
    n_chillers: int,
    crac_rated_kw: float = 100.0,
) -> Dict[str, float]:
    """
    Compute energy and thermal metrics for a given pair of setpoints.
    All power values in kW; carbon in kg CO₂/hr.
    """
    eff_chw = _effective_chw(chw_setpoint_c, crac_setpoint_c)
    cop = _chiller_cop(eff_chw)
    chiller_power_kw = cooling_kw / max(cop, 0.1)
    fan_power_kw = _crac_fan_power_kw(crac_setpoint_c, avg_outlet_c, n_cracs, crac_rated_kw)
    overhead_kw = it_kw * _OVERHEAD_FRAC
    total_kw = it_kw + chiller_power_kw + fan_power_kw + overhead_kw
    pue = total_kw / max(it_kw, 1.0)
    carbon = total_kw * _CARBON_KG_PER_KWH
    risk = _thermal_risk(crac_setpoint_c, avg_outlet_c)

    return {
        "chw_setpoint_c": round(chw_setpoint_c, 3),
        "crac_setpoint_c": round(crac_setpoint_c, 3),
        "effective_chw_c": round(eff_chw, 3),
        "pue": round(pue, 4),
        "carbon_kg_hr": round(carbon, 3),
        "cop": round(cop, 3),
        "crac_fan_power_kw": round(fan_power_kw, 2),
        "chiller_power_kw": round(chiller_power_kw, 2),
        "thermal_risk_score": round(risk, 4),
    }


# ─── Single-point optimizer ───────────────────────────────────────────────────

def optimize_setpoints(
    snapshot: TelemetrySnapshot,
    w_pue: float = 0.5,
    w_carbon: float = 0.3,
    w_risk: float = 0.2,
) -> Dict[str, Any]:
    """
    Find (chw_setpoint, crac_setpoint) that minimises a weighted combination of
    PUE, carbon emissions, and thermal risk for the current facility state.

    Weights are re-normalised internally so they sum to 1.
    """
    it_kw = sum(r.power_kw for r in snapshot.racks)
    cooling_kw = sum(c.cooling_load_kw for c in snapshot.cracs)
    avg_outlet_c = (
        sum(r.outlet_temp_c for r in snapshot.racks) / max(len(snapshot.racks), 1)
    )
    n_cracs = len(snapshot.cracs)
    n_chillers = len(snapshot.chillers)

    total_w = w_pue + w_carbon + w_risk
    w_pue /= total_w
    w_carbon /= total_w
    w_risk /= total_w

    # Normalisation reference (midpoint of setpoint range)
    ref = evaluate_setpoints(10.0, 20.0, it_kw, cooling_kw, avg_outlet_c, n_cracs, n_chillers)
    ref_pue = max(ref["pue"], 1.01)
    ref_carbon = max(ref["carbon_kg_hr"], 0.01)

    def _obj(x: np.ndarray) -> float:
        m = evaluate_setpoints(float(x[0]), float(x[1]), it_kw, cooling_kw, avg_outlet_c, n_cracs, n_chillers)
        return (
            w_pue * m["pue"] / ref_pue
            + w_carbon * m["carbon_kg_hr"] / ref_carbon
            + w_risk * m["thermal_risk_score"]
        )

    result = minimize(
        _obj,
        x0=np.array([10.0, 19.0]),
        method="L-BFGS-B",
        bounds=[(CHW_MIN, CHW_MAX), (CRAC_MIN, CRAC_MAX)],
        options={"maxiter": 300, "ftol": 1e-10},
    )

    opt_chw = float(result.x[0])
    opt_crac = float(result.x[1])
    opt = evaluate_setpoints(opt_chw, opt_crac, it_kw, cooling_kw, avg_outlet_c, n_cracs, n_chillers)

    # Current operating point from live telemetry
    cur_chw = snapshot.chillers[0].chw_supply_temp_c if snapshot.chillers else 8.0
    cur_crac = snapshot.cracs[0].setpoint_c if snapshot.cracs else 18.0
    cur = evaluate_setpoints(cur_chw, cur_crac, it_kw, cooling_kw, avg_outlet_c, n_cracs, n_chillers)

    return {
        "optimal_chw_setpoint_c": round(opt_chw, 2),
        "optimal_crac_setpoint_c": round(opt_crac, 2),
        "expected_pue": opt["pue"],
        "expected_carbon_kg_hr": opt["carbon_kg_hr"],
        "expected_cop": opt["cop"],
        "thermal_risk_score": opt["thermal_risk_score"],
        "crac_fan_power_kw": opt["crac_fan_power_kw"],
        "chiller_power_kw": opt["chiller_power_kw"],
        "current_pue": cur["pue"],
        "pue_improvement": round(cur["pue"] - opt["pue"], 4),
        "carbon_reduction_kg_hr": round(cur["carbon_kg_hr"] - opt["carbon_kg_hr"], 3),
        "it_load_kw": round(it_kw, 2),
        "avg_rack_outlet_c": round(avg_outlet_c, 2),
        "weights": {"pue": round(w_pue, 3), "carbon": round(w_carbon, 3), "thermal_risk": round(w_risk, 3)},
        "converged": bool(result.success),
        "iterations": int(result.nit),
    }


# ─── Pareto front ─────────────────────────────────────────────────────────────

def compute_pareto_front(
    snapshot: TelemetrySnapshot,
    n_chw: int = 10,
    n_crac: int = 10,
) -> List[Dict[str, float]]:
    """
    Approximate the PUE vs thermal-risk Pareto front via uniform grid sampling.

    A point is Pareto-efficient if no other sampled point dominates it on
    *both* PUE and thermal_risk_score simultaneously.
    """
    it_kw = sum(r.power_kw for r in snapshot.racks)
    cooling_kw = sum(c.cooling_load_kw for c in snapshot.cracs)
    avg_outlet_c = (
        sum(r.outlet_temp_c for r in snapshot.racks) / max(len(snapshot.racks), 1)
    )
    n_cracs = len(snapshot.cracs)
    n_chillers = len(snapshot.chillers)

    grid: List[Dict[str, float]] = []
    for chw in np.linspace(CHW_MIN, CHW_MAX, n_chw):
        for crac in np.linspace(CRAC_MIN, CRAC_MAX, n_crac):
            m = evaluate_setpoints(float(chw), float(crac), it_kw, cooling_kw, avg_outlet_c, n_cracs, n_chillers)
            grid.append(m)

    pareto: List[Dict[str, float]] = []
    for i, pt in enumerate(grid):
        dominated = any(
            (other["pue"] <= pt["pue"] and other["thermal_risk_score"] <= pt["thermal_risk_score"])
            and (other["pue"] < pt["pue"] or other["thermal_risk_score"] < pt["thermal_risk_score"])
            for j, other in enumerate(grid) if j != i
        )
        if not dominated:
            pareto.append(pt)

    pareto.sort(key=lambda x: x["pue"])
    return pareto
