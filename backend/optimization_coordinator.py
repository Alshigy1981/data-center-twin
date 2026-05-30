"""
Unifies all four optimization layers. Conflict detection and resolution.

Depends on:
  backend/optimizer.py
  backend/rl_agent.py
  backend/workload_placement.py
  backend/energy_scheduler.py
  backend/grid_pricing.py

Engineering assumptions:
- Thermal safety always wins over cost optimization
- Layer 4 (energy scheduling) wins over Layer 2 (cooling) on pre-cooling conflicts
- PPO maintenance scheduling is advisory; demand response takes precedence
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from backend.optimizer import evaluate_setpoints, optimize_setpoints
from backend.rl_agent import get_agent
from backend.workload_placement import placement_optimizer
from backend.energy_scheduler import scheduler as energy_scheduler, DEFAULT_LOADS
from backend.grid_pricing import tariff

logger = logging.getLogger(__name__)


class OptimizationCoordinator:
    """
    Unification layer that integrates all four optimization modules
    and detects/resolves conflicts between their recommendations.
    """

    def detect_conflicts(
        self,
        snapshot,
        rate_forecast: List[dict],
        cooling_result: Optional[dict] = None,
        maintenance_schedule: Optional[dict] = None,
    ) -> List[dict]:
        """
        Detect conflicts between optimization layers.

        Checks 4 conflict patterns:
          1. COOLING_VS_ENERGY: pre-cooling active but cooling optimizer raises setpoints
          2. PLACEMENT_VS_SCHEDULING: (returned empty unless explicit workload queued)
          3. MAINTENANCE_VS_DEMAND: maintenance scheduled during on-peak/shoulder
          4. THERMAL_VS_COST: hot racks + on-peak rate (cost wants to raise setpoints)
        """
        conflicts = []
        if not snapshot or not rate_forecast:
            return conflicts

        now = datetime.utcnow()
        current_period = tariff.get_period_type(now)
        current_rate = tariff.get_current_rate(now)

        # Check if on-peak in next 2 hours
        on_peak_soon = any(
            h.get("period_type", h.get("period", "off_peak")) == "on_peak"
            for h in rate_forecast[:2]
        )

        # 1. COOLING_VS_ENERGY
        if cooling_result is not None:
            opt_chw = cooling_result.get("optimal_chw_setpoint_c", 10.0)
            # Pre-cooling: we want lower CHW. If cooling optimizer recommends high CHW
            # during pre-cooling window, that's a conflict.
            if on_peak_soon and opt_chw > 11.0:
                conflicts.append({
                    "conflict_type": "COOLING_VS_ENERGY",
                    "layer_a": "Layer 2 (Cooling Optimizer)",
                    "layer_b": "Layer 4 (Energy Scheduler)",
                    "description": (
                        f"Cooling optimizer recommends CHW={opt_chw:.1f}°C "
                        f"but pre-cooling requires lower CHW setpoint before on-peak window."
                    ),
                    "resolution": "Apply pre-cooling protocol: lower CHW by 1.5°C for 2h before on-peak.",
                    "winner": "Layer 4 (Energy Scheduler)",
                })

        # 2. PLACEMENT_VS_SCHEDULING — skip unless explicit workload queued
        # (per spec: return empty for this pattern)

        # 3. MAINTENANCE_VS_DEMAND
        if maintenance_schedule is not None:
            scheduled_assets = maintenance_schedule.get("assets_scheduled", [])
            if scheduled_assets and current_period in ("on_peak", "shoulder"):
                conflicts.append({
                    "conflict_type": "MAINTENANCE_VS_DEMAND",
                    "layer_a": "Layer 3 (Maintenance Agent)",
                    "layer_b": "Layer 4 (Energy Scheduler)",
                    "description": (
                        f"Maintenance scheduled for {len(scheduled_assets)} asset(s) "
                        f"during {current_period} period (${current_rate:.2f}/kWh). "
                        f"Affected: {', '.join(scheduled_assets[:3])}."
                    ),
                    "resolution": "Defer maintenance to next off-peak window to avoid demand charges.",
                    "winner": "Layer 4 (Energy Scheduler)",
                })

        # 4. THERMAL_VS_COST
        if snapshot and snapshot.racks:
            hot_racks = [r for r in snapshot.racks if r.inlet_temp_c > 26.0]
            if hot_racks and current_period == "on_peak":
                conflicts.append({
                    "conflict_type": "THERMAL_VS_COST",
                    "layer_a": "Layer 4 (Energy Scheduler)",
                    "layer_b": "Layer 2 (Cooling Optimizer)",
                    "description": (
                        f"{len(hot_racks)} rack(s) have inlet temp > 26°C "
                        f"during on-peak period. Cost optimization would raise setpoints "
                        f"but thermal safety requires maintaining cooling."
                    ),
                    "resolution": "Maintain current cooling setpoints; thermal safety overrides cost optimization.",
                    "winner": "Thermal Safety (Layer 2)",
                })

        return conflicts

    def get_unified_recommendations(self, snapshot, pue: float) -> dict:
        """
        Aggregate recommendations from all 4 optimization layers.

        Returns dict with all recommendation types, detected conflicts,
        estimated savings, and execution priority order.
        """
        now = datetime.utcnow()
        rate_forecast = tariff.get_rate_forecast_24h(now)

        # Layer 2: Cooling optimizer
        cooling_actions = []
        cooling_result = None
        try:
            cooling_result = optimize_setpoints(snapshot)
            cooling_actions = [
                {
                    "action": "adjust_cooling_setpoints",
                    "optimal_chw_setpoint_c": cooling_result.get("optimal_chw_setpoint_c"),
                    "optimal_crac_setpoint_c": cooling_result.get("optimal_crac_setpoint_c"),
                    "expected_pue": cooling_result.get("expected_pue"),
                    "pue_improvement": cooling_result.get("pue_improvement"),
                    "carbon_reduction_kg_hr": cooling_result.get("carbon_reduction_kg_hr"),
                }
            ]
        except Exception as exc:
            logger.warning("Cooling optimizer failed: %s", exc)

        # Layer 3 (PPO maintenance agent)
        maintenance_actions = []
        maintenance_schedule = None
        try:
            agent = get_agent()
            maintenance_schedule = agent.get_schedule()
            for asset_id in maintenance_schedule.get("assets_scheduled", []):
                risk_info = next(
                    (r for r in maintenance_schedule.get("risk_summary", [])
                     if r["asset_id"] == asset_id),
                    {},
                )
                maintenance_actions.append({
                    "action": "schedule_maintenance",
                    "asset_id": asset_id,
                    "health_score": risk_info.get("health_score", 0.0),
                    "failure_probability_30d": risk_info.get("failure_probability_30d", 0.0),
                    "policy": maintenance_schedule.get("policy", "rule_based"),
                })
        except Exception as exc:
            logger.warning("Maintenance agent failed: %s", exc)

        # Layer 4: Energy scheduler
        schedule_actions = []
        try:
            it_kw = sum(r.power_kw for r in snapshot.racks) if snapshot and snapshot.racks else 200.0
            sched_result = energy_scheduler.solve_schedule(DEFAULT_LOADS, now, it_kw)
            schedule_actions = [
                {
                    "action": "schedule_load",
                    "load_name": l["load_name"],
                    "start_time": l["start_time_iso"],
                    "power_kw": l["power_kw"],
                    "cost_saving": l["cost_saving"],
                }
                for l in sched_result.get("loads", [])
            ]
        except Exception as exc:
            logger.warning("Energy scheduler failed: %s", exc)

        # Conflict detection
        conflicts = self.detect_conflicts(
            snapshot, rate_forecast, cooling_result, maintenance_schedule
        )

        # Estimated savings
        it_kw = sum(r.power_kw for r in snapshot.racks) if snapshot and snapshot.racks else 200.0
        facility_kw = it_kw * 1.4
        current_rate = tariff.get_current_rate(now)
        carbon_factor = tariff.get_carbon_intensity(now)

        pue_improvement = cooling_result.get("pue_improvement", 0.0) if cooling_result else 0.0
        energy_cost_saving = facility_kw * pue_improvement * current_rate * 24
        demand_saving = it_kw * 0.05 * 18.50
        carbon_saving = facility_kw * pue_improvement * carbon_factor * 24

        estimated_savings = {
            "energy_cost_per_day": round(energy_cost_saving, 2),
            "demand_charge_per_month": round(demand_saving, 2),
            "pue_improvement": round(pue_improvement, 4),
            "carbon_per_day": round(carbon_saving, 2),
            "total_annual": round((energy_cost_saving * 365) + (demand_saving * 12), 2),
        }

        # Priority order (thermal safety first)
        priority_order = [
            "Thermal Safety (Cooling Optimizer)",
            "Demand Response (Energy Scheduler)",
            "Workload Placement",
            "Maintenance Scheduling",
        ]

        return {
            "cooling_actions": cooling_actions,
            "placement_actions": [],
            "schedule_actions": schedule_actions,
            "maintenance_actions": maintenance_actions,
            "conflicts_detected": conflicts,
            "estimated_savings": estimated_savings,
            "priority_order": priority_order,
        }

    def compute_4obj_pareto(self, snapshot) -> dict:
        """
        Sample 200 operating points by varying weights and compute 4-objective Pareto front.

        Objectives (all minimize):
          - pue
          - energy_cost_per_hr
          - reliability_score (= 1.0 - thermal_risk)
          - carbon_kg_hr
        """
        if not snapshot or not snapshot.racks:
            return {
                "points": [],
                "pareto_points": [],
                "current_point": {},
                "recommended_point": {},
                "objectives": ["pue", "energy_cost_per_hr", "reliability_score", "carbon_kg_hr"],
            }

        it_kw = sum(r.power_kw for r in snapshot.racks)
        cooling_kw = sum(c.cooling_load_kw for c in snapshot.cracs) if snapshot.cracs else it_kw * 0.4
        avg_outlet_c = (
            sum(r.outlet_temp_c for r in snapshot.racks) / max(len(snapshot.racks), 1)
        )
        n_cracs = len(snapshot.cracs) if snapshot.cracs else 4
        n_chillers = len(snapshot.chillers) if snapshot.chillers else 2

        current_rate = tariff.get_current_rate(datetime.utcnow())

        # Sample 200 points with varied weights
        rng = np.random.default_rng(42)
        points = []

        for _ in range(200):
            # Random weights for [w_pue, w_carbon, w_risk]
            w = rng.dirichlet([1.0, 1.0, 1.0])
            w_pue, w_carbon, w_risk = float(w[0]), float(w[1]), float(w[2])

            try:
                result = optimize_setpoints(
                    snapshot,
                    w_pue=w_pue,
                    w_carbon=w_carbon,
                    w_risk=w_risk,
                )
                # Get metric values at the optimal point
                m = evaluate_setpoints(
                    result["optimal_chw_setpoint_c"],
                    result["optimal_crac_setpoint_c"],
                    it_kw,
                    cooling_kw,
                    avg_outlet_c,
                    n_cracs,
                    n_chillers,
                )
                pue_val = m["pue"]
                carbon = m["carbon_kg_hr"]
                thermal_risk = m["thermal_risk_score"]
                reliability = max(0.0, 1.0 - thermal_risk / 10.0)  # normalize
                energy_cost_hr = it_kw * pue_val * current_rate

                points.append({
                    "pue": round(pue_val, 4),
                    "energy_cost_per_hr": round(energy_cost_hr, 4),
                    "reliability_score": round(reliability, 4),
                    "carbon_kg_hr": round(carbon, 4),
                    "w_pue": round(w_pue, 3),
                    "w_carbon": round(w_carbon, 3),
                    "w_risk": round(w_risk, 3),
                    "chw_setpoint": result["optimal_chw_setpoint_c"],
                    "crac_setpoint": result["optimal_crac_setpoint_c"],
                })
            except Exception:
                continue

        # Pareto filter on (pue, energy_cost_per_hr) — minimize both
        pareto_points = []
        for i, pt in enumerate(points):
            dominated = any(
                (
                    other["pue"] <= pt["pue"]
                    and other["energy_cost_per_hr"] <= pt["energy_cost_per_hr"]
                    and (
                        other["pue"] < pt["pue"]
                        or other["energy_cost_per_hr"] < pt["energy_cost_per_hr"]
                    )
                )
                for j, other in enumerate(points)
                if j != i
            )
            if not dominated:
                pareto_points.append(pt)

        pareto_points.sort(key=lambda x: x["pue"])

        # Current operating point
        cur_chw = snapshot.chillers[0].chw_supply_temp_c if snapshot.chillers else 8.0
        cur_crac = snapshot.cracs[0].setpoint_c if snapshot.cracs else 18.0
        cur_m = evaluate_setpoints(
            cur_chw, cur_crac, it_kw, cooling_kw, avg_outlet_c, n_cracs, n_chillers
        )
        current_point = {
            "pue": cur_m["pue"],
            "energy_cost_per_hr": round(it_kw * cur_m["pue"] * current_rate, 4),
            "reliability_score": max(0.0, 1.0 - cur_m["thermal_risk_score"] / 10.0),
            "carbon_kg_hr": cur_m["carbon_kg_hr"],
            "chw_setpoint": cur_chw,
            "crac_setpoint": cur_crac,
        }

        # Recommended: Pareto point with best balance (closest to utopia)
        recommended_point = {}
        if pareto_points:
            # Utopia point: minimum of each objective across Pareto front
            min_pue = min(p["pue"] for p in pareto_points)
            min_cost = min(p["energy_cost_per_hr"] for p in pareto_points)
            max_rel = max(p["reliability_score"] for p in pareto_points)
            min_carbon = min(p["carbon_kg_hr"] for p in pareto_points)

            def _distance(pt):
                return (
                    (pt["pue"] - min_pue) ** 2
                    + (pt["energy_cost_per_hr"] - min_cost) ** 2
                    + (max_rel - pt["reliability_score"]) ** 2
                    + (pt["carbon_kg_hr"] - min_carbon) ** 2
                )

            recommended_point = min(pareto_points, key=_distance)

        return {
            "points": points,
            "pareto_points": pareto_points,
            "current_point": current_point,
            "recommended_point": recommended_point,
            "objectives": ["pue", "energy_cost_per_hr", "reliability_score", "carbon_kg_hr"],
        }

    def get_savings_summary(self, snapshot, it_kw: float, facility_kw: float) -> dict:
        """
        Estimate savings across all optimization layers.

        Returns dict with annualized savings estimates.
        """
        now = datetime.utcnow()
        current_rate = tariff.get_current_rate(now)
        off_peak_rate = 0.06
        carbon_factor = tariff.get_carbon_intensity(now)

        # Layer 4: Scheduling 20% of loads to off-peak
        shiftable_fraction = 0.20
        on_peak_fraction = 0.15  # fraction of day at on-peak rates
        shiftable_kwh = facility_kw * shiftable_fraction * 24
        rate_diff = 0.28 - off_peak_rate  # on-peak vs off-peak
        energy_cost_saving_per_day = round(shiftable_kwh * on_peak_fraction * rate_diff, 2)

        # Demand charge savings from peak avoidance
        avoided_peak_kw = facility_kw * 0.05  # 5% demand reduction via scheduling
        demand_charge_saving_per_month = round(avoided_peak_kw * 18.50, 2)

        # PUE improvement from cooling optimizer
        try:
            if snapshot and snapshot.racks:
                cooling_result = optimize_setpoints(snapshot)
                pue_improvement = cooling_result.get("pue_improvement", 0.02)
            else:
                pue_improvement = 0.02
        except Exception:
            pue_improvement = 0.02

        # Carbon saving from PUE improvement
        carbon_saving_kg_per_day = round(
            facility_kw * pue_improvement * carbon_factor * 24, 2
        )

        # Total annual
        annual_energy = energy_cost_saving_per_day * 365
        annual_demand = demand_charge_saving_per_month * 12
        total_annual = round(annual_energy + annual_demand, 2)

        return {
            "energy_cost_saving_per_day": energy_cost_saving_per_day,
            "demand_charge_saving_per_month": demand_charge_saving_per_month,
            "pue_improvement": round(pue_improvement, 4),
            "carbon_saving_kg_per_day": carbon_saving_kg_per_day,
            "total_annual_estimate": total_annual,
        }


# ─── Module-level singleton ───────────────────────────────────────────────────

coordinator = OptimizationCoordinator()
