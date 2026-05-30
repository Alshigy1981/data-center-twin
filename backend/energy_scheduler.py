"""
Dynamic programming energy cost scheduler.
Depends on: backend/grid_pricing.py

Engineering assumptions:
- DP greedy heuristic prioritizes loads by priority (HIGH first)
- Pre-cooling uses 45-min thermal mass time constant for raised floor DC
- Demand charges calculated on rolling 15-minute interval per ANSI C12.20
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

from backend.grid_pricing import tariff, PEAK_DEMAND_CHARGE, RATCHET_FACTOR


# ─── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class DeferrableLoad:
    name: str
    load_type: str
    power_kw: float
    duration_hrs: float
    deadline_hrs: float
    min_gap_hrs: float
    can_interrupt: bool
    priority: str  # LOW/MEDIUM/HIGH


# ─── Default load catalog ─────────────────────────────────────────────────────

DEFAULT_LOADS: List[DeferrableLoad] = [
    DeferrableLoad("UPS Battery Test", "ups_test", 15.0, 2.0, 168.0, 120.0, False, "LOW"),
    DeferrableLoad("Generator Test", "generator_test", 45.0, 1.0, 720.0, 600.0, False, "LOW"),
    DeferrableLoad("Batch Compute", "batch_compute", 20.0, 4.0, 24.0, 0.0, True, "MEDIUM"),
    DeferrableLoad("Firmware Update", "firmware_update", 5.0, 1.0, 48.0, 0.0, False, "MEDIUM"),
    DeferrableLoad("CRAC Filter Maintenance", "crac_filter", -8.0, 0.5, 72.0, 0.0, False, "HIGH"),
]


class EnergyCostScheduler:
    """
    Greedy DP energy cost scheduler.

    Schedules deferrable loads to minimize electricity cost by placing them
    during off-peak hours while respecting deadlines and priority order.
    """

    def solve_schedule(
        self,
        pending_loads: List[DeferrableLoad],
        from_dt: datetime,
        current_demand_peak_kw: float = 0.0,
    ) -> dict:
        """
        Schedule pending loads to minimize cost.

        Processes loads in priority order: HIGH → MEDIUM → LOW.
        For each load, finds the cheapest available window within its deadline.

        Returns dict with: loads, total_cost, total_saving_vs_asap,
            peak_demand_avoided_kw, carbon_saved_kg, schedule_horizon_hrs
        """
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        sorted_loads = sorted(pending_loads, key=lambda l: priority_order.get(l.priority, 3))

        schedule_horizon_hrs = 24
        start = from_dt.replace(minute=0, second=0, microsecond=0)

        # Get rate forecast for scheduling window
        # Extend to cover deadlines up to 168 hrs (1 week)
        max_deadline = max((l.deadline_hrs for l in pending_loads), default=24.0)
        look_ahead = max(schedule_horizon_hrs, int(min(max_deadline, 168)))

        rate_forecast = []
        for i in range(look_ahead):
            dt = start + timedelta(hours=i)
            rate_forecast.append({
                "hour": i,
                "dt": dt,
                "rate": tariff.get_current_rate(dt),
                "carbon": tariff.get_carbon_intensity(dt),
                "period": tariff.get_period_type(dt),
            })

        scheduled_loads = []
        total_cost = 0.0
        total_saving = 0.0
        peak_demand_avoided_kw = 0.0
        carbon_saved_kg = 0.0

        # Track occupied hours (for non-interruptible loads)
        occupied: List[bool] = [False] * look_ahead

        for load in sorted_loads:
            duration_slots = math.ceil(load.duration_hrs)
            deadline_slot = min(int(load.deadline_hrs), look_ahead - duration_slots)

            if deadline_slot < 0:
                deadline_slot = look_ahead - duration_slots

            # Cost if scheduled immediately (hour 0)
            immediate_cost = 0.0
            for h in range(min(duration_slots, len(rate_forecast))):
                immediate_cost += load.power_kw * rate_forecast[h]["rate"]

            # Find cheapest window within deadline
            best_start_slot = 0
            best_cost = float("inf")

            for start_slot in range(0, deadline_slot + 1):
                end_slot = start_slot + duration_slots
                if end_slot > look_ahead:
                    break

                # Check if all slots are available
                if not load.can_interrupt:
                    if any(occupied[start_slot:end_slot]):
                        continue

                window_cost = 0.0
                for h in range(start_slot, end_slot):
                    window_cost += load.power_kw * rate_forecast[h]["rate"]

                if window_cost < best_cost:
                    best_cost = window_cost
                    best_start_slot = start_slot

            # Mark slots as occupied
            if not load.can_interrupt:
                for h in range(best_start_slot, min(best_start_slot + duration_slots, look_ahead)):
                    occupied[h] = True

            # Calculate carbon savings (shifting from on-peak to off-peak)
            immediate_carbon = sum(
                load.power_kw * rate_forecast[h]["carbon"]
                for h in range(min(duration_slots, len(rate_forecast)))
            )
            scheduled_carbon = sum(
                load.power_kw * rate_forecast[h]["carbon"]
                for h in range(best_start_slot, min(best_start_slot + duration_slots, look_ahead))
            )
            carbon_saving = max(0.0, immediate_carbon - scheduled_carbon)

            cost_saving = max(0.0, immediate_cost - best_cost)
            total_cost += best_cost
            total_saving += cost_saving
            carbon_saved_kg += carbon_saving

            # Demand peak avoided: if scheduled in off-peak vs on-peak
            if (rate_forecast[0]["period"] in ("on_peak", "shoulder")
                    and best_start_slot > 0
                    and rate_forecast[best_start_slot]["period"] == "off_peak"):
                peak_demand_avoided_kw += max(0.0, load.power_kw)

            start_dt_sched = start + timedelta(hours=best_start_slot)
            end_dt_sched = start_dt_sched + timedelta(hours=load.duration_hrs)

            scheduled_loads.append({
                "load_name": load.name,
                "load_type": load.load_type,
                "start_time_iso": start_dt_sched.isoformat(),
                "end_time_iso": end_dt_sched.isoformat(),
                "power_kw": load.power_kw,
                "energy_cost": round(best_cost, 4),
                "cost_vs_immediate": round(immediate_cost, 4),
                "cost_saving": round(cost_saving, 4),
            })

        return {
            "loads": scheduled_loads,
            "total_cost": round(total_cost, 4),
            "total_saving_vs_asap": round(total_saving, 4),
            "peak_demand_avoided_kw": round(peak_demand_avoided_kw, 2),
            "carbon_saved_kg": round(carbon_saved_kg, 4),
            "schedule_horizon_hrs": schedule_horizon_hrs,
        }

    def optimize_precooling(
        self,
        rate_forecast: List[dict],
        current_chw_setpoint: float,
        current_crac_setpoint: float,
    ) -> dict:
        """
        Generate pre-cooling strategy based on rate forecast.

        Strategy:
        - 2 hours before on-peak windows: lower CHW by 1.5°C (pre-cool)
        - During on-peak: raise CHW by 1.5°C (coast on thermal mass)
        - Otherwise: normal setpoints

        Returns dict with: actions, total_cost_saving, peak_load_reduction_kw,
            start_time_iso, end_time_iso
        """
        actions = []
        total_saving = 0.0
        peak_load_reduction_kw = 0.0

        # Build per-hour action map
        n = len(rate_forecast)
        hour_actions: List[str] = ["normal"] * n

        # Find on-peak hours
        on_peak_hours = set()
        for i, h in enumerate(rate_forecast):
            if h.get("period_type", h.get("period", "off_peak")) == "on_peak":
                on_peak_hours.add(i)

        # Mark pre-cool hours (2 hrs before on-peak)
        for i in range(n):
            if i in on_peak_hours:
                hour_actions[i] = "coast"
                # Mark 2 hours before as pre-cool
                for offset in [1, 2]:
                    pre_idx = i - offset
                    if 0 <= pre_idx < n and pre_idx not in on_peak_hours:
                        hour_actions[pre_idx] = "pre_cool"

        on_peak_rate = 0.28
        off_peak_rate = 0.06
        # Assume 30% of facility load is HVAC (~100 kW HVAC load for a typical DC)
        hvac_load_kw = 100.0
        precooling_extra_kw = 15.0   # extra kW during pre-cool
        coast_saving_kw = 20.0       # kW saved during coasting

        for i, h in enumerate(rate_forecast):
            action = hour_actions[i]
            rate = h.get("rate_per_kwh", h.get("rate", 0.10))
            dt_iso = h.get("dt_iso", h.get("dt", datetime.utcnow()).isoformat()
                          if not isinstance(h.get("dt"), str) else h.get("dt"))
            if callable(getattr(h.get("dt", None), "isoformat", None)):
                dt_iso = h["dt"].isoformat()

            if action == "pre_cool":
                chw = current_chw_setpoint - 1.5
                crac = current_crac_setpoint - 1.5
                # Small extra cost during pre-cooling
                extra_cost = precooling_extra_kw * rate
                # But saves on-peak cost later
                cost_saving = coast_saving_kw * on_peak_rate - extra_cost
                expected_saving = round(max(0.0, cost_saving), 4)
                total_saving += expected_saving
            elif action == "coast":
                chw = current_chw_setpoint + 1.5
                crac = current_crac_setpoint + 1.5
                expected_saving = round(coast_saving_kw * rate, 4)
                total_saving += expected_saving
                peak_load_reduction_kw += coast_saving_kw
            else:
                chw = current_chw_setpoint
                crac = current_crac_setpoint
                expected_saving = 0.0

            actions.append({
                "hour": i,
                "dt_iso": dt_iso if isinstance(dt_iso, str) else str(dt_iso),
                "chw_setpoint": round(chw, 2),
                "crac_setpoint": round(crac, 2),
                "action_type": action,
                "expected_cost_saving": expected_saving,
            })

        start_time = actions[0]["dt_iso"] if actions else datetime.utcnow().isoformat()
        end_time = actions[-1]["dt_iso"] if actions else datetime.utcnow().isoformat()

        return {
            "actions": actions,
            "total_cost_saving": round(total_saving, 4),
            "peak_load_reduction_kw": round(peak_load_reduction_kw, 2),
            "start_time": start_time,
            "end_time": end_time,
        }

    def track_15min_demand(self, power_readings: List[float]) -> float:
        """
        Return rolling 15-minute average of the last 4 readings (or all if < 4).
        """
        if not power_readings:
            return 0.0
        last_four = power_readings[-4:]
        return round(sum(last_four) / len(last_four), 2)

    def recommend_demand_shed(
        self, current_15min_kw: float, target_kw: float
    ) -> List[dict]:
        """
        Recommend load shedding actions to bring demand from current to target.

        Returns list of shed actions sorted by impact (largest first).
        """
        if current_15min_kw <= target_kw:
            return []

        reduction_needed = current_15min_kw - target_kw
        actions = []

        # Standard shed options (asset_id, action, kw_reduction, duration_mins, impact)
        shed_options = [
            ("CRAC-A1", "Raise CRAC setpoint +2°C", 12.0, 30, "MEDIUM"),
            ("CRAC-A2", "Raise CRAC setpoint +2°C", 12.0, 30, "MEDIUM"),
            ("BATCH-01", "Defer batch compute job", 35.0, 60, "LOW"),
            ("UPS-01", "Defer UPS battery test", 25.0, 240, "LOW"),
            ("LIGHTING-01", "Reduce non-essential lighting and auxiliary loads 80%", 16.0, 120, "LOW"),
        ]

        cumulative_reduction = 0.0
        for asset_id, action, kw_reduction, duration_mins, impact in shed_options:
            if cumulative_reduction >= reduction_needed:
                break
            actions.append({
                "asset_id": asset_id,
                "action": action,
                "load_reduction_kw": kw_reduction,
                "duration_mins": duration_mins,
                "impact_level": impact,
            })
            cumulative_reduction += kw_reduction

        return actions[:5]  # max 5 actions

    def project_monthly_bill(
        self,
        current_it_kw: float,
        days_elapsed: int,
        days_in_month: int,
        current_month_peak_kw: float,
        peak_12mo_kw: float = 0.0,
    ) -> dict:
        """
        Project monthly electricity bill based on current usage patterns.

        Returns dict with projected costs, savings estimate, ratchet risk.
        """
        # Project energy consumption for full month
        daily_kwh = current_it_kw * 1.4 * 24  # PUE 1.4, 24 hrs
        projected_kwh = daily_kwh * days_in_month

        # Use weighted average rate (~15% on-peak, 30% shoulder, 55% off-peak mix for DC)
        avg_rate = 0.06 * 0.55 + 0.13 * 0.30 + 0.28 * 0.15
        projected_energy_cost = round(projected_kwh * avg_rate, 2)

        # Demand charge
        ratchet_kw = RATCHET_FACTOR * peak_12mo_kw
        billed_demand = max(current_month_peak_kw, ratchet_kw)
        projected_demand_charge = round(billed_demand * PEAK_DEMAND_CHARGE, 2)

        ratchet_risk = ratchet_kw > current_month_peak_kw
        ratchet_amount = max(0.0, ratchet_kw - current_month_peak_kw) * PEAK_DEMAND_CHARGE

        # Scheduling savings (20% of energy from shifting to off-peak)
        on_peak_kwh_estimate = projected_kwh * 0.15
        savings_available = round(on_peak_kwh_estimate * (0.28 - 0.06) * 0.5, 2)

        projected_total = round(
            projected_energy_cost
            + projected_demand_charge
            + 850.0,  # facilities charge
            2,
        )

        days_remaining = days_in_month - days_elapsed

        return {
            "projected_energy_cost": projected_energy_cost,
            "projected_demand_charge": projected_demand_charge,
            "projected_total": projected_total,
            "savings_available": savings_available,
            "ratchet_risk": ratchet_risk,
            "ratchet_amount": round(ratchet_amount, 2),
            "days_remaining": days_remaining,
            "current_month_peak_kw": round(current_month_peak_kw, 2),
        }


# ─── Module-level singleton ───────────────────────────────────────────────────

scheduler = EnergyCostScheduler()
