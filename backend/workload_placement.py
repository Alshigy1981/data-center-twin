"""
Rack assignment MILP using PuLP.

Engineering assumptions:
- Thermal resistance factor 0.15 degC/kW assumes hot-aisle containment
- Without containment, factor is typically 0.25-0.40 degC/kW
- Cooling overhead factor 1.4 assumes PUE ~1.4
- Migration downtime estimate assumes cold migration (no live migration)
- Phase assignment rack_index % 3 assumes standard 3-phase PDU wiring
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

_THERMAL_RESISTANCE = 0.15   # degC/kW (hot-aisle containment assumed)
_COOLING_OVERHEAD = 1.4      # PUE ~ 1.4
_RACK_RATED_KW = 20.0        # kW per rack
_PDU_RATED_KW = 100.0        # kW per PDU
_UPS_RATED_KW = 200.0        # kW per UPS
_ASHRAE_A2_INLET_LIMIT = 35.0  # degC (ASHRAE A2 hard limit)
_RACK_POWER_LIMIT = 20.0     # kW
_PDU_LOAD_LIMIT_PCT = 0.90
_UPS_LOAD_LIMIT_PCT = 0.90
_MILP_TIMEOUT_SEC = 5

# ASHRAE inlet thresholds
_ASHRAE_CLASSES = [
    (0.0, 21.0, "A1"),
    (21.0, 24.0, "A1"),
    (24.0, 27.0, "A2"),
    (27.0, 32.0, "A2"),
    (32.0, 35.0, "A2"),
    (35.0, 40.0, "A3"),
    (40.0, 999.0, "Above A2"),
]


def _ashrae_class(inlet_c: float) -> str:
    if inlet_c <= 21.0:
        return "A1"
    elif inlet_c <= 24.0:
        return "A1"
    elif inlet_c <= 27.0:
        return "A2"
    elif inlet_c <= 32.0:
        return "A2"
    elif inlet_c <= 35.0:
        return "A2"
    elif inlet_c <= 40.0:
        return "A3"
    return "Above A2"


class WorkloadPlacementOptimizer:
    """
    Rack assignment optimizer using MILP (PuLP/CBC).

    Scores each rack on four criteria:
      w1: thermal risk
      w2: PUE impact
      w3: stranded capacity loss
      w4: phase imbalance
    """

    def __init__(
        self,
        w1: float = 0.4,
        w2: float = 0.3,
        w3: float = 0.2,
        w4: float = 0.1,
    ) -> None:
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

    def set_weights(self, w1: float, w2: float, w3: float, w4: float) -> None:
        total = w1 + w2 + w3 + w4
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

    # ── Internal score computations ──────────────────────────────────────────

    def _thermal_risk_score(self, rack_telemetry, workload_kw: float) -> float:
        """Compute thermal risk score [0, inf)."""
        projected_inlet = rack_telemetry.inlet_temp_c + workload_kw * _THERMAL_RESISTANCE
        if projected_inlet <= 24.0:
            return 0.0
        elif projected_inlet <= 32.0:
            return (projected_inlet - 24.0) / 8.0
        else:
            return 1.0 + (projected_inlet - 32.0) * 0.5

    def _pue_impact_score(self, workload_kw: float, total_it_kw: float) -> float:
        """Compute PUE impact score [0, inf)."""
        delta_cooling = workload_kw * _COOLING_OVERHEAD
        delta_pue = delta_cooling / max(total_it_kw, 1.0)
        return delta_pue / 0.1  # normalize: 0.1 delta_pue = score of 1.0

    def _stranded_capacity_loss_score(
        self,
        rack_telemetry,
        workload_kw: float,
        rack_rated_kw: float = _RACK_RATED_KW,
    ) -> float:
        """Compute stranded capacity loss score [0, 1+]."""
        remaining_after = rack_rated_kw - rack_telemetry.power_kw - workload_kw
        return 1.0 - max(0.0, remaining_after) / rack_rated_kw

    def _phase_imbalance_score(
        self,
        rack_index: int,
        workload_kw: float,
        pdu_telemetry_list,
    ) -> float:
        """Compute phase imbalance score [0, inf)."""
        phase = rack_index % 3  # A=0, B=1, C=2
        if not pdu_telemetry_list:
            return 0.0

        # Use first PDU as representative
        pdu = pdu_telemetry_list[0]
        phases = [pdu.phase_a_kw, pdu.phase_b_kw, pdu.phase_c_kw]
        new_phase_load = phases[phase] + workload_kw
        new_phases = phases.copy()
        new_phases[phase] = new_phase_load
        mean_phase_load = sum(new_phases) / 3.0
        return abs(new_phase_load - mean_phase_load) / max(mean_phase_load, 1.0)

    # ── Feasibility checks ───────────────────────────────────────────────────

    def _check_feasibility(
        self,
        rack,
        workload_kw: float,
        pdu_telemetry_list,
        ups_telemetry_list,
    ) -> Optional[str]:
        """
        Return None if feasible, or a reason string if infeasible.
        """
        projected_power = rack.power_kw + workload_kw
        if projected_power > _RACK_POWER_LIMIT:
            return f"Rack power {projected_power:.1f} kW exceeds limit {_RACK_POWER_LIMIT} kW"

        projected_inlet = rack.inlet_temp_c + workload_kw * _THERMAL_RESISTANCE
        if projected_inlet > _ASHRAE_A2_INLET_LIMIT:
            return f"Projected inlet {projected_inlet:.1f}°C exceeds ASHRAE A2 limit ({_ASHRAE_A2_INLET_LIMIT}°C)"

        # PDU feasibility check — match by room (handle "Room A" vs "A" naming)
        for pdu in pdu_telemetry_list:
            if hasattr(pdu, "room") and hasattr(rack, "room"):
                rack_room = rack.room.replace("Room ", "").strip()
                pdu_room = pdu.room.replace("Room ", "").strip()
                if pdu_room != rack_room:
                    continue
            pdu_new_load = pdu.total_kw + workload_kw
            if pdu_new_load > _PDU_RATED_KW * _PDU_LOAD_LIMIT_PCT:
                return f"PDU {pdu.asset_id} would reach {pdu_new_load:.1f} kW (limit {_PDU_RATED_KW * _PDU_LOAD_LIMIT_PCT:.1f} kW)"

        # UPS feasibility check — find the UPS with most headroom, check if workload fits.
        # In a typical DC, a new rack would be assigned to the least-loaded UPS.
        # The workload only flows through one UPS path.
        if ups_telemetry_list:
            # Find the UPS with the lowest current load (best candidate for new workload)
            best_ups = min(ups_telemetry_list, key=lambda u: u.output_kw)
            ups_limit = _UPS_RATED_KW * _UPS_LOAD_LIMIT_PCT
            # Check: would adding workload to this UPS exceed the per-UPS limit?
            if best_ups.output_kw + workload_kw > ups_limit:
                headroom = max(0.0, ups_limit - best_ups.output_kw)
                return (
                    f"Best UPS {best_ups.asset_id} headroom {headroom:.1f} kW "
                    f"< workload {workload_kw:.1f} kW (UPS limit {ups_limit:.1f} kW)"
                )

        return None

    # ── Public methods ────────────────────────────────────────────────────────

    def score_all_racks(self, workload_kw: float, snapshot) -> List[dict]:
        """Score all racks in the snapshot for the given workload."""
        racks = snapshot.racks
        total_it_kw = sum(r.power_kw for r in racks)
        pdus = getattr(snapshot, "pdus", [])
        upses = getattr(snapshot, "upses", [])

        scores = []
        for idx, rack in enumerate(racks):
            infeasibility = self._check_feasibility(rack, workload_kw, pdus, upses)
            feasible = infeasibility is None

            thermal = self._thermal_risk_score(rack, workload_kw)
            pue = self._pue_impact_score(workload_kw, total_it_kw)
            stranded = self._stranded_capacity_loss_score(rack, workload_kw)
            phase = self._phase_imbalance_score(idx, workload_kw, pdus)

            overall = (
                self.w1 * thermal
                + self.w2 * pue
                + self.w3 * stranded
                + self.w4 * phase
            )

            scores.append({
                "rack_id": rack.asset_id,
                "overall_score": round(overall, 4),
                "thermal_risk": round(thermal, 4),
                "pue_impact": round(pue, 4),
                "stranded_loss": round(stranded, 4),
                "phase_impact": round(phase, 4),
                "feasible": feasible,
                "infeasibility_reason": infeasibility,
            })

        return scores

    def recommend_placement(self, workload_kw: float, snapshot) -> dict:
        """
        Recommend best rack for a workload using MILP.

        Returns dict with recommended_rack_id and scoring details.
        If no feasible rack, recommended_rack_id is None with infeasibility_reason.
        """
        try:
            import pulp
        except ImportError:
            logger.error("PuLP not installed; falling back to greedy placement")
            return self._greedy_placement(workload_kw, snapshot)

        racks = snapshot.racks
        n_racks = len(racks)
        if n_racks == 0:
            return {
                "recommended_rack_id": None,
                "workload_kw": workload_kw,
                "thermal_risk_score": 0.0,
                "projected_inlet_temp": 0.0,
                "projected_outlet_temp": 0.0,
                "ashrae_class": "A1",
                "pue_impact": 0.0,
                "phase_impact": 0.0,
                "overall_score": 0.0,
                "rationale": "No racks available.",
                "infeasibility_reason": "No racks in snapshot",
                "alternatives": [],
            }

        total_it_kw = sum(r.power_kw for r in racks)
        pdus = getattr(snapshot, "pdus", [])
        upses = getattr(snapshot, "upses", [])

        # Score all racks
        scored = []
        for idx, rack in enumerate(racks):
            infeasibility = self._check_feasibility(rack, workload_kw, pdus, upses)
            feasible = infeasibility is None

            thermal = self._thermal_risk_score(rack, workload_kw)
            pue = self._pue_impact_score(workload_kw, total_it_kw)
            stranded = self._stranded_capacity_loss_score(rack, workload_kw)
            phase = self._phase_imbalance_score(idx, workload_kw, pdus)
            overall = self.w1 * thermal + self.w2 * pue + self.w3 * stranded + self.w4 * phase

            scored.append({
                "idx": idx,
                "rack": rack,
                "thermal": thermal,
                "pue": pue,
                "stranded": stranded,
                "phase": phase,
                "overall": overall,
                "feasible": feasible,
                "infeasibility_reason": infeasibility,
            })

        feasible_items = [s for s in scored if s["feasible"]]

        if not feasible_items:
            reasons = list({s["infeasibility_reason"] for s in scored if s["infeasibility_reason"]})
            return {
                "recommended_rack_id": None,
                "workload_kw": workload_kw,
                "thermal_risk_score": 0.0,
                "projected_inlet_temp": 0.0,
                "projected_outlet_temp": 0.0,
                "ashrae_class": "A1",
                "pue_impact": 0.0,
                "phase_impact": 0.0,
                "overall_score": 0.0,
                "rationale": "No feasible rack found.",
                "infeasibility_reason": "; ".join(reasons[:3]),
                "alternatives": [],
            }

        # MILP via PuLP
        prob = pulp.LpProblem("rack_placement", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n_racks)]

        # Objective
        prob += pulp.lpSum(
            s["overall"] * x[s["idx"]] for s in scored
        )

        # Exactly one rack must be chosen
        prob += pulp.lpSum(x) == 1

        # Infeasible racks are forced to 0
        for s in scored:
            if not s["feasible"]:
                prob += x[s["idx"]] == 0

        solver = pulp.PULP_CBC_CMD(
            msg=0,
            timeLimit=_MILP_TIMEOUT_SEC,
        )
        status = prob.solve(solver)

        chosen_idx = None
        if pulp.LpStatus[prob.status] in ("Optimal", "Not Solved"):
            for s in scored:
                if abs(pulp.value(x[s["idx"]]) - 1.0) < 0.5:
                    chosen_idx = s["idx"]
                    break

        # Fallback to greedy if MILP failed
        if chosen_idx is None and feasible_items:
            best = min(feasible_items, key=lambda s: s["overall"])
            chosen_idx = best["idx"]

        if chosen_idx is None:
            return {
                "recommended_rack_id": None,
                "workload_kw": workload_kw,
                "thermal_risk_score": 0.0,
                "projected_inlet_temp": 0.0,
                "projected_outlet_temp": 0.0,
                "ashrae_class": "A1",
                "pue_impact": 0.0,
                "phase_impact": 0.0,
                "overall_score": 0.0,
                "rationale": "MILP solver found no feasible solution.",
                "alternatives": [],
            }

        chosen = scored[chosen_idx]
        chosen_rack = chosen["rack"]
        projected_inlet = chosen_rack.inlet_temp_c + workload_kw * _THERMAL_RESISTANCE
        projected_outlet = chosen_rack.outlet_temp_c + workload_kw * _THERMAL_RESISTANCE * 2.5

        # Top 5 alternatives (feasible, sorted by score, excluding chosen)
        alts = sorted(
            [s for s in feasible_items if s["idx"] != chosen_idx],
            key=lambda s: s["overall"],
        )[:5]

        alternatives = []
        for s in alts:
            alternatives.append({
                "rack_id": s["rack"].asset_id,
                "overall_score": round(s["overall"], 4),
                "thermal_risk": round(s["thermal"], 4),
                "pue_impact": round(s["pue"], 4),
                "stranded_loss": round(s["stranded"], 4),
                "phase_impact": round(s["phase"], 4),
                "feasible": True,
                "infeasibility_reason": None,
            })

        return {
            "recommended_rack_id": chosen_rack.asset_id,
            "workload_kw": workload_kw,
            "thermal_risk_score": round(chosen["thermal"], 4),
            "projected_inlet_temp": round(projected_inlet, 2),
            "projected_outlet_temp": round(projected_outlet, 2),
            "ashrae_class": _ashrae_class(projected_inlet),
            "pue_impact": round(chosen["pue"], 4),
            "phase_impact": round(chosen["phase"], 4),
            "overall_score": round(chosen["overall"], 4),
            "rationale": (
                f"Rack {chosen_rack.asset_id} selected: "
                f"current inlet {chosen_rack.inlet_temp_c:.1f}°C, "
                f"power {chosen_rack.power_kw:.1f} kW, "
                f"projected inlet {projected_inlet:.1f}°C after +{workload_kw:.1f} kW."
            ),
            "alternatives": alternatives,
        }

    def _greedy_placement(self, workload_kw: float, snapshot) -> dict:
        """Greedy fallback when PuLP unavailable."""
        racks = snapshot.racks
        total_it_kw = sum(r.power_kw for r in racks)
        pdus = getattr(snapshot, "pdus", [])
        upses = getattr(snapshot, "upses", [])

        best = None
        best_score = float("inf")

        for idx, rack in enumerate(racks):
            if self._check_feasibility(rack, workload_kw, pdus, upses) is not None:
                continue
            thermal = self._thermal_risk_score(rack, workload_kw)
            pue = self._pue_impact_score(workload_kw, total_it_kw)
            stranded = self._stranded_capacity_loss_score(rack, workload_kw)
            phase = self._phase_imbalance_score(idx, workload_kw, pdus)
            score = self.w1 * thermal + self.w2 * pue + self.w3 * stranded + self.w4 * phase
            if score < best_score:
                best_score = score
                best = (idx, rack, thermal, pue, stranded, phase)

        if best is None:
            return {
                "recommended_rack_id": None,
                "workload_kw": workload_kw,
                "thermal_risk_score": 0.0,
                "projected_inlet_temp": 0.0,
                "projected_outlet_temp": 0.0,
                "ashrae_class": "A1",
                "pue_impact": 0.0,
                "phase_impact": 0.0,
                "overall_score": 0.0,
                "rationale": "No feasible rack found.",
                "alternatives": [],
            }

        idx, rack, thermal, pue, stranded, phase = best
        projected_inlet = rack.inlet_temp_c + workload_kw * _THERMAL_RESISTANCE
        projected_outlet = rack.outlet_temp_c + workload_kw * _THERMAL_RESISTANCE * 2.5
        return {
            "recommended_rack_id": rack.asset_id,
            "workload_kw": workload_kw,
            "thermal_risk_score": round(thermal, 4),
            "projected_inlet_temp": round(projected_inlet, 2),
            "projected_outlet_temp": round(projected_outlet, 2),
            "ashrae_class": _ashrae_class(projected_inlet),
            "pue_impact": round(pue, 4),
            "phase_impact": round(phase, 4),
            "overall_score": round(best_score, 4),
            "rationale": f"Greedy: rack {rack.asset_id} selected.",
            "alternatives": [],
        }

    def find_migration_candidates(self, snapshot, n: int = 5) -> List[dict]:
        """
        Find top n migration candidates from hot/overloaded racks to better destinations.

        Hot racks: inlet > 27°C OR power > 16 kW.
        """
        racks = snapshot.racks
        pdus = getattr(snapshot, "pdus", [])
        upses = getattr(snapshot, "upses", [])
        total_it_kw = sum(r.power_kw for r in racks)

        hot_racks = [r for r in racks if r.inlet_temp_c > 27.0 or r.power_kw > 16.0]

        candidates = []
        for source_rack in hot_racks:
            workload_kw = source_rack.power_kw

            # Find best destination (excluding source)
            best_dest = None
            best_score = float("inf")
            best_thermal_imp = 0.0

            for idx, dest_rack in enumerate(racks):
                if dest_rack.asset_id == source_rack.asset_id:
                    continue
                infeas = self._check_feasibility(dest_rack, workload_kw, pdus, upses)
                if infeas is not None:
                    continue

                thermal = self._thermal_risk_score(dest_rack, workload_kw)
                pue = self._pue_impact_score(workload_kw, total_it_kw)
                stranded = self._stranded_capacity_loss_score(dest_rack, workload_kw)
                phase = self._phase_imbalance_score(idx, workload_kw, pdus)
                score = self.w1 * thermal + self.w2 * pue + self.w3 * stranded + self.w4 * phase

                thermal_imp = source_rack.inlet_temp_c - dest_rack.inlet_temp_c
                if score < best_score:
                    best_score = score
                    best_dest = dest_rack
                    best_thermal_imp = thermal_imp

            if best_dest is None:
                continue

            # Compute benefit score (lower score = higher benefit)
            source_score = (
                self._thermal_risk_score(source_rack, 0)
                + self._stranded_capacity_loss_score(source_rack, 0)
            )
            benefit_score = max(0.0, source_score - best_score)

            # Migration risk
            if workload_kw < 10.0:
                risk = "LOW"
            elif workload_kw <= 16.0:
                risk = "MEDIUM"
            else:
                risk = "HIGH"

            downtime_mins = int(workload_kw * 3)
            estimated_savings_kw = max(0.0, best_thermal_imp * 0.15)
            pue_improvement = best_thermal_imp * 0.001  # rough estimate

            candidates.append({
                "source_rack_id": source_rack.asset_id,
                "destination_rack_id": best_dest.asset_id,
                "benefit_score": round(benefit_score, 4),
                "thermal_improvement_c": round(best_thermal_imp, 2),
                "pue_improvement": round(pue_improvement, 4),
                "estimated_savings_kw": round(estimated_savings_kw, 2),
                "migration_risk": risk,
                "downtime_estimate_mins": downtime_mins,
                "rationale": (
                    f"Move {workload_kw:.1f} kW from {source_rack.asset_id} "
                    f"(inlet {source_rack.inlet_temp_c:.1f}°C) to "
                    f"{best_dest.asset_id} (inlet {best_dest.inlet_temp_c:.1f}°C)"
                ),
            })

        # Sort by benefit score descending, return top n
        candidates.sort(key=lambda c: c["benefit_score"], reverse=True)
        return candidates[:n]

    def get_thermal_preview(
        self, rack_id: str, workload_kw: float, snapshot
    ) -> Optional[dict]:
        """
        Return thermal before/after preview for adding workload to a specific rack.
        Returns None if rack not found.
        """
        rack = next((r for r in snapshot.racks if r.asset_id == rack_id), None)
        if rack is None:
            return None

        before_inlet = rack.inlet_temp_c
        before_outlet = rack.outlet_temp_c
        before_ashrae = _ashrae_class(before_inlet)

        after_inlet = before_inlet + workload_kw * _THERMAL_RESISTANCE
        after_outlet = before_outlet + workload_kw * _THERMAL_RESISTANCE * 2.5
        after_ashrae = _ashrae_class(after_inlet)

        thermal_risk = self._thermal_risk_score(rack, workload_kw)

        if after_inlet <= 24.0:
            risk_level = "safe"
        elif after_inlet <= 32.0:
            risk_level = "warning"
        else:
            risk_level = "critical"

        return {
            "rack_id": rack_id,
            "workload_kw": workload_kw,
            "before_inlet_c": round(before_inlet, 2),
            "before_outlet_c": round(before_outlet, 2),
            "before_ashrae_class": before_ashrae,
            "after_inlet_c": round(after_inlet, 2),
            "after_outlet_c": round(after_outlet, 2),
            "after_ashrae_class": after_ashrae,
            "thermal_risk_score": round(thermal_risk, 4),
            "risk_level": risk_level,
        }


# ─── Module-level singleton ───────────────────────────────────────────────────

placement_optimizer = WorkloadPlacementOptimizer()
