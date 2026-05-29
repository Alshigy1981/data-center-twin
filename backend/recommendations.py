"""
Cooling and power optimization recommendation engine.

All check_* functions operate on plain numeric inputs so they can be tested
independently.  generate_all_recommendations() takes a TelemetrySnapshot and
orchestrates all checks, returning a deduplicated list sorted by priority.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from backend.models import Priority, TelemetrySnapshot

# ─────────────────────────────────────────
# Thresholds (engineering constants)
# ─────────────────────────────────────────

# Cooling setpoint adjustment triggers
SETPOINT_RAISE_INLET_MAX_C = 24.0     # All rack inlets below this → consider raising CHW setpoint
SETPOINT_LOWER_INLET_THRESHOLD_C = 27.0  # Any rack inlet above this → lower CHW setpoint

HEALTHY_DELTA_T_MIN = 8.0             # CRAC delta-T below this = low delta-T syndrome
HOT_AISLE_THRESHOLD_C = 35.0         # Hot aisle alert for airflow containment check
ROW_IMBALANCE_THRESHOLD_PCT = 30.0   # Row is this % hotter than mean → flag unbalanced load
UNDERPERFORMING_CRAC_DELTA = 3.0     # CRAC delta-T this far below fleet mean → underperforming

# Expected savings models
CHW_SETPOINT_1C_SAVINGS_PCT = 3.0    # Raising CW setpoint 1°C saves ~3% chiller energy
CRAC_RATED_KW = 100.0


# ─────────────────────────────────────────
# Recommendation ID helper
# ─────────────────────────────────────────

def _rec_id(rec_type: str, asset_id: Optional[str] = None) -> str:
    key = f"{rec_type}::{asset_id or 'facility'}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


# ─────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────

def check_increase_chilled_water_setpoint(
    avg_inlet_temp_c: float,
    max_inlet_temp_c: float,
    crac_delta_t: float,
    current_setpoint_c: float = 8.0,
) -> Optional[Dict]:
    """
    Recommend raising the chilled-water setpoint by 1–2°C when:
    - All rack inlets are below SETPOINT_RAISE_INLET_MAX_C (24°C), AND
    - CRAC delta-T is healthy (≥ HEALTHY_DELTA_T_MIN).

    Raising the setpoint reduces chiller compressor work by ~3 % per °C.
    """
    if max_inlet_temp_c >= SETPOINT_RAISE_INLET_MAX_C:
        return None  # Inlets too warm; do not raise setpoint
    if crac_delta_t < HEALTHY_DELTA_T_MIN:
        return None  # Low delta-T indicates cooling inefficiency; fix that first

    raise_by = 2.0 if avg_inlet_temp_c < 21.0 else 1.0
    new_setpoint = current_setpoint_c + raise_by
    # Approximate savings: 3% per degree × number of CRAC units × rated cooling kW
    estimated_savings_kw = raise_by * CHW_SETPOINT_1C_SAVINGS_PCT / 100 * CRAC_RATED_KW * 8
    pue_improvement = estimated_savings_kw / max(1, avg_inlet_temp_c * 5)  # rough estimate

    return {
        "id": _rec_id("raise_chw_setpoint"),
        "recommendation_type": "cooling_setpoint_raise",
        "asset_id": None,
        "priority": Priority.MEDIUM.value,
        "action_text": (
            f"Raise chilled-water setpoint by {raise_by:.0f}°C "
            f"(current {current_setpoint_c:.1f}°C → {new_setpoint:.1f}°C).  "
            f"All rack inlets at {avg_inlet_temp_c:.1f}°C mean / {max_inlet_temp_c:.1f}°C max — "
            f"within safe range with healthy delta-T of {crac_delta_t:.1f}°C."
        ),
        "expected_savings_kw": round(estimated_savings_kw, 1),
        "pue_impact": round(pue_improvement, 4),
        "confidence": 85,
        "rationale": (
            f"Raising chilled-water supply temperature by {raise_by:.0f}°C reduces chiller compressor lift, "
            f"saving approximately {estimated_savings_kw:.1f} kW cooling energy.  "
            f"ASHRAE guideline: operate at the warmest setpoint that keeps inlet temps below 27°C."
        ),
    }


def check_decrease_chilled_water_setpoint(
    max_inlet_temp_c: float,
    current_setpoint_c: float = 8.0,
) -> Optional[Dict]:
    """
    Recommend lowering the chilled-water setpoint when any rack inlet
    exceeds SETPOINT_LOWER_INLET_THRESHOLD_C (27°C).
    """
    if max_inlet_temp_c <= SETPOINT_LOWER_INLET_THRESHOLD_C:
        return None

    overshoot = max_inlet_temp_c - SETPOINT_LOWER_INLET_THRESHOLD_C
    lower_by = max(1.0, min(4.0, overshoot / 2))
    new_setpoint = max(6.0, current_setpoint_c - lower_by)

    return {
        "id": _rec_id("lower_chw_setpoint"),
        "recommendation_type": "cooling_setpoint_lower",
        "asset_id": None,
        "priority": Priority.HIGH.value,
        "action_text": (
            f"Lower chilled-water setpoint by {lower_by:.1f}°C "
            f"(current {current_setpoint_c:.1f}°C → {new_setpoint:.1f}°C).  "
            f"Rack inlet maximum is {max_inlet_temp_c:.1f}°C — above the 27°C operating threshold."
        ),
        "expected_savings_kw": 0.0,
        "pue_impact": 0.0,
        "confidence": 90,
        "rationale": (
            f"Rack inlet temperature {max_inlet_temp_c:.1f}°C exceeds the 27°C recommended upper bound "
            f"(ASHRAE A1 class design intent).  Lowering setpoint will increase cooling capacity and "
            f"reduce risk of thermal throttling or hardware shutdown."
        ),
    }


def check_low_delta_t(
    fleet_mean_delta_t: float,
    crac_delta_t_list: List[float],
    crac_ids: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Identify low delta-T syndrome:
    - Fleet mean below HEALTHY_DELTA_T_MIN (8°C): facility-wide recommendation.
    - Individual CRAC more than UNDERPERFORMING_CRAC_DELTA (3°C) below fleet mean.

    Low delta-T means cool supply air is short-circuiting back to CRAC return
    without absorbing rack heat — indicating containment failure or bypass paths.
    """
    results: List[Dict] = []
    ids = crac_ids or [f"CRAC-{i + 1}" for i in range(len(crac_delta_t_list))]

    if fleet_mean_delta_t < HEALTHY_DELTA_T_MIN:
        results.append(
            {
                "id": _rec_id("low_delta_t_facility"),
                "recommendation_type": "low_delta_t_syndrome",
                "asset_id": None,
                "priority": Priority.HIGH.value,
                "action_text": (
                    f"Investigate airflow containment.  Fleet mean CRAC delta-T is "
                    f"{fleet_mean_delta_t:.1f}°C — below the 8°C healthy minimum.  "
                    f"Seal blanking panels, repair aisle containment curtains, and check for floor tile placement errors."
                ),
                "expected_savings_kw": 15.0,
                "pue_impact": 0.05,
                "confidence": 88,
                "rationale": (
                    f"Low delta-T ({fleet_mean_delta_t:.1f}°C) indicates significant bypass airflow.  "
                    f"Correcting containment typically reduces cooling energy 10–25 % and lowers PUE by 0.05–0.15."
                ),
            }
        )

    for crac_id, dt in zip(ids, crac_delta_t_list):
        if dt < (fleet_mean_delta_t - UNDERPERFORMING_CRAC_DELTA):
            results.append(
                {
                    "id": _rec_id("underperforming_crac", crac_id),
                    "recommendation_type": "underperforming_crac",
                    "asset_id": crac_id,
                    "priority": Priority.MEDIUM.value,
                    "action_text": (
                        f"Inspect CRAC unit {crac_id}: delta-T {dt:.1f}°C is "
                        f"{fleet_mean_delta_t - dt:.1f}°C below fleet mean {fleet_mean_delta_t:.1f}°C.  "
                        f"Check coil fouling, refrigerant charge, and supply plenum routing."
                    ),
                    "expected_savings_kw": 5.0,
                    "pue_impact": 0.02,
                    "confidence": 75,
                    "rationale": (
                        f"A CRAC unit with {UNDERPERFORMING_CRAC_DELTA:.0f}°C lower delta-T than fleet peers "
                        f"is delivering less useful cooling per kW of fan and compressor energy."
                    ),
                }
            )

    return results


def check_unbalanced_rack_load(
    row_avg_inlet_temps: Dict[str, float],
) -> Optional[Dict]:
    """
    Flag when any row is more than ROW_IMBALANCE_THRESHOLD_PCT (30%) above
    the mean row temperature — indicating unbalanced IT load distribution.
    """
    if not row_avg_inlet_temps:
        return None
    mean_temp = sum(row_avg_inlet_temps.values()) / len(row_avg_inlet_temps)
    hottest_row = max(row_avg_inlet_temps, key=row_avg_inlet_temps.get)
    hottest_temp = row_avg_inlet_temps[hottest_row]

    if mean_temp <= 0:
        return None
    deviation_pct = (hottest_temp - mean_temp) / mean_temp * 100

    if deviation_pct < ROW_IMBALANCE_THRESHOLD_PCT:
        return None

    return {
        "id": _rec_id("unbalanced_row_load", hottest_row),
        "recommendation_type": "unbalanced_rack_load",
        "asset_id": hottest_row,
        "priority": Priority.MEDIUM.value,
        "action_text": (
            f"Row {hottest_row} inlet temperature {hottest_temp:.1f}°C is "
            f"{deviation_pct:.0f} % above row mean {mean_temp:.1f}°C.  "
            f"Redistribute IT workloads to cooler rows or add targeted airflow to row {hottest_row}."
        ),
        "expected_savings_kw": 3.0,
        "pue_impact": 0.01,
        "confidence": 78,
        "rationale": (
            f"A {deviation_pct:.0f} % row temperature imbalance suggests uneven IT load distribution.  "
            f"Balancing workloads reduces thermal hotspots and improves overall CRAC efficiency."
        ),
    }


def check_hot_aisle_containment(
    max_outlet_temp_c: float,
    hot_rack_id: Optional[str] = None,
) -> Optional[Dict]:
    """
    Recommend airflow containment check when hot aisle outlet temps exceed
    HOT_AISLE_THRESHOLD_C (35°C).
    """
    if max_outlet_temp_c < HOT_AISLE_THRESHOLD_C:
        return None

    return {
        "id": _rec_id("hot_aisle_containment", hot_rack_id),
        "recommendation_type": "hot_aisle_containment",
        "asset_id": hot_rack_id,
        "priority": Priority.HIGH.value,
        "action_text": (
            f"Rack outlet temperature {max_outlet_temp_c:.1f}°C exceeds 35°C hot-aisle threshold.  "
            f"Inspect physical containment: seal gaps in row end caps, overhead containment panels, "
            f"and between racks.  Consider increasing CRAC airflow on nearest unit."
        ),
        "expected_savings_kw": 8.0,
        "pue_impact": 0.04,
        "confidence": 82,
        "rationale": (
            f"Outlet temperatures above 35°C indicate recirculation of exhaust air back to intakes.  "
            f"This degrades CRAC effectiveness and risks equipment thermal throttling."
        ),
    }


# ─────────────────────────────────────────
# Aggregate recommendation generator
# ─────────────────────────────────────────

PRIORITY_ORDER = {Priority.HIGH.value: 0, Priority.MEDIUM.value: 1, Priority.LOW.value: 2}


def generate_all_recommendations(snapshot: TelemetrySnapshot) -> List[Dict]:
    """
    Run all recommendation checks against a complete telemetry snapshot.

    Returns a deduplicated list sorted by priority (HIGH first).
    """
    results: List[Dict] = []

    rack_inlets = [r.inlet_temp_c for r in snapshot.racks]
    rack_outlets = [r.outlet_temp_c for r in snapshot.racks]
    crac_delta_ts = [c.delta_t_c for c in snapshot.cracs]
    crac_ids = [c.asset_id for c in snapshot.cracs]

    if not rack_inlets or not crac_delta_ts:
        return []

    avg_inlet = sum(rack_inlets) / len(rack_inlets)
    max_inlet = max(rack_inlets)
    max_outlet = max(rack_outlets)
    fleet_mean_delta_t = sum(crac_delta_ts) / len(crac_delta_ts) if crac_delta_ts else 0

    # Build row temperature map
    row_temps: Dict[str, List[float]] = {}
    for r in snapshot.racks:
        key = f"{r.room} {r.row}"
        row_temps.setdefault(key, []).append(r.inlet_temp_c)
    row_avg = {k: sum(v) / len(v) for k, v in row_temps.items()}

    # ── Cooling setpoint ──────────────────
    raise_rec = check_increase_chilled_water_setpoint(
        avg_inlet, max_inlet, fleet_mean_delta_t
    )
    if raise_rec:
        results.append(raise_rec)

    lower_rec = check_decrease_chilled_water_setpoint(max_inlet)
    if lower_rec:
        results.append(lower_rec)

    # ── Low delta-T ───────────────────────
    low_dt = check_low_delta_t(fleet_mean_delta_t, crac_delta_ts, crac_ids)
    results.extend(low_dt)

    # ── Unbalanced row load ───────────────
    imbalance = check_unbalanced_rack_load(row_avg)
    if imbalance:
        results.append(imbalance)

    # ── Hot aisle containment ─────────────
    if rack_outlets:
        hot_rack = snapshot.racks[rack_outlets.index(max_outlet)].asset_id
        hot_aisle = check_hot_aisle_containment(max_outlet, hot_rack)
        if hot_aisle:
            results.append(hot_aisle)

    # Sort by priority
    results.sort(key=lambda r: PRIORITY_ORDER.get(r["priority"], 9))

    # Deduplicate by ID (defensive)
    seen = set()
    deduped = []
    for r in results:
        if r["id"] not in seen:
            seen.add(r["id"])
            deduped.append(r)

    return deduped
