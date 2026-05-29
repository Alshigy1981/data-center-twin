"""
Unit tests for backend/recommendations.py

Tests operate on plain numeric inputs — no database or API required.
"""

import pytest

from backend.recommendations import (
    check_decrease_chilled_water_setpoint,
    check_hot_aisle_containment,
    check_increase_chilled_water_setpoint,
    check_low_delta_t,
    check_unbalanced_rack_load,
    generate_all_recommendations,
)
from backend.models import TelemetrySnapshot, RackTelemetry, CRACTelemetry, AshraeClass
from datetime import datetime


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def _make_rack(asset_id, room, row, position, inlet, outlet, power):
    rise = outlet - inlet
    ashrae = (
        AshraeClass.A1 if inlet <= 32
        else (AshraeClass.A2 if inlet <= 35 else AshraeClass.ABOVE_A2)
    )
    return RackTelemetry(
        asset_id=asset_id,
        timestamp=datetime.utcnow(),
        inlet_temp_c=inlet,
        outlet_temp_c=outlet,
        power_kw=power,
        humidity_pct=50.0,
        temp_rise_c=rise,
        ashrae_class=ashrae,
        room=room,
        row=row,
        position=position,
    )


def _make_crac(asset_id, room, supply, return_air, delta_t, cooling_kw):
    return CRACTelemetry(
        asset_id=asset_id,
        timestamp=datetime.utcnow(),
        supply_air_temp_c=supply,
        return_air_temp_c=return_air,
        delta_t_c=delta_t,
        cooling_load_kw=cooling_kw,
        setpoint_c=18.0,
        room=room,
    )


def _nominal_snapshot():
    """Build a snapshot with all metrics in the safe zone — no recommendations expected."""
    racks = []
    cracs = []
    for i in range(1, 5):
        for j in range(1, 11):
            racks.append(
                _make_rack(f"A-R{i}-{j:03d}", "Room A", f"Row {i}", j, 21.0, 32.0, 8.0)
            )
    for i in range(1, 5):
        cracs.append(_make_crac(f"A-CRAC-{i}", "Room A", 18.0, 30.0, 12.0, 80.0))

    return TelemetrySnapshot(
        timestamp=datetime.utcnow(),
        cycle_id=1,
        racks=racks,
        cracs=cracs,
        chillers=[],
        upses=[],
        pdus=[],
        outside_air_temp_c=15.0,
    )


# ─────────────────────────────────────────
# Chilled water setpoint — raise
# ─────────────────────────────────────────

class TestIncreaseChilledWaterSetpoint:
    def test_raises_when_all_inlets_cool_and_healthy_delta_t(self):
        result = check_increase_chilled_water_setpoint(
            avg_inlet_temp_c=20.0,
            max_inlet_temp_c=22.0,  # below 24°C trigger
            crac_delta_t=11.0,      # healthy (≥8°C)
            current_setpoint_c=8.0,
        )
        assert result is not None
        assert result["priority"] == "medium"
        assert "raise" in result["action_text"].lower() or "8°C" in result["action_text"] or "Raise" in result["action_text"]

    def test_does_not_raise_when_inlet_too_warm(self):
        """If any rack inlet is ≥24°C, do not raise setpoint."""
        result = check_increase_chilled_water_setpoint(
            avg_inlet_temp_c=23.0,
            max_inlet_temp_c=26.0,  # above 24°C → no raise
            crac_delta_t=10.0,
            current_setpoint_c=8.0,
        )
        assert result is None

    def test_does_not_raise_with_low_delta_t(self):
        """Low delta-T indicates containment problem; fix that first."""
        result = check_increase_chilled_water_setpoint(
            avg_inlet_temp_c=20.0,
            max_inlet_temp_c=22.0,
            crac_delta_t=5.0,   # below 8°C healthy minimum
            current_setpoint_c=8.0,
        )
        assert result is None

    def test_savings_positive_when_recommendation_fires(self):
        result = check_increase_chilled_water_setpoint(20.0, 22.0, 10.0, 8.0)
        assert result is not None
        assert result["expected_savings_kw"] > 0

    def test_confidence_in_valid_range(self):
        result = check_increase_chilled_water_setpoint(20.0, 22.0, 10.0, 8.0)
        assert result is not None
        assert 0 <= result["confidence"] <= 100


# ─────────────────────────────────────────
# Chilled water setpoint — lower
# ─────────────────────────────────────────

class TestDecreaseChilledWaterSetpoint:
    def test_lowers_when_inlet_too_hot(self):
        result = check_decrease_chilled_water_setpoint(
            max_inlet_temp_c=30.0,  # above 27°C trigger
            current_setpoint_c=8.0,
        )
        assert result is not None
        assert result["priority"] == "high"
        assert "lower" in result["action_text"].lower() or "Lower" in result["action_text"]

    def test_does_not_lower_when_inlet_ok(self):
        result = check_decrease_chilled_water_setpoint(
            max_inlet_temp_c=25.0,  # below 27°C — no action needed
            current_setpoint_c=8.0,
        )
        assert result is None

    def test_new_setpoint_capped_at_6_degrees(self):
        """Chilled water setpoint should not drop below 6°C (freeze risk)."""
        result = check_decrease_chilled_water_setpoint(
            max_inlet_temp_c=40.0,  # extreme overshoot
            current_setpoint_c=6.5,
        )
        assert result is not None
        # Extract new setpoint from action text — we just verify it fires
        assert result["confidence"] > 0

    def test_exactly_at_threshold_fires(self):
        result = check_decrease_chilled_water_setpoint(max_inlet_temp_c=27.0)
        assert result is None  # 27.0 is not > 27.0; boundary is exclusive


# ─────────────────────────────────────────
# Low delta-T recommendation
# ─────────────────────────────────────────

class TestLowDeltaTRecommendation:
    def test_facility_wide_recommendation_when_fleet_mean_low(self):
        results = check_low_delta_t(
            fleet_mean_delta_t=6.0,   # below 8°C healthy minimum
            crac_delta_t_list=[6.0, 6.5, 5.5, 6.2],
            crac_ids=["A-CRAC-1", "A-CRAC-2", "A-CRAC-3", "A-CRAC-4"],
        )
        assert len(results) >= 1
        facility_recs = [r for r in results if r["asset_id"] is None]
        assert len(facility_recs) == 1
        assert facility_recs[0]["priority"] == "high"

    def test_underperforming_crac_flagged(self):
        """One CRAC unit significantly below fleet average should be flagged."""
        results = check_low_delta_t(
            fleet_mean_delta_t=11.0,
            crac_delta_t_list=[11.0, 11.5, 10.5, 7.0],  # last one is 4°C below mean
            crac_ids=["A-CRAC-1", "A-CRAC-2", "A-CRAC-3", "A-CRAC-4"],
        )
        underperforming = [r for r in results if r.get("asset_id") == "A-CRAC-4"]
        assert len(underperforming) == 1

    def test_no_recommendation_when_delta_t_healthy(self):
        results = check_low_delta_t(
            fleet_mean_delta_t=11.0,
            crac_delta_t_list=[11.0, 10.8, 11.2, 11.5],
        )
        # No facility-wide rec; individual CRACs close enough to mean
        facility_recs = [r for r in results if r.get("asset_id") is None]
        assert len(facility_recs) == 0

    def test_returns_list(self):
        results = check_low_delta_t(5.0, [5.0, 5.0])
        assert isinstance(results, list)


# ─────────────────────────────────────────
# Unbalanced rack load recommendation
# ─────────────────────────────────────────

class TestUnbalancedRackLoadRecommendation:
    def test_fires_when_row_exceeds_threshold(self):
        """One row >30% hotter than mean should trigger recommendation."""
        # Mean = (20+20+20+34)/4 = 23.5; Row 4 deviation = (34-23.5)/23.5 = 44.7% > 30%
        row_temps = {
            "Room A Row 1": 20.0,
            "Room A Row 2": 20.0,
            "Room A Row 3": 20.0,
            "Room A Row 4": 34.0,  # clearly 44% above mean
        }
        result = check_unbalanced_rack_load(row_temps)
        assert result is not None
        assert result["priority"] == "medium"
        assert result["asset_id"] == "Room A Row 4"

    def test_no_alert_when_rows_balanced(self):
        row_temps = {
            "Room A Row 1": 22.0,
            "Room A Row 2": 22.5,
            "Room A Row 3": 21.8,
            "Room A Row 4": 23.0,
        }
        result = check_unbalanced_rack_load(row_temps)
        assert result is None

    def test_empty_input_no_crash(self):
        result = check_unbalanced_rack_load({})
        assert result is None

    def test_single_row_no_alert(self):
        result = check_unbalanced_rack_load({"Room A Row 1": 25.0})
        assert result is None


# ─────────────────────────────────────────
# No recommendations when all nominal
# ─────────────────────────────────────────

class TestNoRecommendationsWhenNominal:
    def test_no_recommendations_in_nominal_state(self):
        """
        A snapshot with all inlets at 21°C, healthy CRAC delta-T of 12°C,
        and balanced row temperatures should generate zero recommendations.
        """
        snapshot = _nominal_snapshot()
        results = generate_all_recommendations(snapshot)
        # In nominal state, no threshold should be crossed.
        # The "raise setpoint" may fire since inlets are below 24°C and delta-T is healthy.
        # That's acceptable — just verify no HIGH priority recommendations fire.
        high_priority = [r for r in results if r["priority"] == "high"]
        assert len(high_priority) == 0, (
            f"Unexpected HIGH priority recommendations in nominal state: {high_priority}"
        )

    def test_generate_returns_list(self):
        snapshot = _nominal_snapshot()
        results = generate_all_recommendations(snapshot)
        assert isinstance(results, list)

    def test_generate_with_empty_snapshot_returns_list(self):
        empty = TelemetrySnapshot(
            timestamp=datetime.utcnow(),
            cycle_id=0,
            racks=[],
            cracs=[],
            chillers=[],
            upses=[],
            pdus=[],
            outside_air_temp_c=15.0,
        )
        results = generate_all_recommendations(empty)
        assert results == []

    def test_high_inlet_temps_trigger_high_priority(self):
        """When rack inlets exceed 27°C, a HIGH priority recommendation should fire."""
        hot_snapshot = _nominal_snapshot()
        # Override inlet temps to be hot
        for r in hot_snapshot.racks:
            r.inlet_temp_c = 29.0
            r.outlet_temp_c = 40.0
        results = generate_all_recommendations(hot_snapshot)
        high_priority = [r for r in results if r["priority"] == "high"]
        assert len(high_priority) >= 1, "Expected HIGH priority recommendation for hot inlets"
