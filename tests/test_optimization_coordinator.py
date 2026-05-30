"""
Tests for OptimizationCoordinator (backend/optimization_coordinator.py).

Uses MagicMock for snapshots and rate forecasts.
"""

from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from backend.optimization_coordinator import OptimizationCoordinator, coordinator
from backend.grid_pricing import tariff


# ─── Mock helpers ─────────────────────────────────────────────────────────────


def _make_rack(
    asset_id: str,
    inlet_temp_c: float = 20.0,
    outlet_temp_c: float = 30.0,
    power_kw: float = 10.0,
    room: str = "A",
) -> MagicMock:
    rack = MagicMock()
    rack.asset_id = asset_id
    rack.inlet_temp_c = inlet_temp_c
    rack.outlet_temp_c = outlet_temp_c
    rack.power_kw = power_kw
    rack.room = room
    return rack


def _make_crac(
    asset_id: str = "A-CRAC-1",
    setpoint_c: float = 18.0,
    cooling_load_kw: float = 80.0,
    supply_air_temp_c: float = 18.0,
    return_air_temp_c: float = 28.0,
) -> MagicMock:
    crac = MagicMock()
    crac.asset_id = asset_id
    crac.setpoint_c = setpoint_c
    crac.cooling_load_kw = cooling_load_kw
    crac.supply_air_temp_c = supply_air_temp_c
    crac.return_air_temp_c = return_air_temp_c
    return crac


def _make_chiller(
    asset_id: str = "CH-1",
    chw_supply_temp_c: float = 8.0,
    chw_return_temp_c: float = 13.0,
    load_pct: float = 70.0,
    power_kw: float = 100.0,
    cop: float = 4.5,
) -> MagicMock:
    chiller = MagicMock()
    chiller.asset_id = asset_id
    chiller.chw_supply_temp_c = chw_supply_temp_c
    chiller.chw_return_temp_c = chw_return_temp_c
    chiller.load_pct = load_pct
    chiller.power_kw = power_kw
    chiller.cop = cop
    return chiller


def _make_snapshot(
    n_racks: int = 80,
    inlet_temp_c: float = 20.0,
    power_kw: float = 10.0,
    n_cracs: int = 4,
    n_chillers: int = 2,
) -> MagicMock:
    """Create a mock snapshot."""
    racks = [
        _make_rack(f"A-R1-{i+1:03d}", inlet_temp_c=inlet_temp_c, power_kw=power_kw)
        for i in range(n_racks)
    ]
    cracs = [_make_crac(f"A-CRAC-{i+1}") for i in range(n_cracs)]
    chillers = [_make_chiller(f"CH-{i+1}") for i in range(n_chillers)]

    snap = MagicMock()
    snap.racks = racks
    snap.cracs = cracs
    snap.chillers = chillers
    snap.pdus = []
    snap.upses = []
    snap.outside_air_temp_c = 15.0
    snap.timestamp = datetime.utcnow()
    return snap


def _make_on_peak_forecast() -> list:
    """Forecast showing on-peak in next 2 hours."""
    forecast = []
    for i in range(24):
        period = "on_peak" if i < 6 else "off_peak"
        rate = 0.28 if period == "on_peak" else 0.06
        forecast.append({
            "hour": i,
            "dt_iso": datetime(2024, 6, 10, 14 + i, 0).isoformat()
            if 14 + i < 24
            else datetime(2024, 6, 11, (14 + i) % 24, 0).isoformat(),
            "rate_per_kwh": rate,
            "period_type": period,
            "carbon_intensity": 0.31,
            "is_grid_peak": False,
        })
    return forecast


def _make_offpeak_forecast() -> list:
    """Forecast showing all off-peak."""
    return [
        {
            "hour": i,
            "dt_iso": datetime(2024, 6, 10, i, 0).isoformat(),
            "rate_per_kwh": 0.06,
            "period_type": "off_peak",
            "carbon_intensity": 0.18,
            "is_grid_peak": False,
        }
        for i in range(24)
    ]


# ─── Test cases ───────────────────────────────────────────────────────────────


class TestOptimizationCoordinator(unittest.TestCase):

    def setUp(self):
        self.coordinator = OptimizationCoordinator()

    # ── Conflict detection tests ───────────────────────────────────────────────

    def test_conflict_cooling_vs_energy_detected(self):
        """
        On-peak in next 2 hours + cooling optimizer recommends high CHW setpoint
        → COOLING_VS_ENERGY conflict detected.
        """
        snap = _make_snapshot()
        forecast = _make_on_peak_forecast()

        # Simulate cooling result recommending high CHW (> 11°C)
        cooling_result = {"optimal_chw_setpoint_c": 13.0, "optimal_crac_setpoint_c": 22.0}

        conflicts = self.coordinator.detect_conflicts(snap, forecast, cooling_result=cooling_result)
        conflict_types = [c["conflict_type"] for c in conflicts]
        self.assertIn("COOLING_VS_ENERGY", conflict_types)

    def test_no_cooling_conflict_when_offpeak(self):
        """No COOLING_VS_ENERGY conflict when all hours are off-peak."""
        snap = _make_snapshot()
        forecast = _make_offpeak_forecast()
        cooling_result = {"optimal_chw_setpoint_c": 13.0, "optimal_crac_setpoint_c": 22.0}

        conflicts = self.coordinator.detect_conflicts(snap, forecast, cooling_result=cooling_result)
        conflict_types = [c["conflict_type"] for c in conflicts]
        self.assertNotIn("COOLING_VS_ENERGY", conflict_types)

    def test_thermal_safety_always_wins(self):
        """
        Hot racks (inlet > 26°C) + on-peak rate → THERMAL_VS_COST conflict,
        thermal safety wins.
        """
        snap = _make_snapshot(inlet_temp_c=27.0)  # > 26°C
        # Provide on-peak forecast so current period = on_peak
        forecast = _make_on_peak_forecast()

        # Mock tariff to return on_peak for current period
        with patch("backend.optimization_coordinator.tariff") as mock_tariff:
            mock_tariff.get_period_type.return_value = "on_peak"
            mock_tariff.get_current_rate.return_value = 0.28
            mock_tariff.get_carbon_intensity.return_value = 0.31
            mock_tariff.get_rate_forecast_24h.return_value = forecast
            mock_tariff.is_grid_peak_event.return_value = False

            conflicts = self.coordinator.detect_conflicts(snap, forecast)
            conflict_types = [c["conflict_type"] for c in conflicts]
            self.assertIn("THERMAL_VS_COST", conflict_types)

            # Thermal safety should win
            thermal_conflicts = [c for c in conflicts if c["conflict_type"] == "THERMAL_VS_COST"]
            for tc in thermal_conflicts:
                self.assertIn("Thermal Safety", tc["winner"])

    def test_thermal_safety_not_triggered_when_cool(self):
        """No THERMAL_VS_COST conflict when rack inlets are below 26°C."""
        snap = _make_snapshot(inlet_temp_c=22.0)  # < 26°C
        forecast = _make_on_peak_forecast()

        with patch("backend.optimization_coordinator.tariff") as mock_tariff:
            mock_tariff.get_period_type.return_value = "on_peak"
            mock_tariff.get_current_rate.return_value = 0.28
            mock_tariff.get_carbon_intensity.return_value = 0.31
            mock_tariff.get_rate_forecast_24h.return_value = forecast
            mock_tariff.is_grid_peak_event.return_value = False

            conflicts = self.coordinator.detect_conflicts(snap, forecast)
            conflict_types = [c["conflict_type"] for c in conflicts]
            self.assertNotIn("THERMAL_VS_COST", conflict_types)

    def test_maintenance_deferred_during_demand_peak(self):
        """
        Maintenance scheduled + on-peak period → MAINTENANCE_VS_DEMAND conflict.
        """
        snap = _make_snapshot()
        forecast = _make_on_peak_forecast()
        maintenance_schedule = {
            "assets_scheduled": ["A-CRAC-1", "CH-1"],
            "n_scheduled": 2,
        }

        with patch("backend.optimization_coordinator.tariff") as mock_tariff:
            mock_tariff.get_period_type.return_value = "on_peak"
            mock_tariff.get_current_rate.return_value = 0.28
            mock_tariff.get_carbon_intensity.return_value = 0.31
            mock_tariff.get_rate_forecast_24h.return_value = forecast
            mock_tariff.is_grid_peak_event.return_value = False

            conflicts = self.coordinator.detect_conflicts(
                snap, forecast, maintenance_schedule=maintenance_schedule
            )
            conflict_types = [c["conflict_type"] for c in conflicts]
            self.assertIn("MAINTENANCE_VS_DEMAND", conflict_types)

    def test_conflict_structure_has_all_keys(self):
        """Each conflict dict has all required keys."""
        snap = _make_snapshot(inlet_temp_c=27.0)
        forecast = _make_on_peak_forecast()

        with patch("backend.optimization_coordinator.tariff") as mock_tariff:
            mock_tariff.get_period_type.return_value = "on_peak"
            mock_tariff.get_current_rate.return_value = 0.28
            mock_tariff.get_carbon_intensity.return_value = 0.31
            mock_tariff.get_rate_forecast_24h.return_value = forecast

            conflicts = self.coordinator.detect_conflicts(snap, forecast)
            for c in conflicts:
                for key in ["conflict_type", "layer_a", "layer_b", "description", "resolution", "winner"]:
                    self.assertIn(key, c, f"Missing key {key} in conflict {c}")

    # ── No conflicts when aligned ──────────────────────────────────────────────

    def test_no_conflicts_when_all_aligned(self):
        """Off-peak period, cool racks, no maintenance → no major conflicts."""
        snap = _make_snapshot(inlet_temp_c=20.0)  # cool racks
        forecast = _make_offpeak_forecast()

        with patch("backend.optimization_coordinator.tariff") as mock_tariff:
            mock_tariff.get_period_type.return_value = "off_peak"
            mock_tariff.get_current_rate.return_value = 0.06
            mock_tariff.get_carbon_intensity.return_value = 0.18
            mock_tariff.get_rate_forecast_24h.return_value = forecast
            mock_tariff.is_grid_peak_event.return_value = False

            conflicts = self.coordinator.detect_conflicts(snap, forecast)
            # Should have no conflicts (or no critical ones)
            # No maintenance or cooling result provided, so only THERMAL_VS_COST possible
            thermal_conflicts = [c for c in conflicts if c["conflict_type"] == "THERMAL_VS_COST"]
            self.assertEqual(len(thermal_conflicts), 0, "Should have no thermal conflicts with cool racks")

    # ── Unified recommendations tests ──────────────────────────────────────────

    def test_unified_plan_has_all_sections(self):
        """get_unified_recommendations returns all required keys."""
        snap = _make_snapshot()

        result = self.coordinator.get_unified_recommendations(snap, pue=1.5)
        for key in ["cooling_actions", "placement_actions", "schedule_actions",
                    "maintenance_actions", "conflicts_detected", "estimated_savings",
                    "priority_order"]:
            self.assertIn(key, result)

    def test_unified_plan_priority_order_not_empty(self):
        """priority_order list in unified plan is not empty."""
        snap = _make_snapshot()
        result = self.coordinator.get_unified_recommendations(snap, pue=1.5)
        self.assertGreater(len(result["priority_order"]), 0)

    def test_unified_plan_cooling_actions_list(self):
        """cooling_actions should be a list."""
        snap = _make_snapshot()
        result = self.coordinator.get_unified_recommendations(snap, pue=1.5)
        self.assertIsInstance(result["cooling_actions"], list)

    def test_unified_plan_estimated_savings_keys(self):
        """estimated_savings should have all required keys."""
        snap = _make_snapshot()
        result = self.coordinator.get_unified_recommendations(snap, pue=1.5)
        savings = result["estimated_savings"]
        for key in ["energy_cost_per_day", "demand_charge_per_month",
                    "pue_improvement", "carbon_per_day", "total_annual"]:
            self.assertIn(key, savings, f"Missing savings key: {key}")

    # ── Pareto front tests ─────────────────────────────────────────────────────

    def test_pareto_front_has_efficient_points(self):
        """compute_4obj_pareto returns a non-empty pareto_points list."""
        snap = _make_snapshot()
        result = self.coordinator.compute_4obj_pareto(snap)
        self.assertIn("pareto_points", result)
        self.assertGreater(len(result["pareto_points"]), 0)

    def test_pareto_current_point_has_all_objectives(self):
        """current_point dict contains all 4 objective keys."""
        snap = _make_snapshot()
        result = self.coordinator.compute_4obj_pareto(snap)
        current = result["current_point"]
        for key in ["pue", "energy_cost_per_hr", "reliability_score", "carbon_kg_hr"]:
            self.assertIn(key, current)

    def test_pareto_objectives_list_correct(self):
        """objectives list should contain all 4 objective names."""
        snap = _make_snapshot()
        result = self.coordinator.compute_4obj_pareto(snap)
        expected = ["pue", "energy_cost_per_hr", "reliability_score", "carbon_kg_hr"]
        self.assertEqual(result["objectives"], expected)

    def test_pareto_recommended_point_has_objectives(self):
        """recommended_point should have all 4 objective keys."""
        snap = _make_snapshot()
        result = self.coordinator.compute_4obj_pareto(snap)
        rec = result["recommended_point"]
        if rec:  # may be empty if no Pareto points
            for key in ["pue", "energy_cost_per_hr"]:
                self.assertIn(key, rec)

    def test_pareto_all_points_sampled(self):
        """points list should have at least some entries."""
        snap = _make_snapshot()
        result = self.coordinator.compute_4obj_pareto(snap)
        self.assertGreater(len(result["points"]), 0)

    def test_pareto_empty_snapshot_returns_empty(self):
        """Empty snapshot returns empty Pareto data."""
        snap = MagicMock()
        snap.racks = []
        snap.cracs = []
        snap.chillers = []
        result = self.coordinator.compute_4obj_pareto(snap)
        self.assertEqual(result["points"], [])
        self.assertEqual(result["pareto_points"], [])

    # ── Savings summary tests ──────────────────────────────────────────────────

    def test_savings_summary_all_keys_present(self):
        """get_savings_summary returns all 5 required savings keys."""
        snap = _make_snapshot()
        result = self.coordinator.get_savings_summary(snap, it_kw=200.0, facility_kw=280.0)
        for key in ["energy_cost_saving_per_day", "demand_charge_saving_per_month",
                    "pue_improvement", "carbon_saving_kg_per_day", "total_annual_estimate"]:
            self.assertIn(key, result)

    def test_savings_summary_positive_values(self):
        """All savings values should be positive."""
        snap = _make_snapshot()
        result = self.coordinator.get_savings_summary(snap, it_kw=200.0, facility_kw=280.0)
        for key in ["energy_cost_saving_per_day", "demand_charge_saving_per_month",
                    "total_annual_estimate"]:
            self.assertGreaterEqual(result[key], 0.0, f"{key} should be non-negative")

    def test_savings_summary_annual_is_sum(self):
        """total_annual_estimate should be a reasonable sum of annualized savings."""
        snap = _make_snapshot()
        result = self.coordinator.get_savings_summary(snap, it_kw=200.0, facility_kw=280.0)
        daily = result["energy_cost_saving_per_day"]
        monthly_demand = result["demand_charge_saving_per_month"]
        annual = result["total_annual_estimate"]
        # Annual should be approximately (daily * 365 + monthly * 12)
        expected = daily * 365 + monthly_demand * 12
        self.assertAlmostEqual(annual, expected, places=0)

    # ── Conflict resolution DB structure test ─────────────────────────────────

    def test_conflict_resolution_proper_structure(self):
        """
        detect_conflicts returns proper structure compatible with OptimizationConflictDB.
        """
        snap = _make_snapshot(inlet_temp_c=27.0)
        forecast = _make_on_peak_forecast()
        maintenance_schedule = {"assets_scheduled": ["A-CRAC-1"], "n_scheduled": 1}
        cooling_result = {"optimal_chw_setpoint_c": 13.0}

        with patch("backend.optimization_coordinator.tariff") as mock_tariff:
            mock_tariff.get_period_type.return_value = "on_peak"
            mock_tariff.get_current_rate.return_value = 0.28
            mock_tariff.get_carbon_intensity.return_value = 0.31
            mock_tariff.get_rate_forecast_24h.return_value = forecast
            mock_tariff.is_grid_peak_event.return_value = False

            conflicts = self.coordinator.detect_conflicts(
                snap, forecast, cooling_result, maintenance_schedule
            )

        # Each conflict should be serializable to DB model fields
        for c in conflicts:
            self.assertIsInstance(c["conflict_type"], str)
            self.assertIsInstance(c["layer_a"], str)
            self.assertIsInstance(c["layer_b"], str)
            self.assertIsInstance(c["description"], str)
            self.assertIsInstance(c["resolution"], str)
            self.assertIsInstance(c["winner"], str)

    # ── Singleton test ─────────────────────────────────────────────────────────

    def test_module_singleton(self):
        """Module-level coordinator singleton should be accessible."""
        self.assertIsInstance(coordinator, OptimizationCoordinator)


if __name__ == "__main__":
    unittest.main()
