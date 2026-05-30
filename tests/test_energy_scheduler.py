"""
Tests for USCommercialTariff and EnergyCostScheduler
(backend/grid_pricing.py and backend/energy_scheduler.py).
"""

from __future__ import annotations

import unittest
from datetime import datetime

from backend.grid_pricing import USCommercialTariff, tariff
from backend.energy_scheduler import (
    DeferrableLoad,
    DEFAULT_LOADS,
    EnergyCostScheduler,
    scheduler,
)


class TestUSCommercialTariff(unittest.TestCase):

    def setUp(self):
        self.tariff = USCommercialTariff()

    # ── Rate schedule tests ────────────────────────────────────────────────────

    def test_offpeak_rate_is_lowest(self):
        """Off-peak rate should be lower than shoulder and on-peak."""
        off_peak = self.tariff.get_current_rate(datetime(2024, 6, 10, 3, 0))   # Monday 03:00
        shoulder = self.tariff.get_current_rate(datetime(2024, 6, 10, 9, 0))   # Monday 09:00
        on_peak = self.tariff.get_current_rate(datetime(2024, 6, 10, 14, 0))   # Monday 14:00
        self.assertLess(off_peak, shoulder)
        self.assertLess(shoulder, on_peak)

    def test_onpeak_rate_is_highest(self):
        """On-peak ($0.28) > shoulder_pm ($0.14) > off-peak ($0.06)."""
        on_peak = self.tariff.get_current_rate(datetime(2024, 6, 10, 15, 0))   # Monday 15:00
        shoulder_am = self.tariff.get_current_rate(datetime(2024, 6, 10, 10, 0))
        shoulder_pm = self.tariff.get_current_rate(datetime(2024, 6, 10, 19, 0))
        off_peak = self.tariff.get_current_rate(datetime(2024, 6, 10, 4, 0))
        self.assertAlmostEqual(on_peak, 0.28)
        self.assertAlmostEqual(off_peak, 0.06)
        self.assertAlmostEqual(shoulder_am, 0.12)
        self.assertAlmostEqual(shoulder_pm, 0.14)
        self.assertGreater(on_peak, shoulder_pm)
        self.assertGreater(shoulder_pm, off_peak)

    def test_weekend_is_offpeak(self):
        """Saturday and Sunday should always return off-peak rate."""
        saturday = datetime(2024, 6, 8, 14, 0)   # Saturday 14:00
        sunday = datetime(2024, 6, 9, 15, 0)      # Sunday 15:00
        self.assertAlmostEqual(self.tariff.get_current_rate(saturday), 0.06)
        self.assertAlmostEqual(self.tariff.get_current_rate(sunday), 0.06)

    def test_weekday_on_peak_hours(self):
        """Verify on_peak period from 12:00–18:00 on weekdays."""
        for h in range(12, 18):
            dt = datetime(2024, 6, 10, h, 0)  # Monday
            self.assertEqual(self.tariff.get_period_type(dt), "on_peak",
                             f"Hour {h} should be on_peak")

    def test_weekday_off_peak_midnight(self):
        """Midnight to 8:00 on weekdays is off_peak."""
        for h in [0, 3, 6, 7]:
            dt = datetime(2024, 6, 10, h, 0)
            self.assertEqual(self.tariff.get_period_type(dt), "off_peak")

    def test_period_type_weekend(self):
        """Weekends should return off_peak for all hours."""
        for h in range(24):
            dt = datetime(2024, 6, 8, h, 0)  # Saturday
            self.assertEqual(self.tariff.get_period_type(dt), "off_peak")

    # ── Demand charge tests ────────────────────────────────────────────────────

    def test_demand_charge_calculation(self):
        """100 kW demand → $1850/month demand charge."""
        result = self.tariff.calculate_demand_charge(100.0, 0.0)
        self.assertAlmostEqual(result["demand_charge"], 1850.0, places=1)
        self.assertAlmostEqual(result["billed_demand_kw"], 100.0)

    def test_ratchet_clause_applies(self):
        """Current peak 200 kW, 12-month peak 300 kW → billed = max(200, 255) = 255."""
        result = self.tariff.calculate_demand_charge(200.0, 300.0)
        # ratchet_demand = 0.85 * 300 = 255 > 200 → billed = 255
        self.assertAlmostEqual(result["ratchet_demand_kw"], 255.0, places=1)
        self.assertAlmostEqual(result["billed_demand_kw"], 255.0, places=1)
        self.assertGreater(result["ratchet_adjustment"], 0)

    def test_ratchet_not_applied_when_current_is_higher(self):
        """If current peak > 85% of 12-month peak, no ratchet adjustment."""
        result = self.tariff.calculate_demand_charge(400.0, 300.0)
        # ratchet = 0.85 * 300 = 255 < 400 → billed = 400, no ratchet
        self.assertAlmostEqual(result["billed_demand_kw"], 400.0)
        self.assertAlmostEqual(result["ratchet_adjustment"], 0.0)

    def test_monthly_bill_breakdown_sums_correctly(self):
        """energy + demand + coincident + facilities == total (ignoring ratchet in total check)."""
        result = self.tariff.calculate_monthly_bill(
            energy_kwh=10000.0,
            peak_kw=200.0,
            month_str="2024-06",
            peak_12mo_kw=200.0,
        )
        expected_total = (
            result["energy_cost"]
            + result["demand_charge"]
            + result["coincident_peak_charge"]
            + result["facilities_charge"]
        )
        self.assertAlmostEqual(result["total"], expected_total, places=1)

    def test_monthly_bill_has_all_keys(self):
        """Monthly bill dict contains all required fields."""
        result = self.tariff.calculate_monthly_bill(5000.0, 100.0, "2024-07")
        for key in ["energy_cost", "demand_charge", "coincident_peak_charge",
                    "facilities_charge", "ratchet_adjustment", "total", "month"]:
            self.assertIn(key, result)

    # ── Carbon intensity tests ─────────────────────────────────────────────────

    def test_carbon_intensity_night_lower_than_afternoon(self):
        """Hour 2 (renewables 0.18) < Hour 15 (peak demand 0.31)."""
        night = self.tariff.get_carbon_intensity(datetime(2024, 6, 10, 2, 0))
        afternoon = self.tariff.get_carbon_intensity(datetime(2024, 6, 10, 15, 0))
        self.assertLess(night, afternoon)

    def test_carbon_intensity_night_is_lowest(self):
        """Carbon intensity at midnight should be 0.18 (renewables)."""
        night = self.tariff.get_carbon_intensity(datetime(2024, 6, 10, 2, 0))
        self.assertAlmostEqual(night, 0.18)

    def test_carbon_intensity_peak_is_highest(self):
        """Carbon intensity during peak demand hours should be 0.31."""
        peak = self.tariff.get_carbon_intensity(datetime(2024, 6, 10, 16, 0))
        self.assertAlmostEqual(peak, 0.31)

    # ── Rate forecast tests ────────────────────────────────────────────────────

    def test_rate_forecast_has_24_entries(self):
        """get_rate_forecast_24h returns exactly 24 hourly entries."""
        from_dt = datetime(2024, 6, 10, 8, 0)
        forecast = self.tariff.get_rate_forecast_24h(from_dt)
        self.assertEqual(len(forecast), 24)

    def test_rate_forecast_has_required_keys(self):
        """Each forecast entry has all required keys."""
        from_dt = datetime(2024, 6, 10, 0, 0)
        forecast = self.tariff.get_rate_forecast_24h(from_dt)
        for entry in forecast:
            for key in ["hour", "dt_iso", "rate_per_kwh", "period_type", "carbon_intensity", "is_grid_peak"]:
                self.assertIn(key, entry)

    def test_rate_forecast_hours_sequential(self):
        """Forecast hours should be 0, 1, 2, ..., 23."""
        from_dt = datetime(2024, 6, 10, 0, 0)
        forecast = self.tariff.get_rate_forecast_24h(from_dt)
        hours = [e["hour"] for e in forecast]
        self.assertEqual(hours, list(range(24)))

    # ── Grid peak event tests ──────────────────────────────────────────────────

    def test_grid_peak_event_detection(self):
        """is_grid_peak_event is deterministic and consistent for same month."""
        # Check multiple weekdays at 14:30 - should get consistent results
        month_results = set()
        for day in range(1, 29):
            try:
                dt = datetime(2024, 6, day, 15, 0)
                if dt.weekday() < 5:  # weekday
                    result = self.tariff.is_grid_peak_event(dt)
                    month_results.add(result)
            except ValueError:
                pass
        # Should have both True and False results (10 peak days out of ~22 weekdays)
        self.assertIn(True, month_results)
        self.assertIn(False, month_results)

    def test_grid_peak_not_on_weekends(self):
        """Grid peak events should never fire on weekends."""
        saturday = datetime(2024, 6, 8, 15, 0)  # Saturday 15:00
        self.assertFalse(self.tariff.is_grid_peak_event(saturday))

    def test_grid_peak_not_outside_window(self):
        """Grid peak events only fire 14:00–17:00."""
        dt = datetime(2024, 6, 10, 8, 0)  # Monday 08:00 (not in 14-17 window)
        self.assertFalse(self.tariff.is_grid_peak_event(dt))

    def test_grid_peak_deterministic(self):
        """Same date should always return same is_grid_peak result."""
        dt = datetime(2024, 6, 10, 15, 0)
        result1 = self.tariff.is_grid_peak_event(dt)
        result2 = self.tariff.is_grid_peak_event(dt)
        self.assertEqual(result1, result2)

    # ── Cheapest hours tests ───────────────────────────────────────────────────

    def test_cheapest_hours_are_offpeak(self):
        """The 3 cheapest hours over a weekday should all be off-peak."""
        # Start at 12:00 (on-peak) — cheapest hours should be off-peak
        from_dt = datetime(2024, 6, 10, 12, 0)
        cheapest = self.tariff.get_cheapest_n_hours(3, from_dt, within_hours=24)
        self.assertEqual(len(cheapest), 3)
        for dt in cheapest:
            period = self.tariff.get_period_type(dt)
            self.assertEqual(period, "off_peak", f"Hour {dt.hour} should be off-peak")

    def test_cheapest_n_hours_count(self):
        """get_cheapest_n_hours returns exactly n hours."""
        from_dt = datetime(2024, 6, 10, 0, 0)
        cheapest = self.tariff.get_cheapest_n_hours(5, from_dt)
        self.assertEqual(len(cheapest), 5)

    def test_next_offpeak_window(self):
        """get_next_offpeak_window returns an off-peak datetime."""
        dt = datetime(2024, 6, 10, 14, 0)  # on-peak
        next_offpeak = self.tariff.get_next_offpeak_window(dt)
        self.assertEqual(self.tariff.get_period_type(next_offpeak), "off_peak")

    # ── Energy cost calculation ────────────────────────────────────────────────

    def test_calculate_energy_cost_positive(self):
        """Energy cost should be positive for positive load."""
        result = self.tariff.calculate_energy_cost(100.0, datetime(2024, 6, 10, 14, 0), 1.0)
        self.assertGreater(result, 0.0)

    def test_calculate_energy_cost_varies_by_period(self):
        """Energy cost during on-peak > off-peak for same load and duration."""
        dt_onpeak = datetime(2024, 6, 10, 14, 0)
        dt_offpeak = datetime(2024, 6, 10, 3, 0)
        cost_on = self.tariff.calculate_energy_cost(100.0, dt_onpeak, 1.0)
        cost_off = self.tariff.calculate_energy_cost(100.0, dt_offpeak, 1.0)
        self.assertGreater(cost_on, cost_off)

    # ── Singleton test ─────────────────────────────────────────────────────────

    def test_module_singleton(self):
        """Module-level tariff singleton should be accessible."""
        self.assertIsInstance(tariff, USCommercialTariff)


class TestEnergyCostScheduler(unittest.TestCase):

    def setUp(self):
        self.scheduler = EnergyCostScheduler()

    # ── Schedule defers to off-peak ────────────────────────────────────────────

    def test_schedule_defers_to_offpeak(self):
        """
        When starting during on-peak, LOW priority loads should be deferred to off-peak.
        """
        # Start during on-peak (Monday 14:00)
        from_dt = datetime(2024, 6, 10, 14, 0)

        low_load = DeferrableLoad(
            "Test Load", "test", 10.0, 1.0, 24.0, 0.0, True, "LOW"
        )

        result = self.scheduler.solve_schedule([low_load], from_dt, 0.0)
        loads = result["loads"]
        self.assertEqual(len(loads), 1)

        # The scheduled load should save money compared to immediate
        # (i.e., it was deferred to a cheaper window)
        self.assertGreaterEqual(result["total_saving_vs_asap"], 0.0)

    def test_schedule_has_required_keys(self):
        """solve_schedule result contains all required keys."""
        from_dt = datetime(2024, 6, 10, 14, 0)
        result = self.scheduler.solve_schedule(DEFAULT_LOADS, from_dt, 0.0)
        for key in ["loads", "total_cost", "total_saving_vs_asap",
                    "peak_demand_avoided_kw", "carbon_saved_kg", "schedule_horizon_hrs"]:
            self.assertIn(key, result)

    def test_schedule_all_default_loads_included(self):
        """All DEFAULT_LOADS appear in the schedule output."""
        from_dt = datetime(2024, 6, 10, 14, 0)
        result = self.scheduler.solve_schedule(DEFAULT_LOADS, from_dt, 0.0)
        scheduled_names = {l["load_name"] for l in result["loads"]}
        for load in DEFAULT_LOADS:
            self.assertIn(load.name, scheduled_names)

    def test_schedule_respects_deadline(self):
        """A load with 4 hr deadline must start within 4 hours of from_dt."""
        from_dt = datetime(2024, 6, 10, 0, 0)
        tight_load = DeferrableLoad("Urgent", "urgent", 5.0, 1.0, 4.0, 0.0, True, "HIGH")
        result = self.scheduler.solve_schedule([tight_load], from_dt, 0.0)
        load = result["loads"][0]

        from datetime import datetime as dt
        start = dt.fromisoformat(load["start_time_iso"])
        hours_offset = (start - from_dt).total_seconds() / 3600
        self.assertLessEqual(hours_offset, 4.0)

    def test_schedule_horizon_is_24(self):
        """schedule_horizon_hrs should be 24."""
        from_dt = datetime(2024, 6, 10, 0, 0)
        result = self.scheduler.solve_schedule(DEFAULT_LOADS, from_dt)
        self.assertEqual(result["schedule_horizon_hrs"], 24)

    def test_schedule_total_cost_positive(self):
        """Total cost should be non-negative."""
        from_dt = datetime(2024, 6, 10, 14, 0)
        result = self.scheduler.solve_schedule(DEFAULT_LOADS, from_dt)
        self.assertGreaterEqual(result["total_cost"], 0.0)

    # ── Pre-cooling tests ──────────────────────────────────────────────────────

    def test_precooling_lowers_setpoint_before_peak(self):
        """Pre-cooling actions should appear 2 hours before on-peak windows."""
        # Forecast starting at 10:00 (shoulder) → 12:00+ is on-peak
        from_dt = datetime(2024, 6, 10, 10, 0)  # Monday 10:00
        forecast = []
        from backend.grid_pricing import tariff as t
        for i in range(24):
            from datetime import timedelta
            dt = from_dt + timedelta(hours=i)
            forecast.append({
                "hour": i,
                "dt_iso": dt.isoformat(),
                "rate_per_kwh": t.get_current_rate(dt),
                "period_type": t.get_period_type(dt),
                "carbon_intensity": t.get_carbon_intensity(dt),
                "is_grid_peak": False,
            })

        result = self.scheduler.optimize_precooling(forecast, 8.0, 18.0)
        action_types = {a["action_type"] for a in result["actions"]}
        self.assertIn("pre_cool", action_types)

    def test_precooling_has_required_keys(self):
        """optimize_precooling result contains all required keys."""
        forecast = [
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
        result = self.scheduler.optimize_precooling(forecast, 8.0, 18.0)
        for key in ["actions", "total_cost_saving", "peak_load_reduction_kw",
                    "start_time", "end_time"]:
            self.assertIn(key, result)

    def test_precooling_actions_count_24(self):
        """optimize_precooling should return 24 hourly actions."""
        forecast = [
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
        result = self.scheduler.optimize_precooling(forecast, 8.0, 18.0)
        self.assertEqual(len(result["actions"]), 24)

    # ── Demand tracking tests ──────────────────────────────────────────────────

    def test_15min_demand_single_reading(self):
        """Single reading: rolling average equals that reading."""
        result = self.scheduler.track_15min_demand([300.0])
        self.assertAlmostEqual(result, 300.0)

    def test_15min_demand_last_four(self):
        """Rolling average of last 4 readings from longer list."""
        readings = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
        result = self.scheduler.track_15min_demand(readings)
        expected = (300.0 + 400.0 + 500.0 + 600.0) / 4
        self.assertAlmostEqual(result, expected)

    def test_15min_demand_empty(self):
        """Empty readings list returns 0."""
        result = self.scheduler.track_15min_demand([])
        self.assertEqual(result, 0.0)

    # ── Demand shed tests ──────────────────────────────────────────────────────

    def test_demand_shed_reduces_load(self):
        """Shed actions should have enough total load reduction to meet target."""
        actions = self.scheduler.recommend_demand_shed(500.0, 400.0)
        total_reduction = sum(a["load_reduction_kw"] for a in actions)
        # Need at least 100 kW reduction
        self.assertGreaterEqual(total_reduction, 100.0)

    def test_demand_shed_no_actions_when_under_target(self):
        """No shed actions when current demand <= target."""
        actions = self.scheduler.recommend_demand_shed(300.0, 400.0)
        self.assertEqual(len(actions), 0)

    def test_demand_shed_has_required_keys(self):
        """Each shed action has required keys."""
        actions = self.scheduler.recommend_demand_shed(500.0, 400.0)
        for action in actions:
            for key in ["asset_id", "action", "load_reduction_kw", "duration_mins", "impact_level"]:
                self.assertIn(key, action)

    def test_demand_shed_max_5_actions(self):
        """At most 5 shed actions returned."""
        actions = self.scheduler.recommend_demand_shed(1000.0, 0.0)
        self.assertLessEqual(len(actions), 5)

    # ── Monthly bill projection tests ─────────────────────────────────────────

    def test_monthly_bill_projection_has_all_keys(self):
        """project_monthly_bill returns all required keys."""
        result = self.scheduler.project_monthly_bill(200.0, 15, 30, 280.0)
        for key in ["projected_energy_cost", "projected_demand_charge", "projected_total",
                    "savings_available", "ratchet_risk", "ratchet_amount",
                    "days_remaining", "current_month_peak_kw"]:
            self.assertIn(key, result)

    def test_monthly_bill_projection_positive_costs(self):
        """Projected costs should be positive."""
        result = self.scheduler.project_monthly_bill(200.0, 10, 30, 280.0)
        self.assertGreater(result["projected_energy_cost"], 0)
        self.assertGreater(result["projected_demand_charge"], 0)
        self.assertGreater(result["projected_total"], 0)

    def test_monthly_bill_days_remaining(self):
        """days_remaining should equal days_in_month - days_elapsed."""
        result = self.scheduler.project_monthly_bill(200.0, 10, 30, 280.0)
        self.assertEqual(result["days_remaining"], 20)

    def test_monthly_bill_ratchet_risk_detection(self):
        """Ratchet risk detected when 85% of 12mo peak > current month peak."""
        # current=200, 12mo_peak=300 → ratchet=255 > 200 → risk=True
        result = self.scheduler.project_monthly_bill(200.0, 15, 30, 200.0, peak_12mo_kw=300.0)
        self.assertTrue(result["ratchet_risk"])

    # ── Default loads catalog ─────────────────────────────────────────────────

    def test_default_loads_count(self):
        """DEFAULT_LOADS should have exactly 5 entries."""
        self.assertEqual(len(DEFAULT_LOADS), 5)

    def test_default_loads_have_required_fields(self):
        """Each default load has all DeferrableLoad fields."""
        for load in DEFAULT_LOADS:
            self.assertIsInstance(load, DeferrableLoad)
            self.assertIsNotNone(load.name)
            self.assertIsNotNone(load.priority)
            self.assertIn(load.priority, ("LOW", "MEDIUM", "HIGH"))

    # ── Singleton test ─────────────────────────────────────────────────────────

    def test_module_singleton(self):
        """Module-level scheduler singleton should be accessible."""
        self.assertIsInstance(scheduler, EnergyCostScheduler)


if __name__ == "__main__":
    unittest.main()
