"""
Unit tests for backend/metrics.py

Tests operate on plain numeric inputs — no database or API required.
"""

import pytest

from backend.metrics import (
    BindingConstraint,
    calculate_carbon,
    calculate_pue,
    calculate_rack_temp_rise,
    calculate_stranded_capacity,
    chiller_cop_estimate,
    forecast_pue,
    pue_forecast_series,
)


# ─────────────────────────────────────────
# PUE calculation
# ─────────────────────────────────────────

class TestPUECalculation:
    def test_normal_case(self):
        """Typical data center: 500 kW IT, 250 kW cooling → PUE ~1.575."""
        pue = calculate_pue(it_load_kw=500, cooling_load_kw=250)
        # Total = 500 + 250 + 25 (5% overhead) = 775; PUE = 775/500 = 1.55
        assert 1.0 < pue < 3.0, f"PUE {pue} outside expected range"
        expected = 775 / 500  # 1.55
        assert abs(pue - expected) < 0.001

    def test_pue_never_below_one(self):
        """PUE must be ≥ 1.0 by definition."""
        pue = calculate_pue(it_load_kw=1000, cooling_load_kw=0, overhead_factor=0)
        assert pue >= 1.0

    def test_zero_it_load_guard(self):
        """Division by zero must be handled gracefully."""
        pue = calculate_pue(it_load_kw=0, cooling_load_kw=100)
        assert pue == 1.0, "Should return 1.0 sentinel when IT load is 0"

    def test_negative_it_load_guard(self):
        """Negative IT load should be treated as zero."""
        pue = calculate_pue(it_load_kw=-10, cooling_load_kw=50)
        assert pue == 1.0

    def test_high_efficiency_pue(self):
        """World-class data center: large IT, tiny cooling load."""
        pue = calculate_pue(it_load_kw=1000, cooling_load_kw=100)
        # Total = 1000 + 100 + 50 = 1150; PUE = 1.15
        assert 1.1 < pue < 1.3, f"High-efficiency PUE {pue} unexpected"

    def test_overhead_factor_effect(self):
        """Higher overhead factor must increase PUE."""
        pue_low = calculate_pue(500, 200, overhead_factor=0.03)
        pue_high = calculate_pue(500, 200, overhead_factor=0.10)
        assert pue_high > pue_low

    def test_pue_result_is_rounded(self):
        """PUE should be returned to 4 decimal places."""
        pue = calculate_pue(500, 300)
        assert pue == round(pue, 4)


# ─────────────────────────────────────────
# Stranded capacity
# ─────────────────────────────────────────

class TestStrandedCapacity:
    def test_cooling_limited(self):
        """When cooling headroom < power headroom, constraint is cooling-limited."""
        result = calculate_stranded_capacity(
            current_it_load_kw=400,
            current_cooling_load_kw=750,   # only 50 kW headroom in 800 kW total CRAC
            total_crac_rated_kw=800,
            total_pdu_rated_kw=1000,
            total_ups_rated_kw=2000,
        )
        assert result["binding_constraint"] == BindingConstraint.COOLING_LIMITED
        assert result["cooling_headroom_kw"] < result["power_headroom_kw"]
        assert result["stranded_capacity_kw"] == result["cooling_headroom_kw"]

    def test_power_limited(self):
        """When power headroom < cooling headroom, constraint is power-limited."""
        result = calculate_stranded_capacity(
            current_it_load_kw=750,   # only 50 kW PDU headroom in 800 kW total
            current_cooling_load_kw=200,
            total_crac_rated_kw=800,
            total_pdu_rated_kw=800,
            total_ups_rated_kw=2000,
        )
        assert result["binding_constraint"] == BindingConstraint.POWER_LIMITED
        assert result["stranded_capacity_kw"] == result["power_headroom_kw"]

    def test_stranded_never_negative(self):
        """Stranded capacity should be clipped at 0 even when over limit."""
        result = calculate_stranded_capacity(
            current_it_load_kw=900,
            current_cooling_load_kw=850,
            total_crac_rated_kw=800,   # over cooling limit
            total_pdu_rated_kw=800,    # over power limit
        )
        assert result["stranded_capacity_kw"] >= 0
        assert result["cooling_headroom_kw"] >= 0
        assert result["power_headroom_kw"] >= 0

    def test_balanced_constraint(self):
        """When both headrooms are equal, constraint should be balanced."""
        result = calculate_stranded_capacity(
            current_it_load_kw=500,
            current_cooling_load_kw=500,
            total_crac_rated_kw=800,   # cooling headroom = 300
            total_pdu_rated_kw=800,    # power headroom = 300
            total_ups_rated_kw=2000,   # ups headroom = 1500 (not binding)
        )
        assert result["binding_constraint"] == BindingConstraint.BALANCED
        assert result["stranded_capacity_kw"] == 300.0

    def test_returns_required_keys(self):
        result = calculate_stranded_capacity(400, 300)
        required_keys = {
            "cooling_headroom_kw", "power_headroom_kw",
            "stranded_capacity_kw", "binding_constraint",
            "total_crac_rated_kw", "total_pdu_rated_kw",
            "current_it_load_kw", "current_cooling_load_kw",
        }
        assert required_keys.issubset(result.keys())


# ─────────────────────────────────────────
# Carbon calculation
# ─────────────────────────────────────────

class TestCarbonCalculation:
    def test_basic_calculation(self):
        """1000 kW facility at UK grid average 0.233 kgCO₂/kWh."""
        result = calculate_carbon(total_facility_kw=1000, carbon_factor=0.233)
        assert abs(result["carbon_kg_per_hr"] - 233.0) < 0.01
        expected_annual = 233.0 * 8760 / 1000
        assert abs(result["carbon_tonnes_per_year"] - expected_annual) < 0.1

    def test_zero_power_gives_zero_carbon(self):
        result = calculate_carbon(total_facility_kw=0, carbon_factor=0.233)
        assert result["carbon_kg_per_hr"] == 0.0
        assert result["carbon_tonnes_per_year"] == 0.0

    def test_carbon_factor_respected(self):
        """Higher carbon factor must yield proportionally higher emissions."""
        r1 = calculate_carbon(500, 0.1)
        r2 = calculate_carbon(500, 0.3)
        assert abs(r2["carbon_kg_per_hr"] / r1["carbon_kg_per_hr"] - 3.0) < 0.01

    def test_returns_required_keys(self):
        result = calculate_carbon(800, 0.233)
        assert "carbon_kg_per_hr" in result
        assert "carbon_tonnes_per_year" in result
        assert "carbon_factor" in result

    def test_annualised_formula(self):
        """Verify annualised = hourly × 8760 / 1000."""
        result = calculate_carbon(600, 0.4)
        expected = result["carbon_kg_per_hr"] * 8760 / 1000
        assert abs(result["carbon_tonnes_per_year"] - expected) < 0.01


# ─────────────────────────────────────────
# Rack temperature rise
# ─────────────────────────────────────────

class TestRackTemperatureRise:
    def test_normal_rise(self):
        rise = calculate_rack_temp_rise(inlet_temp_c=22.0, outlet_temp_c=32.0)
        assert rise == 10.0

    def test_zero_rise(self):
        rise = calculate_rack_temp_rise(25.0, 25.0)
        assert rise == 0.0

    def test_negative_rise_not_physically_expected_but_handled(self):
        """Function should not crash on inverted input."""
        rise = calculate_rack_temp_rise(30.0, 25.0)
        assert rise == -5.0

    def test_result_is_rounded(self):
        rise = calculate_rack_temp_rise(22.123, 33.456)
        assert rise == round(33.456 - 22.123, 2)


# ─────────────────────────────────────────
# Chiller COP
# ─────────────────────────────────────────

class TestChillerCOP:
    def test_typical_cop(self):
        cop = chiller_cop_estimate(cooling_output_kw=400, electrical_input_kw=100)
        assert cop == 4.0

    def test_zero_input_returns_zero(self):
        cop = chiller_cop_estimate(400, 0)
        assert cop == 0.0

    def test_cop_is_positive(self):
        cop = chiller_cop_estimate(300, 80)
        assert cop > 0


# ─────────────────────────────────────────
# PUE forecast
# ─────────────────────────────────────────

class TestPUEForecast:
    def _make_history(self, values):
        return [{"pue": v, "timestamp": f"2024-01-01T{i:02d}:00:00"} for i, v in enumerate(values)]

    def test_forecast_returns_none_with_few_data(self):
        history = self._make_history([1.5, 1.6])
        result = forecast_pue(history)
        assert result is None

    def test_forecast_returns_float_with_sufficient_data(self):
        history = self._make_history([1.4, 1.45, 1.5, 1.48, 1.52, 1.55, 1.53, 1.57])
        result = forecast_pue(history, horizon_steps=10)
        assert result is not None
        assert isinstance(result, float)
        assert result >= 1.0

    def test_rising_trend_forecasts_higher(self):
        values = [1.3 + i * 0.05 for i in range(10)]  # clear upward trend
        history = self._make_history(values)
        result = forecast_pue(history, horizon_steps=5)
        assert result is not None
        assert result > values[-1], "Rising trend should forecast above current"

    def test_forecast_series_returns_three_lists(self):
        history = self._make_history([1.4, 1.42, 1.45, 1.43, 1.47, 1.5, 1.48, 1.52])
        forecasts, uppers, lowers = pue_forecast_series(history, horizon_steps=10)
        assert len(forecasts) == 10
        assert len(uppers) == 10
        assert len(lowers) == 10
        for f, u, l in zip(forecasts, uppers, lowers):
            assert u >= f >= l
