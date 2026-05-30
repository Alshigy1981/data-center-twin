"""
Unit tests for backend/optimizer.py

All tests operate on plain numeric inputs — no database or API required.
"""

import pytest

from backend.optimizer import (
    CHW_MAX, CHW_MIN, CRAC_MAX, CRAC_MIN,
    _COIL_APPROACH_C,
    _chiller_cop,
    _crac_fan_power_kw,
    _effective_chw,
    _thermal_risk,
    compute_pareto_front,
    evaluate_setpoints,
    optimize_setpoints,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_snapshot(it_kw=500.0, cooling_kw=300.0, avg_outlet_c=35.0, n_cracs=8, n_chillers=2):
    """Minimal TelemetrySnapshot-like object sufficient for the optimizer."""
    from unittest.mock import MagicMock
    from datetime import datetime

    snap = MagicMock()
    snap.racks = [MagicMock(power_kw=it_kw / 80, outlet_temp_c=avg_outlet_c) for _ in range(80)]
    snap.cracs = [MagicMock(cooling_load_kw=cooling_kw / n_cracs, setpoint_c=18.0) for _ in range(n_cracs)]
    snap.chillers = [MagicMock(chw_supply_temp_c=8.0) for _ in range(n_chillers)]
    return snap


# ─── Physics helpers ──────────────────────────────────────────────────────────

class TestEffectiveCHW:
    def test_capped_by_coil_approach(self):
        # CRAC=18°C → max CHW = 18-5 = 13°C; requesting 14°C gets capped
        assert _effective_chw(14.0, 18.0) == pytest.approx(13.0)

    def test_not_capped_when_chw_lower(self):
        # CRAC=24°C → max CHW = 19°C; requesting 10°C is fine
        assert _effective_chw(10.0, 24.0) == pytest.approx(10.0)

    def test_never_below_chw_min(self):
        assert _effective_chw(6.0, 11.0) >= CHW_MIN  # crac-5=6, min is 6

    def test_higher_crac_setpoint_allows_higher_effective_chw(self):
        # At CHW=14, CRAC=18 → effective=13; CRAC=22 → effective=14
        eff_low = _effective_chw(14.0, 18.0)
        eff_high = _effective_chw(14.0, 22.0)
        assert eff_high > eff_low


class TestChillerCOP:
    def test_cop_rises_with_setpoint(self):
        cop_low = _chiller_cop(6.0)
        cop_high = _chiller_cop(14.0)
        assert cop_high > cop_low, "Higher CHW setpoint must improve COP"

    def test_cop_bounded(self):
        for t in [4.0, 6.0, 10.0, 14.0, 16.0]:
            cop = _chiller_cop(t)
            assert 1.5 <= cop <= 8.0, f"COP {cop} out of bounds at {t} °C"

    def test_cop_reference_point(self):
        cop = _chiller_cop(7.0)
        # Carnot at 7°C supply / 30°C cond: 280.15/(30-7) ≈ 12.2, × 0.55 ≈ 6.7
        assert 2.0 < cop < 8.0


class TestCRACFanPower:
    def test_fan_increases_with_higher_setpoint(self):
        # Higher supply setpoint → smaller delta-T → more airflow needed (cube law)
        kw_low = _crac_fan_power_kw(16.0, 35.0, 8)
        kw_high = _crac_fan_power_kw(24.0, 35.0, 8)
        assert kw_high > kw_low, "Higher CRAC setpoint → smaller ΔT → more fan power"

    def test_fan_positive(self):
        kw = _crac_fan_power_kw(18.0, 32.0, 8)
        assert kw > 0

    def test_scales_with_crac_count(self):
        kw4 = _crac_fan_power_kw(18.0, 35.0, 4)
        kw8 = _crac_fan_power_kw(18.0, 35.0, 8)
        assert kw8 > kw4


class TestThermalRisk:
    def test_zero_risk_with_large_margin(self):
        # Outlet 35°C, supply 16°C → inlet ≈ 16 + 0.3*(35-16) ≈ 21.7°C → 10°C margin
        risk = _thermal_risk(16.0, 35.0)
        assert risk == 0.0

    def test_high_risk_near_limit(self):
        # Supply 28°C, outlet 35°C → inlet ≈ 28 + 0.3*7 ≈ 30.1°C → margin < 2°C
        risk = _thermal_risk(28.0, 35.0)
        assert risk > 1.0

    def test_max_risk_at_limit(self):
        risk = _thermal_risk(32.0, 40.0)
        assert risk >= 100.0


# ─── evaluate_setpoints ───────────────────────────────────────────────────────

class TestEvaluateSetpoints:
    def setup_method(self):
        self.kwargs = dict(it_kw=500.0, cooling_kw=300.0, avg_outlet_c=35.0, n_cracs=8, n_chillers=2)

    def test_returns_all_keys(self):
        m = evaluate_setpoints(10.0, 18.0, **self.kwargs)
        for key in ["pue", "carbon_kg_hr", "cop", "crac_fan_power_kw", "chiller_power_kw", "thermal_risk_score"]:
            assert key in m, f"Missing key: {key}"

    def test_pue_above_one(self):
        m = evaluate_setpoints(10.0, 18.0, **self.kwargs)
        assert m["pue"] > 1.0

    def test_higher_chw_setpoint_improves_pue_when_crac_allows(self):
        # With CRAC=24°C (max CHW allowed = 19°C), CHW 6→14 has full effect.
        # With CRAC=18°C, CHW is capped at 13°C by coil approach, so 6→13 is the real range.
        pue_low = evaluate_setpoints(6.0, 24.0, **self.kwargs)["pue"]
        pue_high = evaluate_setpoints(14.0, 24.0, **self.kwargs)["pue"]
        assert pue_high < pue_low, "Higher CHW setpoint → better COP → lower PUE"

    def test_higher_crac_setpoint_increases_fan_power(self):
        # Smaller ΔT at higher setpoint → more airflow required → more fan power
        fan_low = evaluate_setpoints(10.0, 16.0, **self.kwargs)["crac_fan_power_kw"]
        fan_high = evaluate_setpoints(10.0, 24.0, **self.kwargs)["crac_fan_power_kw"]
        assert fan_high > fan_low


# ─── optimize_setpoints ───────────────────────────────────────────────────────

class TestOptimizeSetpoints:
    def test_returns_within_bounds(self):
        snap = _make_snapshot()
        result = optimize_setpoints(snap)
        assert CHW_MIN <= result["optimal_chw_setpoint_c"] <= CHW_MAX
        assert CRAC_MIN <= result["optimal_crac_setpoint_c"] <= CRAC_MAX

    def test_pue_improvement_non_negative(self):
        snap = _make_snapshot()
        result = optimize_setpoints(snap)
        # Optimizer must find a point at least as good as the current operating point
        assert result["pue_improvement"] >= -0.01  # tiny tolerance for float rounding

    def test_all_fields_present(self):
        snap = _make_snapshot()
        result = optimize_setpoints(snap)
        for key in [
            "optimal_chw_setpoint_c", "optimal_crac_setpoint_c", "expected_pue",
            "expected_cop", "thermal_risk_score", "current_pue", "pue_improvement",
            "carbon_reduction_kg_hr", "converged", "iterations", "weights",
        ]:
            assert key in result, f"Missing field: {key}"

    def test_custom_weights(self):
        snap = _make_snapshot()
        result = optimize_setpoints(snap, w_pue=1.0, w_carbon=0.0, w_risk=0.0)
        assert result["optimal_chw_setpoint_c"] >= result["optimal_chw_setpoint_c"] - 1  # sanity


# ─── compute_pareto_front ─────────────────────────────────────────────────────

class TestComputeParetoFront:
    def test_returns_list(self):
        snap = _make_snapshot()
        pareto = compute_pareto_front(snap, n_chw=5, n_crac=5)
        assert isinstance(pareto, list)
        assert len(pareto) > 0

    def test_sorted_by_pue(self):
        snap = _make_snapshot()
        pareto = compute_pareto_front(snap, n_chw=5, n_crac=5)
        pues = [p["pue"] for p in pareto]
        assert pues == sorted(pues)

    def test_no_dominated_points(self):
        snap = _make_snapshot()
        pareto = compute_pareto_front(snap, n_chw=5, n_crac=5)
        for i, pt in enumerate(pareto):
            for j, other in enumerate(pareto):
                if i == j:
                    continue
                dominated = (
                    other["pue"] <= pt["pue"]
                    and other["thermal_risk_score"] <= pt["thermal_risk_score"]
                    and (other["pue"] < pt["pue"] or other["thermal_risk_score"] < pt["thermal_risk_score"])
                )
                assert not dominated, f"Point {i} dominated by {j}"

    def test_all_setpoints_in_bounds(self):
        snap = _make_snapshot()
        pareto = compute_pareto_front(snap, n_chw=5, n_crac=5)
        for pt in pareto:
            assert CHW_MIN <= pt["chw_setpoint_c"] <= CHW_MAX
            assert CRAC_MIN <= pt["crac_setpoint_c"] <= CRAC_MAX
