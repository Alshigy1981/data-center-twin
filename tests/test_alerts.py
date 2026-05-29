"""
Unit tests for backend/alerts.py

Tests operate on plain numeric inputs — no database or API required.
"""

import pytest

from backend.alerts import (
    calculate_confidence_score,
    detect_pdu_overload,
    detect_phase_imbalance,
    detect_power_spike,
    detect_pue_trend,
    detect_rack_overload,
    detect_ups_high_load,
)


# ─────────────────────────────────────────
# Rack overload
# ─────────────────────────────────────────

class TestRackOverload:
    def test_rack_overload_warning_fires(self):
        result = detect_rack_overload("A-R1-001", power_kw=18.5)
        assert result is not None
        assert result["severity"] == "warning"
        assert result["asset_id"] == "A-R1-001"
        assert result["current_value"] == pytest.approx(18.5, rel=1e-3)

    def test_rack_overload_critical_fires(self):
        result = detect_rack_overload("A-R1-001", power_kw=20.0)
        assert result is not None
        assert result["severity"] == "critical"

    def test_rack_nominal_no_alert(self):
        result = detect_rack_overload("A-R1-001", power_kw=15.0)
        assert result is None

    def test_exactly_at_warning_threshold_fires(self):
        result = detect_rack_overload("A-R1-001", power_kw=18.0)
        assert result is not None
        assert result["severity"] == "warning"

    def test_below_threshold_no_alert(self):
        result = detect_rack_overload("A-R1-001", power_kw=17.9)
        assert result is None

    def test_worsening_increases_confidence(self):
        """Confidence should be higher when power is still rising."""
        result_static = detect_rack_overload("A-R1-001", 19.0, prev_power_kw=19.0)
        result_rising = detect_rack_overload("A-R1-001", 19.0, prev_power_kw=17.0)
        assert result_rising is not None
        assert result_static is not None
        assert result_rising["confidence"] >= result_static["confidence"]

    def test_duration_increases_confidence(self):
        r1 = detect_rack_overload("A-R1-001", 19.0, duration_cycles=1)
        r5 = detect_rack_overload("A-R1-001", 19.0, duration_cycles=5)
        assert r5 is not None
        assert r1 is not None
        assert r5["confidence"] >= r1["confidence"]


# ─────────────────────────────────────────
# PDU overload
# ─────────────────────────────────────────

class TestPDUOverload:
    def test_pdu_warning_fires_at_80_pct(self):
        result = detect_pdu_overload("A-PDU-1", load_kw=82.0, rated_kw=100.0)
        assert result is not None
        assert result["severity"] == "warning"

    def test_pdu_critical_fires_at_90_pct(self):
        result = detect_pdu_overload("A-PDU-1", load_kw=92.0, rated_kw=100.0)
        assert result is not None
        assert result["severity"] == "critical"

    def test_pdu_nominal_no_alert(self):
        result = detect_pdu_overload("A-PDU-1", load_kw=70.0, rated_kw=100.0)
        assert result is None

    def test_pdu_exactly_at_critical(self):
        result = detect_pdu_overload("A-PDU-1", load_kw=90.0, rated_kw=100.0)
        assert result is not None
        assert result["severity"] == "critical"

    def test_pdu_returns_asset_id(self):
        result = detect_pdu_overload("B-PDU-3", load_kw=85.0, rated_kw=100.0)
        assert result is not None
        assert result["asset_id"] == "B-PDU-3"


# ─────────────────────────────────────────
# UPS high load
# ─────────────────────────────────────────

class TestUPSHighLoad:
    def test_ups_warning_at_80_pct(self):
        result = detect_ups_high_load("UPS-1", load_pct=85.0)
        assert result is not None
        assert result["severity"] == "warning"

    def test_ups_critical_at_90_pct(self):
        result = detect_ups_high_load("UPS-1", load_pct=93.0)
        assert result is not None
        assert result["severity"] == "critical"

    def test_ups_nominal_no_alert(self):
        result = detect_ups_high_load("UPS-1", load_pct=75.0)
        assert result is None

    def test_ups_returns_asset_id(self):
        result = detect_ups_high_load("UPS-2", load_pct=88.0)
        assert result is not None
        assert result["asset_id"] == "UPS-2"


# ─────────────────────────────────────────
# Power spike detection
# ─────────────────────────────────────────

class TestPowerSpikeDetection:
    def test_spike_fires_above_threshold(self):
        """Power jumps from 10 kW to 12 kW = +20% — above the 15% threshold."""
        result = detect_power_spike("A-R1-001", current_kw=12.0, prev_kw=10.0)
        assert result is not None
        assert result["severity"] == "warning"

    def test_spike_does_not_fire_below_threshold(self):
        """Power jumps from 10 kW to 11 kW = +10% — below the 15% threshold."""
        result = detect_power_spike("A-R1-001", current_kw=11.0, prev_kw=10.0)
        assert result is None

    def test_spike_does_not_fire_with_zero_prev(self):
        """Zero previous power must not cause division-by-zero."""
        result = detect_power_spike("A-R1-001", current_kw=10.0, prev_kw=0)
        assert result is None

    def test_spike_message_contains_percentage(self):
        result = detect_power_spike("A-R2-005", current_kw=15.0, prev_kw=10.0)
        assert result is not None
        assert "%" in result["message"]
        assert "50.0" in result["message"]  # 50% spike

    def test_exactly_at_threshold_no_alert(self):
        """Exactly at the spike_pct threshold should not fire."""
        result = detect_power_spike("A-R1-001", current_kw=11.5, prev_kw=10.0, spike_pct=15.0)
        assert result is None  # 15% exactly is not > 15%


# ─────────────────────────────────────────
# PUE trend alert
# ─────────────────────────────────────────

class TestPUETrendAlert:
    def test_fires_after_consecutive_cycles(self):
        result = detect_pue_trend(
            pue_value=1.9,
            consecutive_high=4,
            threshold=1.8,
            required_consecutive=3,
        )
        assert result is not None
        assert result["asset_id"] == "facility"
        assert result["severity"] == "warning"

    def test_does_not_fire_before_consecutive_threshold(self):
        result = detect_pue_trend(
            pue_value=1.9,
            consecutive_high=2,
            threshold=1.8,
            required_consecutive=3,
        )
        assert result is None

    def test_does_not_fire_below_pue_threshold(self):
        result = detect_pue_trend(
            pue_value=1.5,
            consecutive_high=10,
            threshold=1.8,
            required_consecutive=3,
        )
        assert result is None


# ─────────────────────────────────────────
# Confidence score range
# ─────────────────────────────────────────

class TestConfidenceScoreRange:
    @pytest.mark.parametrize(
        "deviation,threshold,duration,worsening",
        [
            (0, 10, 1, False),
            (1, 10, 1, False),
            (10, 10, 1, True),
            (50, 10, 10, True),
            (100, 0.001, 20, True),
            (0.001, 10000, 1, False),
        ],
    )
    def test_score_always_in_range(self, deviation, threshold, duration, worsening):
        score = calculate_confidence_score(deviation, threshold, duration, worsening)
        assert 0 <= score <= 100, f"Score {score} out of [0,100] for inputs {(deviation, threshold, duration, worsening)}"

    def test_higher_duration_gives_higher_or_equal_score(self):
        s1 = calculate_confidence_score(5, 10, duration_cycles=1)
        s5 = calculate_confidence_score(5, 10, duration_cycles=5)
        assert s5 >= s1

    def test_worsening_increases_score(self):
        s_not_worsening = calculate_confidence_score(5, 10, is_worsening=False)
        s_worsening = calculate_confidence_score(5, 10, is_worsening=True)
        assert s_worsening >= s_not_worsening

    def test_zero_deviation_gives_low_base_score(self):
        score = calculate_confidence_score(0, 10, duration_cycles=1, is_worsening=False)
        assert score == 0

    def test_large_deviation_high_score(self):
        # Max score = 60 (base) + 30 (duration cap, needs ≥7 cycles) + 10 (worsening) = 100
        score = calculate_confidence_score(100, 10, duration_cycles=7, is_worsening=True)
        assert score == 100


# ─────────────────────────────────────────
# Phase imbalance
# ─────────────────────────────────────────

class TestPhaseImbalance:
    def test_fires_on_imbalanced_phases(self):
        """Phase A carries 60% of load while others carry 20% each → >10% imbalance."""
        result = detect_phase_imbalance("A-PDU-1", phase_a_kw=30.0, phase_b_kw=10.0, phase_c_kw=10.0)
        assert result is not None

    def test_no_alert_on_balanced_phases(self):
        result = detect_phase_imbalance("A-PDU-1", phase_a_kw=20.0, phase_b_kw=20.0, phase_c_kw=20.0)
        assert result is None

    def test_zero_load_no_crash(self):
        result = detect_phase_imbalance("A-PDU-1", 0, 0, 0)
        assert result is None
