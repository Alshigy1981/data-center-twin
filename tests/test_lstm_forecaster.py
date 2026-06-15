"""Tests for TelemetryFeatureEngineer and LSTMForecaster."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.lstm.feature_engineer import (
    ASSET_IDS,
    N_ASSETS,
    N_FEATURES,
    SEQ_LEN,
    FORECAST_HORIZON,
    TelemetryFeatureEngineer,
)
from backend.lstm.forecaster import LSTMForecaster


# ─── Helpers to build mock TelemetrySnapshots ─────────────────────────────────

def _mock_crac(asset_id: str) -> Any:
    c = MagicMock()
    c.asset_id = asset_id
    c.return_air_temp_c = 28.0
    c.supply_air_temp_c = 18.0
    c.delta_t_c = 10.0
    c.cooling_load_kw = 60.0
    c.setpoint_c = 18.0
    c.room = "Room A" if "A-CRAC" in asset_id else "Room B"
    return c


def _mock_chiller(asset_id: str) -> Any:
    ch = MagicMock()
    ch.asset_id = asset_id
    ch.chw_return_temp_c = 12.0
    ch.chw_supply_temp_c = 8.0
    ch.power_kw = 80.0
    ch.load_pct = 60.0
    ch.cop = 4.0
    return ch


def _mock_ups(asset_id: str) -> Any:
    u = MagicMock()
    u.asset_id = asset_id
    u.output_kw = 120.0
    u.input_kw = 124.0
    u.battery_pct = 98.0
    u.load_pct = 60.0
    return u


def _make_snapshot(ts: datetime = None) -> Any:
    snap = MagicMock()
    snap.timestamp = ts or datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
    snap.cracs = [_mock_crac(aid) for aid in ASSET_IDS if "CRAC" in aid]
    snap.chillers = [_mock_chiller(aid) for aid in ASSET_IDS if aid.startswith("CH-")]
    snap.upses = [_mock_ups(aid) for aid in ASSET_IDS if aid.startswith("UPS-")]
    return snap


def _full_history(n: int = SEQ_LEN) -> list:
    base = datetime(2024, 6, 14, 0, 0, tzinfo=timezone.utc)
    return [
        _make_snapshot(datetime(2024, 6, 14, i % 24, 0, tzinfo=timezone.utc))
        for i in range(n)
    ]


# ─── Feature engineer tests ───────────────────────────────────────────────────

class TestTelemetryFeatureEngineer:

    def test_feature_engineer_sequence_shape(self):
        fe = TelemetryFeatureEngineer()
        history = _full_history(SEQ_LEN)
        seq = fe.build_sequence(history, "A-CRAC-1")
        assert seq.shape == (SEQ_LEN, N_FEATURES), (
            f"Expected ({SEQ_LEN}, {N_FEATURES}), got {seq.shape}"
        )

    def test_feature_engineer_all_sequences_shape(self):
        fe = TelemetryFeatureEngineer()
        history = _full_history(SEQ_LEN)
        all_seq = fe.build_all_sequences(history)
        assert all_seq.shape == (N_ASSETS, SEQ_LEN, N_FEATURES), (
            f"Expected ({N_ASSETS}, {SEQ_LEN}, {N_FEATURES}), got {all_seq.shape}"
        )

    def test_normalization_range(self):
        fe = TelemetryFeatureEngineer()
        history = _full_history(SEQ_LEN)
        for aid in ASSET_IDS:
            seq = fe.build_sequence(history, aid)
            assert seq.min() >= 0.0 - 1e-6, (
                f"Asset {aid}: min normalized value {seq.min()} < 0"
            )
            assert seq.max() <= 1.0 + 1e-6, (
                f"Asset {aid}: max normalized value {seq.max()} > 1"
            )

    def test_denormalization_roundtrip(self):
        fe = TelemetryFeatureEngineer()
        # Known CRAC values → normalize manually → denormalize → check roundtrip
        # return_air_temp_c=28.0 → (28-15)/30 = 0.4333
        # cooling_load_kw=60.0 → 60/100 = 0.6
        normalized = np.zeros(N_FEATURES, dtype=np.float32)
        normalized[0] = (28.0 - 15.0) / 30.0   # inlet_temp
        normalized[1] = (18.0 - 15.0) / 30.0   # outlet_temp
        normalized[2] = 60.0 / 100.0            # power_kw
        normalized[3] = 0.5                      # humidity
        normalized[4] = 10.0 / 20.0             # delta_t
        normalized[5] = 0.2                      # wear_score
        normalized[6] = (math.sin(14 * 2 * math.pi / 24) + 1) / 2
        normalized[7] = (math.cos(14 * 2 * math.pi / 24) + 1) / 2

        result = fe.denormalize_features(normalized, "A-CRAC-1")
        assert abs(result["inlet_temp_c"] - 28.0) < 0.5, (
            f"inlet_temp roundtrip failed: {result['inlet_temp_c']}"
        )
        assert abs(result["power_kw"] - 60.0) < 1.0, (
            f"power_kw roundtrip failed: {result['power_kw']}"
        )
        assert abs(result["delta_t_c"] - 10.0) < 0.5, (
            f"delta_t roundtrip failed: {result['delta_t_c']}"
        )
        assert abs(result["wear_score"] - 0.2) < 0.01


# ─── Forecaster tests ─────────────────────────────────────────────────────────

class TestLSTMForecaster:

    def test_forecaster_returns_24h_predictions(self):
        forecaster = LSTMForecaster(model_path=None)  # untrained model
        history = _full_history(SEQ_LEN)
        for snap in history:
            forecaster.update_history(snap)

        result = forecaster.forecast()
        for aid in ASSET_IDS:
            preds = result["forecasts"][aid]
            assert len(preds) == FORECAST_HORIZON, (
                f"Expected 24 predictions for {aid}, got {len(preds)}"
            )
            assert preds.shape[-1] == N_FEATURES

    def test_forecaster_fallback_when_no_history(self):
        forecaster = LSTMForecaster(model_path=None)
        # Only 10 readings — less than the required 48
        history = _full_history(10)
        for snap in history:
            forecaster.update_history(snap)

        assert not forecaster.is_healthy(), "Should not be healthy with < 48 readings"

        # Should not raise; returns empty forecast gracefully
        result = forecaster.forecast()
        assert "forecasts" in result
        assert "timestamp" in result
        for aid in ASSET_IDS:
            assert aid in result["forecasts"]

    def test_risk_timeline_sorted_by_urgency(self):
        forecaster = LSTMForecaster(model_path=None)
        history = _full_history(SEQ_LEN)
        for snap in history:
            forecaster.update_history(snap)

        # Manually inject high wear into some assets' wear states
        forecaster.feature_engineer._wear_states["A-CRAC-1"] = 0.95
        forecaster.feature_engineer._wear_states["B-CRAC-1"] = 0.92

        timeline = forecaster.get_risk_timeline()
        assert len(timeline) == N_ASSETS

        # Timeline should be sorted: CRITICAL/HIGH before LOW
        risk_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        for i in range(len(timeline) - 1):
            a = risk_order[timeline[i]["risk_level"]]
            b = risk_order[timeline[i + 1]["risk_level"]]
            assert a <= b, (
                f"Timeline not sorted: {timeline[i]['risk_level']} before {timeline[i+1]['risk_level']}"
            )

    def test_forecast_confidence_range(self):
        forecaster = LSTMForecaster(model_path=None)
        history = _full_history(SEQ_LEN)
        for snap in history:
            forecaster.update_history(snap)

        result = forecaster.forecast()
        for aid in ASSET_IDS:
            conf = result["confidence"][aid]
            assert 0.0 <= conf <= 1.0, (
                f"Confidence for {aid} out of range: {conf}"
            )
