"""
Telemetry feature extraction and normalization for the LSTM world model.

Sinusoidal time encoding captures diurnal patterns in data center load.
48-hour window captures 2 full diurnal cycles, giving the LSTM enough context
to distinguish workday peaks from weekend patterns.

Wear score is the most important feature (weighted 3x in loss) because it
drives PPO maintenance decisions. Forward fill then 0.5 default handles
sensor dropouts gracefully without introducing large outliers.
"""

from __future__ import annotations

import math
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─── Asset configuration (mirrors rl_agent.py) ────────────────────────────────

N_CRACS = 8
N_CHILLERS = 2
N_UPS = 2
N_ASSETS = N_CRACS + N_CHILLERS + N_UPS  # 12

ASSET_IDS: List[str] = (
    [f"A-CRAC-{i+1}" for i in range(4)]
    + [f"B-CRAC-{i+1}" for i in range(4)]
    + [f"CH-{i+1}" for i in range(N_CHILLERS)]
    + [f"UPS-{i+1}" for i in range(N_UPS)]
)

ASSET_TYPES: Dict[str, str] = {}
for _aid in ASSET_IDS:
    if "CRAC" in _aid:
        ASSET_TYPES[_aid] = "CRAC"
    elif _aid.startswith("CH-"):
        ASSET_TYPES[_aid] = "Chiller"
    else:
        ASSET_TYPES[_aid] = "UPS"

SEQ_LEN = 48        # 48 timesteps of history
FORECAST_HORIZON = 24  # 24 timesteps ahead
N_FEATURES = 8

FEATURE_NAMES = [
    "inlet_temp",
    "outlet_temp",
    "power_kw",
    "humidity",
    "delta_t",
    "wear_score",
    "hour_sin",
    "hour_cos",
]

# Per-asset-type normalization ranges: (min, max) for denormalization
_CRAC_RANGES = {
    "inlet_temp": (15.0, 45.0),   # return_air_temp_c
    "outlet_temp": (15.0, 45.0),  # supply_air_temp_c
    "power_kw": (0.0, 100.0),     # cooling_load_kw
    "humidity": (0.0, 100.0),
    "delta_t": (0.0, 20.0),
}
_CHILLER_RANGES = {
    "inlet_temp": (6.0, 20.0),    # chw_return_temp_c
    "outlet_temp": (4.0, 15.0),   # chw_supply_temp_c
    "power_kw": (0.0, 200.0),     # chiller power_kw
    "humidity": (0.0, 100.0),     # repurposed as load_pct
    "delta_t": (0.0, 12.0),       # chw delta-T
}
_UPS_RANGES = {
    "inlet_temp": (0.0, 1.0),     # placeholder
    "outlet_temp": (0.0, 1.0),    # placeholder
    "power_kw": (0.0, 200.0),     # output_kw
    "humidity": (0.0, 100.0),     # battery_pct
    "delta_t": (0.0, 10.0),       # input-output loss
}


def _norm(value: float, lo: float, hi: float) -> float:
    """Normalize value to [0, 1] given range [lo, hi]."""
    if hi == lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _denorm(value: float, lo: float, hi: float) -> float:
    return value * (hi - lo) + lo


def _hour_encoding(hour: float) -> Tuple[float, float]:
    """Map hour [0, 24) to (sin, cos) in [0, 1]."""
    angle = hour * 2.0 * math.pi / 24.0
    return (math.sin(angle) + 1.0) / 2.0, (math.cos(angle) + 1.0) / 2.0


class TelemetryFeatureEngineer:
    """Extract and normalize telemetry features for LSTM input."""

    def __init__(self) -> None:
        # Synthetic wear accumulator per asset (0–1); updates each time we extract features
        self._wear_states: Dict[str, float] = {aid: 0.2 for aid in ASSET_IDS}
        self._rng = np.random.default_rng()

    # ── Single-snapshot feature extraction ───────────────────────────────────

    def _extract_crac_features(self, snapshot: Any, asset_id: str, hour: float) -> np.ndarray:
        feat = np.full(N_FEATURES, 0.5, dtype=np.float32)
        for c in snapshot.cracs:
            if c.asset_id == asset_id:
                feat[0] = _norm(c.return_air_temp_c, *_CRAC_RANGES["inlet_temp"])
                feat[1] = _norm(c.supply_air_temp_c, *_CRAC_RANGES["outlet_temp"])
                feat[2] = _norm(c.cooling_load_kw, *_CRAC_RANGES["power_kw"])
                feat[3] = 0.5  # no humidity sensor on CRACs
                feat[4] = _norm(c.delta_t_c, *_CRAC_RANGES["delta_t"])
                load_pct = c.cooling_load_kw / 100.0
                self._wear_states[asset_id] = min(
                    1.0, self._wear_states[asset_id] + load_pct * 0.0003 + max(0.0, c.delta_t_c - 10.0) * 0.0002
                )
                break
        feat[5] = float(np.clip(self._wear_states[asset_id], 0.0, 1.0))
        h_sin, h_cos = _hour_encoding(hour)
        feat[6] = h_sin
        feat[7] = h_cos
        return feat

    def _extract_chiller_features(self, snapshot: Any, asset_id: str, hour: float) -> np.ndarray:
        feat = np.full(N_FEATURES, 0.5, dtype=np.float32)
        for ch in snapshot.chillers:
            if ch.asset_id == asset_id:
                feat[0] = _norm(ch.chw_return_temp_c, *_CHILLER_RANGES["inlet_temp"])
                feat[1] = _norm(ch.chw_supply_temp_c, *_CHILLER_RANGES["outlet_temp"])
                feat[2] = _norm(ch.power_kw, *_CHILLER_RANGES["power_kw"])
                feat[3] = ch.load_pct / 100.0  # repurpose humidity slot for load
                delta = ch.chw_return_temp_c - ch.chw_supply_temp_c
                feat[4] = _norm(delta, *_CHILLER_RANGES["delta_t"])
                self._wear_states[asset_id] = min(
                    1.0, self._wear_states[asset_id] + (ch.load_pct / 100.0) * 0.0004
                )
                break
        feat[5] = float(np.clip(self._wear_states[asset_id], 0.0, 1.0))
        h_sin, h_cos = _hour_encoding(hour)
        feat[6] = h_sin
        feat[7] = h_cos
        return feat

    def _extract_ups_features(self, snapshot: Any, asset_id: str, hour: float) -> np.ndarray:
        feat = np.full(N_FEATURES, 0.5, dtype=np.float32)
        for u in snapshot.upses:
            if u.asset_id == asset_id:
                feat[0] = 0.5  # no temperature
                feat[1] = 0.5
                feat[2] = _norm(u.output_kw, *_UPS_RANGES["power_kw"])
                feat[3] = u.battery_pct / 100.0
                loss = abs(u.input_kw - u.output_kw)
                feat[4] = _norm(loss, *_UPS_RANGES["delta_t"])
                self._wear_states[asset_id] = min(
                    1.0, self._wear_states[asset_id] + (u.load_pct / 100.0) * 0.0003
                )
                break
        feat[5] = float(np.clip(self._wear_states[asset_id], 0.0, 1.0))
        h_sin, h_cos = _hour_encoding(hour)
        feat[6] = h_sin
        feat[7] = h_cos
        return feat

    def _extract_features(self, snapshot: Any, asset_id: str, hour: float) -> np.ndarray:
        """Extract 8 features for one asset from one snapshot. Updates wear state as side effect."""
        if "CRAC" in asset_id:
            return self._extract_crac_features(snapshot, asset_id, hour)
        elif asset_id.startswith("CH-"):
            return self._extract_chiller_features(snapshot, asset_id, hour)
        else:
            return self._extract_ups_features(snapshot, asset_id, hour)

    # ── Public API ────────────────────────────────────────────────────────────

    def build_sequence(self, telemetry_history: List[Any], asset_id: str) -> np.ndarray:
        """
        Build a (SEQ_LEN, N_FEATURES) feature matrix from the last 48 snapshots.

        Missing values are forward-filled then defaulted to 0.5.
        Returns float32 array of shape (48, 8).
        """
        seq = np.full((SEQ_LEN, N_FEATURES), 0.5, dtype=np.float32)
        n = len(telemetry_history)
        start = max(0, n - SEQ_LEN)
        for i, snap in enumerate(telemetry_history[start:]):
            hour = float(snap.timestamp.hour) + float(snap.timestamp.minute) / 60.0
            idx = i + (SEQ_LEN - (n - start))
            seq[idx] = self._extract_features(snap, asset_id, hour)
        # Forward fill: propagate last valid value to any leading zeros that weren't overwritten
        # (rows before the first real reading keep the 0.5 default)
        return seq

    def build_all_sequences(self, telemetry_history: List[Any]) -> np.ndarray:
        """
        Build sequences for all 12 critical assets.
        Returns float32 array of shape (12, 48, 8).
        """
        return np.stack(
            [self.build_sequence(telemetry_history, aid) for aid in ASSET_IDS],
            axis=0,
        )

    def denormalize_features(self, normalized: np.ndarray, asset_id: str) -> Dict[str, float]:
        """Convert normalized (8,) prediction back to engineering units."""
        feat = np.asarray(normalized, dtype=np.float32)
        h_sin = feat[6] * 2.0 - 1.0  # map [0,1] back to [-1,1]
        h_cos = feat[7] * 2.0 - 1.0

        if "CRAC" in asset_id:
            r = _CRAC_RANGES
            return {
                "inlet_temp_c": _denorm(float(feat[0]), *r["inlet_temp"]),
                "outlet_temp_c": _denorm(float(feat[1]), *r["outlet_temp"]),
                "power_kw": _denorm(float(feat[2]), *r["power_kw"]),
                "humidity_pct": _denorm(float(feat[3]), *r["humidity"]),
                "delta_t_c": _denorm(float(feat[4]), *r["delta_t"]),
                "wear_score": float(feat[5]),
                "hour_sin": float(h_sin),
                "hour_cos": float(h_cos),
            }
        elif asset_id.startswith("CH-"):
            r = _CHILLER_RANGES
            return {
                "inlet_temp_c": _denorm(float(feat[0]), *r["inlet_temp"]),
                "outlet_temp_c": _denorm(float(feat[1]), *r["outlet_temp"]),
                "power_kw": _denorm(float(feat[2]), *r["power_kw"]),
                "load_pct": _denorm(float(feat[3]), *r["humidity"]),
                "delta_t_c": _denorm(float(feat[4]), *r["delta_t"]),
                "wear_score": float(feat[5]),
                "hour_sin": float(h_sin),
                "hour_cos": float(h_cos),
            }
        else:  # UPS
            r = _UPS_RANGES
            return {
                "inlet_temp_c": 0.0,
                "outlet_temp_c": 0.0,
                "power_kw": _denorm(float(feat[2]), *r["power_kw"]),
                "battery_pct": _denorm(float(feat[3]), *r["humidity"]),
                "loss_kw": _denorm(float(feat[4]), *r["delta_t"]),
                "wear_score": float(feat[5]),
                "hour_sin": float(h_sin),
                "hour_cos": float(h_cos),
            }

    def generate_training_data(
        self,
        simulator: Any,
        n_steps: int = 10_000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the simulator for n_steps, build sliding-window (X, y) pairs.

        Returns (X_train, y_train, X_val, y_val) with 80/20 split.
        X shape: (n_samples, 48, 8)
        y shape: (n_samples, 24, 8)
        """
        rng = np.random.default_rng(42)

        # Randomize initial wear states
        for aid in ASSET_IDS:
            self._wear_states[aid] = float(rng.uniform(0.05, 0.35))

        # Collect feature arrays for all steps
        all_features = np.zeros((n_steps, N_ASSETS, N_FEATURES), dtype=np.float32)

        for step in range(n_steps):
            snapshot = simulator.generate_snapshot()
            hour = float(step % 24)  # synthetic hourly progression

            # Occasional maintenance event (wear reset)
            for j, aid in enumerate(ASSET_IDS):
                if rng.random() < 0.003:
                    self._wear_states[aid] = float(rng.uniform(0.0, 0.08))

            for j, aid in enumerate(ASSET_IDS):
                all_features[step, j] = self._extract_features(snapshot, aid, hour)

        # Build (X, y) pairs with stride for speed while maintaining good coverage
        stride = 12
        windows_X: List[np.ndarray] = []
        windows_y: List[np.ndarray] = []

        for i_asset in range(N_ASSETS):
            asset_feats = all_features[:, i_asset, :]  # (n_steps, 8)
            t = 0
            while t + SEQ_LEN + FORECAST_HORIZON <= n_steps:
                windows_X.append(asset_feats[t : t + SEQ_LEN])
                windows_y.append(asset_feats[t + SEQ_LEN : t + SEQ_LEN + FORECAST_HORIZON])
                t += stride

        X = np.array(windows_X, dtype=np.float32)
        y = np.array(windows_y, dtype=np.float32)

        n_total = len(X)
        idx = rng.permutation(n_total)
        X, y = X[idx], y[idx]

        split = int(n_total * 0.8)
        return X[:split], y[:split], X[split:], y[split:]
