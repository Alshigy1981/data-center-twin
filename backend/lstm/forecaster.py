"""
LSTM inference engine — maintains a rolling telemetry history and produces
24-hour forecasts with attention weights and failure risk estimates.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend.lstm.feature_engineer import (
    ASSET_IDS,
    ASSET_TYPES,
    N_ASSETS,
    SEQ_LEN,
    TelemetryFeatureEngineer,
)

logger = logging.getLogger(__name__)

try:
    import torch
    from backend.lstm.model import DataCenterLSTM
    TORCH_AVAILABLE = True
except ImportError:
    DataCenterLSTM = None  # type: ignore[assignment,misc]
    TORCH_AVAILABLE = False

_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "lstm_world_model.pt"


def _risk_level(hours_to_failure: Optional[float]) -> str:
    if hours_to_failure is None:
        return "LOW"
    if hours_to_failure < 12:
        return "CRITICAL"
    if hours_to_failure < 48:
        return "HIGH"
    if hours_to_failure < 168:
        return "MEDIUM"
    return "LOW"


class LSTMForecaster:
    """
    Real-time LSTM inference engine for all 12 critical assets.

    Maintains a rolling 48-step history buffer. Produces 24-hour forecasts
    including attention weights (which past hours mattered most) and
    predicted failure timelines.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.feature_engineer = TelemetryFeatureEngineer()
        self.history: deque = deque(maxlen=SEQ_LEN)  # TelemetrySnapshot objects
        self._total_forecasts = 0
        self._last_forecast_time: Optional[datetime] = None
        self._forecast_cache: Dict[str, Any] = {}
        self._model_accuracy: float = 0.0

        self.model: Optional[Any] = None

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch unavailable — LSTMForecaster running in stub mode")
            return

        self.model = DataCenterLSTM()
        _path = Path(model_path) if model_path else _MODEL_PATH

        if _path.exists():
            try:
                state = torch.load(str(_path), map_location="cpu", weights_only=True)
                self.model.load_state_dict(state)
                logger.info("Loaded LSTM model from %s", _path)
            except Exception as exc:
                logger.warning("Could not load LSTM weights (%s) — using untrained model", exc)

        self.model.eval()

    # ── History management ────────────────────────────────────────────────────

    def update_history(self, telemetry_snapshot: Any) -> None:
        """Append a new TelemetrySnapshot to the rolling 48-step buffer."""
        self.history.append(telemetry_snapshot)

    # ── Core forecasting ──────────────────────────────────────────────────────

    def forecast(self, asset_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Produce 24-hour forecasts.

        Args:
            asset_id: If given, forecast for one asset; else forecast all 12.

        Returns dict with keys: forecasts, confidence, attention_weights,
        predicted_failure_hours, timestamp.
        """
        assets = [asset_id] if asset_id else ASSET_IDS

        if not self.is_healthy():
            return self._empty_forecast(assets)

        history_list = list(self.history)
        forecasts: Dict[str, np.ndarray] = {}
        confidence: Dict[str, float] = {}
        attention_weights: Dict[str, np.ndarray] = {}
        predicted_failure_hours: Dict[str, Optional[float]] = {}

        for aid in assets:
            try:
                seq = self.feature_engineer.build_sequence(history_list, aid)  # (48, 8)
                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)        # (1, 48, 8)

                with torch.no_grad():
                    preds, attn = self.model(x)

                preds_np = preds.squeeze(0).cpu().numpy()   # (24, 8)
                attn_np = attn.squeeze(0).cpu().numpy()     # (48,)
                preds_np = np.clip(preds_np, 0.0, 1.0)

                forecasts[aid] = preds_np
                confidence[aid] = self._compute_confidence(preds_np)
                attention_weights[aid] = attn_np
                predicted_failure_hours[aid] = self._predict_failure_hours(preds_np, seq)

            except Exception as exc:
                logger.warning("Forecast failed for %s: %s", aid, exc)
                forecasts[aid] = np.full((24, 8), 0.5, dtype=np.float32)
                confidence[aid] = 0.0
                attention_weights[aid] = np.ones(SEQ_LEN, dtype=np.float32) / SEQ_LEN
                predicted_failure_hours[aid] = None

        self._total_forecasts += 1
        self._last_forecast_time = datetime.utcnow()
        self._forecast_cache = {
            "forecasts": forecasts,
            "confidence": confidence,
            "attention_weights": attention_weights,
            "predicted_failure_hours": predicted_failure_hours,
            "timestamp": self._last_forecast_time,
        }
        return self._forecast_cache

    def _compute_confidence(self, preds: np.ndarray) -> float:
        """Confidence based on prediction trajectory smoothness (0–1)."""
        wear = preds[:, 5]
        if len(wear) < 2:
            return 0.5
        smoothness = 1.0 - min(1.0, float(np.std(np.diff(wear))) * 8.0)
        return float(np.clip(smoothness, 0.0, 1.0))

    def _predict_failure_hours(
        self, preds: np.ndarray, seq: np.ndarray
    ) -> Optional[float]:
        """Return estimated hours until predicted wear reaches 0.9."""
        wear_preds = preds[:, 5]
        current_wear = float(seq[-1, 5])

        for h, w in enumerate(wear_preds):
            if w >= 0.9:
                return float(h + 1)

        if len(wear_preds) > 1:
            wear_rate = (float(wear_preds[-1]) - current_wear) / len(wear_preds)
            if wear_rate > 1e-6:
                return float(min((0.9 - current_wear) / wear_rate, 10_000.0))

        return None

    def _empty_forecast(self, assets: List[str]) -> Dict[str, Any]:
        return {
            "forecasts": {aid: np.full((24, 8), 0.5, dtype=np.float32) for aid in assets},
            "confidence": {aid: 0.0 for aid in assets},
            "attention_weights": {
                aid: np.ones(SEQ_LEN, dtype=np.float32) / SEQ_LEN for aid in assets
            },
            "predicted_failure_hours": {aid: None for aid in assets},
            "timestamp": datetime.utcnow(),
        }

    # ── Embedding for PPO ─────────────────────────────────────────────────────

    def get_forecast_embedding(self) -> np.ndarray:
        """
        Compact LSTM context vector for all 12 assets concatenated.
        Shape: (12 * 128,) = (1536,)
        Used by PPO as additional observation features.
        """
        if not self.is_healthy():
            return np.zeros(N_ASSETS * 128, dtype=np.float32)

        history_list = list(self.history)
        embeddings: List[np.ndarray] = []

        for aid in ASSET_IDS:
            try:
                seq = self.feature_engineer.build_sequence(history_list, aid)
                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    emb = self.model.get_forecast_embedding(x)
                embeddings.append(emb.squeeze(0).cpu().numpy())
            except Exception:
                embeddings.append(np.zeros(128, dtype=np.float32))

        return np.concatenate(embeddings).astype(np.float32)

    # ── Risk timeline ─────────────────────────────────────────────────────────

    def get_risk_timeline(self) -> List[Dict[str, Any]]:
        """
        Return all 12 assets sorted by predicted failure urgency.

        Each entry: {asset_id, asset_type, hours_to_failure, risk_level,
                     current_wear, predicted_wear_24h}
        """
        result = self._forecast_cache if self._forecast_cache else self.forecast()
        history_list = list(self.history)

        timeline: List[Dict[str, Any]] = []
        for aid in ASSET_IDS:
            preds = result["forecasts"].get(aid, np.full((24, 8), 0.5))
            hours = result["predicted_failure_hours"].get(aid)

            current_wear = 0.3
            if len(history_list) >= SEQ_LEN:
                seq = self.feature_engineer.build_sequence(history_list, aid)
                current_wear = float(seq[-1, 5])

            predicted_wear_24h = float(np.clip(preds[-1, 5], 0.0, 1.0))
            risk = _risk_level(hours)

            timeline.append(
                {
                    "asset_id": aid,
                    "asset_type": ASSET_TYPES[aid],
                    "hours_to_failure": hours,
                    "risk_level": risk,
                    "current_wear": round(current_wear, 3),
                    "predicted_wear_24h": round(predicted_wear_24h, 3),
                }
            )

        _order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        timeline.sort(
            key=lambda x: (_order[x["risk_level"]], x["hours_to_failure"] or 1e9)
        )
        return timeline

    # ── Health check ──────────────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        """True if model is loaded and buffer has at least 48 readings."""
        return TORCH_AVAILABLE and self.model is not None and len(self.history) >= SEQ_LEN
