"""LSTM training loop with early stopping and per-feature weighted loss."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from backend.lstm.feature_engineer import (
    ASSET_IDS,
    N_FEATURES,
    FEATURE_NAMES,
    TelemetryFeatureEngineer,
)
from backend.lstm.model import DataCenterLSTM

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "lstm_world_model.pt"

# Feature weights: inlet_temp=2.0, wear_score=3.0, others=1.0
_FEATURE_WEIGHTS = torch.tensor(
    [2.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0], dtype=torch.float32
)


def _best_device() -> str:
    """Pick the fastest device for this small LSTM model.

    For small models (< 1M params), CPU with large batches outperforms MPS
    due to kernel launch overhead. Prefer CUDA for larger deployments.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LSTMTrainer:
    """Train and evaluate the DataCenterLSTM model."""

    def __init__(
        self,
        model: DataCenterLSTM,
        learning_rate: float = 1e-3,
        device: str = "auto",
    ) -> None:
        self.model = model
        self.device = torch.device(_best_device() if device == "auto" else device)
        logger.info("LSTMTrainer using device: %s", self.device)
        self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.feature_engineer = TelemetryFeatureEngineer()
        self.model_path = _MODEL_PATH
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Loss ──────────────────────────────────────────────────────────────────

    def calculate_weighted_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """MSE weighted by feature importance (wear_score=3x, inlet_temp=2x)."""
        weights = _FEATURE_WEIGHTS.to(self.device)
        mse = (predictions - targets) ** 2  # (batch, horizon, features)
        weighted = (mse * weights).mean()
        return weighted

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """
        Train for specified epochs with early stopping (patience=10).

        Returns training history dict.
        """
        t0 = time.time()

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.float32)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_v = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_tr, y_tr)
        _pin = self.device.type in ("cuda",)  # MPS doesn't support pin_memory
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=_pin)

        self.model.to(self.device)

        best_val_loss = float("inf")
        best_epoch = 1
        patience_counter = 0
        patience = 10

        train_losses: list = []
        val_losses: list = []
        train_maes: list = []
        val_maes: list = []

        for epoch in range(1, epochs + 1):
            # ── Train ──
            self.model.train()
            ep_loss = 0.0
            ep_mae = 0.0
            n_batches = 0

            for X_b, y_b in loader:
                X_b = X_b.to(self.device)
                y_b = y_b.to(self.device)
                self.optimizer.zero_grad()
                preds, _ = self.model(X_b)
                loss = self.calculate_weighted_loss(preds, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                ep_loss += loss.item()
                ep_mae += (preds - y_b).abs().mean().item()
                n_batches += 1

            ep_loss /= max(n_batches, 1)
            ep_mae /= max(n_batches, 1)

            # ── Validate ──
            self.model.eval()
            with torch.no_grad():
                v_preds, _ = self.model(X_v)
                v_loss = self.calculate_weighted_loss(v_preds, y_v).item()
                v_mae = (v_preds - y_v).abs().mean().item()

            train_losses.append(ep_loss)
            val_losses.append(v_loss)
            train_maes.append(ep_mae)
            val_maes.append(v_mae)

            # ── Early stopping ──
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d (best=%d)", epoch, best_epoch)
                    break

            if epoch % 10 == 0:
                logger.info(
                    "Epoch %3d/%d — train_loss=%.4f val_loss=%.4f val_mae=%.4f",
                    epoch, epochs, ep_loss, v_loss, v_mae,
                )

        return {
            "train_loss": float(train_losses[-1]),
            "val_loss": float(val_losses[-1]),
            "best_epoch": best_epoch,
            "train_mae": float(train_maes[-1]),
            "val_mae": float(val_maes[-1]),
            "training_time_seconds": round(time.time() - t0, 1),
        }

    # ── Pre-training on simulator ─────────────────────────────────────────────

    def pretrain_on_simulator(self, n_steps: int = 4_000) -> Dict[str, Any]:
        """
        Generate synthetic training data from simulator and train the model.

        Target: < 120 seconds on M-series MacBook with n_steps=10000.
        Saves model to models/lstm_world_model.pt.
        """
        from backend.simulator import DataCenterSimulator

        logger.info("Generating %d simulator steps for LSTM pre-training…", n_steps)
        sim = DataCenterSimulator()
        X_tr, y_tr, X_v, y_v = self.feature_engineer.generate_training_data(
            sim, n_steps=n_steps
        )
        logger.info(
            "Training data ready: %d train / %d val samples", len(X_tr), len(X_v)
        )

        # Use larger batch size for faster pre-training while maintaining quality
        history = self.train(X_tr, y_tr, X_v, y_v, epochs=50, batch_size=512)
        history["n_train_samples"] = len(X_tr)
        history["model_path"] = str(self.model_path)
        logger.info(
            "Pre-training complete in %.1fs — best epoch %d, val_mae=%.4f",
            history["training_time_seconds"], history["best_epoch"], history["val_mae"],
        )
        return history

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Return per-feature MAE/RMSE and per-horizon accuracy."""
        X_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds, _ = self.model(X_t)

        preds_np = preds.cpu().numpy()  # (n, 24, 8)
        y_np = y_test  # (n, 24, 8)

        mae_per_feat = np.abs(preds_np - y_np).mean(axis=(0, 1))  # (8,)
        rmse_per_feat = np.sqrt(((preds_np - y_np) ** 2).mean(axis=(0, 1)))  # (8,)

        horizon_mae: Dict[str, float] = {}
        for h, label in [(0, "1h"), (5, "6h"), (11, "12h"), (23, "24h")]:
            horizon_mae[label] = float(np.abs(preds_np[:, h, :] - y_np[:, h, :]).mean())

        overall_mae = float(np.abs(preds_np - y_np).mean())
        accuracy_score = max(0.0, 100.0 * (1.0 - overall_mae))

        return {
            "per_feature_mae": {
                name: float(mae_per_feat[i]) for i, name in enumerate(FEATURE_NAMES)
            },
            "per_feature_rmse": {
                name: float(rmse_per_feat[i]) for i, name in enumerate(FEATURE_NAMES)
            },
            "per_horizon_mae": horizon_mae,
            "overall_accuracy_score": round(accuracy_score, 1),
            "overall_mae": round(overall_mae, 4),
        }
