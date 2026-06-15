"""
2-layer LSTM with self-attention for 48h→24h telemetry forecasting.

Architecture rationale:
  - 2-layer LSTM is sufficient for the relatively low-dimensional (8 features)
    and short-horizon (48→24) task; deeper networks overfit on simulation data.
  - Self-attention over LSTM timestep outputs lets operators inspect which past
    hours the model considers most informative — critical for explainability.
  - Xavier/orthogonal weight initialization prevents vanishing/exploding gradients
    in the first few training epochs.

In production: retrain monthly on real BMS data logs. The attention heatmap
can highlight previously unknown periodic maintenance patterns.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DataCenterLSTM(nn.Module):
    """
    Multi-step telemetry forecaster for data center critical assets.

    Input:  (batch, seq_len=48, input_size=8)
    Output: predictions (batch, 24, 8), attention_weights (batch, 48)
    """

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        forecast_horizon: int = 24,
        output_size: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention: project each timestep output to a scalar score
        self.attention_linear = nn.Linear(hidden_size, 1, bias=False)

        # Fully connected head
        self.fc1 = nn.Linear(hidden_size, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, forecast_horizon * output_size)
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
            elif "weight" in name:
                nn.init.xavier_uniform_(param.data)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)

        Returns:
            predictions:      (batch, forecast_horizon, output_size)
            attention_weights: (batch, seq_len)  — sums to 1.0 per sample
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Self-attention over timesteps
        attn_scores = self.attention_linear(lstm_out).squeeze(-1)  # (batch, seq_len)
        attention_weights = torch.softmax(attn_scores, dim=-1)      # (batch, seq_len)
        context = (attention_weights.unsqueeze(-1) * lstm_out).sum(dim=1)  # (batch, hidden_size)

        # FC head
        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.fc2(out)  # (batch, forecast_horizon * output_size)
        predictions = out.reshape(-1, self.forecast_horizon, self.output_size)

        return predictions, attention_weights

    def get_forecast_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the context vector before the FC head.

        Shape: (batch, hidden_size=128)
        Used by PPO as a compact forward-looking representation.
        """
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attention_linear(lstm_out).squeeze(-1)
        attention_weights = torch.softmax(attn_scores, dim=-1)
        context = (attention_weights.unsqueeze(-1) * lstm_out).sum(dim=1)
        return context
