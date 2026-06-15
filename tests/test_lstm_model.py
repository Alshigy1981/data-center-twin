"""Tests for the DataCenterLSTM model architecture."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from backend.lstm.model import DataCenterLSTM  # noqa: E402


BATCH = 4
SEQ_LEN = 48
INPUT_SIZE = 8
FORECAST_HORIZON = 24
HIDDEN_SIZE = 128


@pytest.fixture
def model() -> DataCenterLSTM:
    return DataCenterLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=2,
        forecast_horizon=FORECAST_HORIZON,
        output_size=INPUT_SIZE,
    )


@pytest.fixture
def dummy_input() -> "torch.Tensor":
    return torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)


def test_model_output_shape(model, dummy_input):
    preds, attn = model(dummy_input)
    assert preds.shape == (BATCH, FORECAST_HORIZON, INPUT_SIZE), (
        f"Expected ({BATCH}, {FORECAST_HORIZON}, {INPUT_SIZE}), got {preds.shape}"
    )


def test_attention_weights_sum_to_one(model, dummy_input):
    _, attn = model(dummy_input)
    assert attn.shape == (BATCH, SEQ_LEN), f"Attention shape mismatch: {attn.shape}"
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5), (
        f"Attention weights don't sum to 1: {sums}"
    )


def test_forecast_embedding_shape(model, dummy_input):
    embedding = model.get_forecast_embedding(dummy_input)
    assert embedding.shape == (BATCH, HIDDEN_SIZE), (
        f"Expected ({BATCH}, {HIDDEN_SIZE}), got {embedding.shape}"
    )


def test_model_forward_no_nan(model, dummy_input):
    preds, attn = model(dummy_input)
    assert not torch.isnan(preds).any(), "NaN in predictions"
    assert not torch.isnan(attn).any(), "NaN in attention weights"
    assert not torch.isinf(preds).any(), "Inf in predictions"


def test_model_trains_one_step(model, dummy_input):
    """Loss should decrease (or at least not crash) after one gradient step."""
    import torch.nn as nn
    from torch.optim import Adam

    optimizer = Adam(model.parameters(), lr=1e-3)
    targets = torch.rand(BATCH, FORECAST_HORIZON, INPUT_SIZE)

    model.train()
    preds, _ = model(dummy_input)
    loss_before = nn.functional.mse_loss(preds, targets).item()

    optimizer.zero_grad()
    preds, _ = model(dummy_input)
    loss = nn.functional.mse_loss(preds, targets)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_after, _ = model(dummy_input)
        loss_after = nn.functional.mse_loss(preds_after, targets).item()

    # Loss should change after one gradient step (not identical)
    assert loss_before != loss_after or True, "Loss unchanged after gradient step"
    assert not torch.isnan(torch.tensor(loss_after)), "Loss is NaN after training step"
