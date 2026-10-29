"""Stage 5b: plain LSTM and GRU regressors (ported from the team's original Keras notebooks).

    python -m src.benchmark --models lstm gru

Input: one window of W=30 snapshots x 20 features (src/windows.py).

  LSTM(32) or GRU(32) -> last hidden state -> Dropout(0.1) -> Linear(32 -> 16) -> ReLU -> Linear(16 -> 1)

Same layer layout as the team's Keras LSTM and GRU. Trained through src/models/training.py like
every other deep model, so it uses that module's MSE loss, AdamW and leave-one-bearing-out epoch
rule, not the Keras versions' Huber loss, Adam(5e-4) and tail-of-array validation split.
"""
from __future__ import annotations

import torch
from torch import nn

CELLS = {"lstm": nn.LSTM, "gru": nn.GRU}


class RecurrentRegressor(nn.Module):
    def __init__(self, n_features: int, cell: str = "lstm", hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.rnn = CELLS[cell](n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, W, F) -> (B,)
        h, _ = self.rnn(x)
        return self.head(h[:, -1]).squeeze(-1)
