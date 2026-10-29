"""Stage 5a: MLP on the flattened window (ported from the team's original Keras MLP notebook).

    python -m src.benchmark --models mlp

  Flatten(W*F) -> Linear(128) -> ReLU -> BatchNorm -> Dropout(0.2)
               -> Linear(64)  -> ReLU -> BatchNorm -> Dropout(0.2)
               -> Linear(32)  -> ReLU -> Linear(1)

The non-sequential deep baseline: it sees the same 30 x 20 window as the recurrent models but
with no notion of time order. Trained through src/models/training.py like every other deep model.
"""
from __future__ import annotations

import torch
from torch import nn

from src.windows import WINDOW


class MLP(nn.Module):
    def __init__(self, n_features: int, window: int = WINDOW, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window * n_features, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, W, F) -> (B,)
        return self.net(x).squeeze(-1)
