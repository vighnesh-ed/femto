"""Stage 5b: plain 1D-CNN regressor (ported from the team's original Keras notebook).

    python -m src.benchmark --models cnn1d

  Conv1d(F->32, k=3) -> ReLU -> BatchNorm
  Conv1d(32->64, k=3) -> ReLU -> BatchNorm -> MaxPool(2)
  Flatten -> Linear(->32) -> ReLU -> Dropout(0.2) -> Linear(32->16) -> ReLU -> Linear(16->1)

No recurrent layer: the CNN-LSTM's convolutional front end plus a dense head, which isolates
what the LSTM adds. Trained through src/models/training.py (see recurrent.py for what that
changes against the notebook).
"""
from __future__ import annotations

import torch
from torch import nn

from src.windows import WINDOW


class CNN1D(nn.Module):
    def __init__(self, n_features: int, window: int = WINDOW, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * (window // 2), 32), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, W, F) -> (B,)
        return self.head(self.conv(x.transpose(1, 2))).squeeze(-1)
