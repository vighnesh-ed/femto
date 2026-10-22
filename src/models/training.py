"""Shared PyTorch training code for every deep sequence model (GRU, 1D-CNN, CNN-LSTM, ...).

Route every deep model through these functions so the numbers are comparable with each other
and with the baselines - same windows, same targets, same rule for choosing epochs:

    split = make_split(train_dfs, test_dfs, "piecewise")                      # src/windows.py
    epochs, curves = select_epochs(model_fn, train_dfs, "piecewise", cfg)
    model, _ = fit(model_fn, split.train, split.y_scale, replace(cfg, epochs=epochs))
    rul = predict(model, split.test.X, split.y_scale)                           # in snapshots

`model_fn` is a zero-argument callable returning a fresh nn.Module that maps a float32 batch
of shape (B, W, n_features) to a (B,) tensor of RUL / y_scale.

The epoch count is chosen without touching the test bearings: each training bearing is held
out in turn, a model is trained on the others, and the epoch with the lowest validation RMSE
averaged over the held-out bearings is used to retrain on all training bearings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.evaluate import SNAPSHOTS_TO_MIN
from src.windows import WINDOW, WindowSet, make_split

ModelFn = Callable[[], nn.Module]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 60
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 0


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def use_cpu_threads(n: int = 4) -> None:
    """Call once at the start of a CPU run. These models are tiny; with PyTorch's default of
    one thread per core, thread overhead made an epoch 10x slower (5 s vs 0.5 s on 20 cores)."""
    torch.set_num_threads(n)


def as_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device())


@torch.no_grad()
def _forward(model: nn.Module, X: np.ndarray, batch_size: int) -> np.ndarray:
    xs = as_tensor(X)
    out = torch.cat([model(xs[i:i + batch_size]) for i in range(0, len(xs), batch_size)])
    return out.cpu().numpy().astype(np.float64)


def predict(model: nn.Module, X: np.ndarray, y_scale: float, batch_size: int = 2048) -> np.ndarray:
    """Deterministic RUL prediction (dropout off), in snapshots."""
    model.eval()
    return _forward(model, X, batch_size) * y_scale


def mc_dropout_predict(model: nn.Module, X: np.ndarray, y_scale: float, n_samples: int = 50,
                       batch_size: int = 2048, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """MC-dropout: dropout stays active at inference; returns mean and std (snapshots) over
    n_samples stochastic passes. Only nn.Dropout layers are switched on."""
    model.eval()
    dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    if not dropouts:
        raise ValueError("model has no nn.Dropout layers, MC-dropout would return zero spread")
    torch.manual_seed(seed)
    try:
        for m in dropouts:
            m.train()
        draws = np.stack([_forward(model, X, batch_size) for _ in range(n_samples)]) * y_scale
    finally:
        model.eval()
    return draws.mean(axis=0), draws.std(axis=0)


def fit(model_fn: ModelFn, train: WindowSet, y_scale: float, cfg: TrainConfig = TrainConfig(),
        val: WindowSet | None = None) -> tuple[nn.Module, pd.DataFrame]:
    """Train for exactly cfg.epochs on MSE of RUL / y_scale.

    If val is given, validation RMSE (minutes, against val.y) is recorded after every epoch.
    That curve is what select_epochs uses; it never picks the weights of a final model.
    """
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = model_fn().to(device())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    X, y = as_tensor(train.X), as_tensor(train.y / y_scale)
    gen = torch.Generator().manual_seed(cfg.seed)

    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(len(X), generator=gen).to(X.device)
        total = 0.0
        for i in range(0, len(X), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(X[idx]), y[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += loss.item() * len(idx)
        row = {"epoch": epoch, "train_mse": total / len(X)}
        if val is not None:
            err = np.clip(predict(model, val.X, y_scale), 0, None) - val.y
            row["val_rmse_min"] = float(np.sqrt(np.mean(err**2))) * SNAPSHOTS_TO_MIN
        history.append(row)
    return model, pd.DataFrame(history)


def select_epochs(model_fn: ModelFn, train_dfs: dict[str, pd.DataFrame], target: str,
                  cfg: TrainConfig = TrainConfig(), window: int = WINDOW) -> tuple[int, pd.DataFrame]:
    """Inner leave-one-bearing-out over the TRAINING bearings to choose the epoch count.

    Each fold refits the scaler and RUL cap on its own training bearings (make_split), so the
    held-out bearing leaks nothing. Returns (best_epoch, curves): curves has one validation
    RMSE column per held-out bearing plus their 'mean', indexed by epoch (1..cfg.epochs).
    """
    curves = {}
    for held_out in train_dfs:
        inner = {b: df for b, df in train_dfs.items() if b != held_out}
        split = make_split(inner, {held_out: train_dfs[held_out]}, target, window)
        _, history = fit(model_fn, split.train, split.y_scale, cfg, val=split.test)
        curves[held_out] = history.set_index("epoch")["val_rmse_min"]
    curves = pd.DataFrame(curves)
    curves["mean"] = curves.mean(axis=1)
    return int(curves["mean"].idxmin()), curves
