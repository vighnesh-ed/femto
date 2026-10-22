"""Stage 4: per-bearing feature frames -> scaled sliding windows.

Each sample is W consecutive snapshots, shape (W, n_features), labelled with the RUL at the
window's LAST step. Scaling statistics come from the training bearings only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features import SIGNAL_FEATURES
from src.labels import fit_rul_cap, rul_linear, rul_piecewise

WINDOW = 30  # 5 minutes of operation

# Split by BEARING, never by row (CLAUDE.md).
DEFAULT_TRAIN = ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2"]
DEFAULT_TEST = ["Bearing3_1", "Bearing3_2"]

# Strictly positive magnitudes spanning orders of magnitude (band1_h: 0.07 healthy -> 27 at
# failure). log10 before z-scoring stops the last few minutes of each run dominating the scale.
LOG_FEATURES = [c for c in SIGNAL_FEATURES if c.split("_")[0] in
                {"rms", "kurt", "peak", "crest", "p2p", "band1", "band2", "band3", "band4"}]


class FeatureScaler:
    """log10 on LOG_FEATURES, then z-score. Fit on training bearings only."""

    def __init__(self, columns: list[str] = SIGNAL_FEATURES):
        self.columns = list(columns)
        self._log_mask = np.array([c in LOG_FEATURES for c in self.columns])
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def _log(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.columns].to_numpy(dtype=np.float64, copy=True)
        x[:, self._log_mask] = np.log10(x[:, self._log_mask])
        return x

    def fit(self, train_dfs: list[pd.DataFrame]) -> "FeatureScaler":
        x = np.concatenate([self._log(df) for df in train_dfs])
        self.mean_, self.std_ = x.mean(axis=0), x.std(axis=0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("FeatureScaler.transform called before fit")
        return (self._log(df) - self.mean_) / self.std_


def sliding_windows(x: np.ndarray, window: int = WINDOW, stride: int = 1) -> np.ndarray:
    """(n, f) -> (m, window, f), m = (n - window) // stride + 1. Returns a read-only view."""
    if len(x) < window:
        raise ValueError(f"sequence of {len(x)} is shorter than window {window}")
    return np.lib.stride_tricks.sliding_window_view(x, window, axis=0)[::stride].transpose(0, 2, 1)


@dataclass
class WindowSet:
    X: np.ndarray            # (m, W, f) scaled features
    y: np.ndarray            # (m,) training target in snapshots (linear or piecewise)
    rul_true: np.ndarray     # (m,) true linear RUL in snapshots - always evaluate against this
    bearing: np.ndarray      # (m,) bearing name per sample - for per-bearing metrics
    snapshot_idx: np.ndarray # (m,) index of each window's last snapshot

    def __len__(self) -> int:
        return len(self.y)


def build_windows(dfs: dict[str, pd.DataFrame], scaler: FeatureScaler, window: int = WINDOW,
                  cap: float | None = None, stride: int = 1) -> WindowSet:
    """Window every bearing in dfs. cap=None -> linear RUL target, else piecewise with that cap."""
    parts = []
    for bearing, df in dfs.items():
        n = len(df)
        last = np.arange(window - 1, n, stride)
        true = rul_linear(n)
        target = true if cap is None else rul_piecewise(n, cap)
        parts.append(WindowSet(
            X=sliding_windows(scaler.transform(df), window, stride),
            y=target[last], rul_true=true[last],
            bearing=np.full(len(last), bearing), snapshot_idx=last,
        ))
    return WindowSet(*(np.concatenate([getattr(p, f) for p in parts])
                       for f in ("X", "y", "rul_true", "bearing", "snapshot_idx")))


@dataclass
class Split:
    train: WindowSet
    test: WindowSet
    scaler: FeatureScaler
    cap: float | None  # piecewise RUL cap in snapshots; None for a linear target

    @property
    def y_scale(self) -> float:
        """Neural nets regress RUL / y_scale: the cap, or the longest training run if linear."""
        return self.cap if self.cap is not None else float(self.train.y.max())


def make_split(train_dfs: dict[str, pd.DataFrame], test_dfs: dict[str, pd.DataFrame],
               target: str = "piecewise", window: int = WINDOW) -> Split:
    """Windows for one bearing-wise split. Scaler and piecewise cap are fitted on train_dfs
    only, then applied to both sides - every model should build its data through this."""
    if set(train_dfs) & set(test_dfs):
        raise ValueError(f"bearings in both train and test: {set(train_dfs) & set(test_dfs)}")
    if target not in ("linear", "piecewise"):
        raise ValueError(f"target must be 'linear' or 'piecewise', got {target!r}")
    scaler = FeatureScaler().fit(list(train_dfs.values()))
    cap = fit_rul_cap(train_dfs.values()) if target == "piecewise" else None
    return Split(train=build_windows(train_dfs, scaler, window, cap),
                 test=build_windows(test_dfs, scaler, window, cap),
                 scaler=scaler, cap=cap)
