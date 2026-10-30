"""Stage 3: health indicator, knee point (first predicting time) and RUL labels.

All RUL values are in snapshots (1 snapshot = 10 s). Multiply by SNAPSHOT_PERIOD_S / 60 for
minutes.

Piecewise RUL uses ONE global cap for every bearing, not a per-bearing cap at each bearing's
own knee. Per-bearing caps give indistinguishable healthy-state inputs wildly different
targets (Bearing1_1 healthy -> ~900 snapshots, Bearing3_1 healthy -> ~5, because 3_1 fails
abruptly), which no model can fit. The cap is the median post-knee life of the TRAINING
bearings (fit_rul_cap). Evaluate against true linear RUL either way.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

SMOOTH_WINDOW = 30   # 5 min of snapshots
BASELINE_SKIP = 30   # first 5 min: run-in transient, kept out of the baseline
BASELINE_LEN = 60    # next 10 min: healthy baseline
KNEE_K = 3.0         # threshold = baseline mean + K * baseline std
KNEE_PERSIST = 6     # smoothed RMS must stay above threshold for 1 min
HEALTHY_PERCENTILE = 5.0  # health indicator's zero: this percentile of the smoothed RMS


def vibration_rms(df: pd.DataFrame) -> np.ndarray:
    """RMS over both channels, sqrt((rms_h^2 + rms_v^2) / 2) - no channel preference baked in."""
    return np.sqrt((df["rms_h"].to_numpy() ** 2 + df["rms_v"].to_numpy() ** 2) / 2)


def smooth(x: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Causal (trailing) rolling median: robust to the isolated spikes in e.g. Bearing1_2, and
    uses only past snapshots, so the same smoothing is valid online on a test bearing."""
    return pd.Series(x).rolling(window, min_periods=1).median().to_numpy()


def baseline_stats(rms: np.ndarray, skip: int = BASELINE_SKIP,
                   length: int = BASELINE_LEN) -> tuple[float, float]:
    """Mean and std of the RAW rms over the early-life baseline window.

    Raw, not smoothed: the smoothed baseline's std is so small that ordinary noise crosses
    mean + 3*std (Bearing1_2 triggered at 17% of life that way).
    """
    base = rms[skip:skip + length]
    if len(base) < length:
        raise ValueError(f"run of {len(rms)} snapshots is shorter than the baseline window")
    return float(base.mean()), float(base.std(ddof=1))


def detect_knee(df: pd.DataFrame, window: int = SMOOTH_WINDOW, skip: int = BASELINE_SKIP,
                length: int = BASELINE_LEN, k: float = KNEE_K, persist: int = KNEE_PERSIST) -> int:
    """Snapshot index where degradation starts: first point at which smoothed RMS stays above
    baseline mean + k*std for `persist` consecutive snapshots. Returns len(df) - 1 if never.

    Only uses the first skip+length snapshots for the threshold and causal smoothing, so it
    can also run online as a first-predicting-time detector.
    """
    rms = vibration_rms(df)
    mean, std = baseline_stats(rms, skip, length)
    above = smooth(rms, window) > mean + k * std
    above[:skip + length] = False
    sustained = np.convolve(above, np.ones(persist, dtype=int), mode="valid") == persist
    hits = np.flatnonzero(sustained)
    return int(hits[0]) if hits.size else len(df) - 1


def health_indicator(df: pd.DataFrame, window: int = SMOOTH_WINDOW, healthy_pct: float = HEALTHY_PERCENTILE) -> np.ndarray:
    """Smoothed RMS rescaled to 0 at the healthy level and 1 at its maximum, so it stays in [0, 1].

    The healthy level is a low percentile of the smoothed RMS, not the early-life baseline that
    detect_knee uses: Bearing2_4..2_7 have run-in transients in minutes 5-15, which put that
    baseline above their real healthy level and made the indicator run to -10.

    Uses the whole run (its percentile and max), so it is for plots and analysis only, never a
    model input.
    """
    s = smooth(vibration_rms(df), window)
    low = np.percentile(s, healthy_pct)
    return np.clip((s - low) / (s.max() - low), 0.0, 1.0)


def rul_linear(n: int) -> np.ndarray:
    """Row i of an n-snapshot run-to-failure has RUL n - 1 - i."""
    return np.arange(n - 1, -1, -1, dtype=np.float64)


def rul_piecewise(n: int, cap: float) -> np.ndarray:
    """Linear RUL clipped at a constant cap during healthy life."""
    return np.minimum(rul_linear(n), cap)


def fit_rul_cap(train_dfs: Iterable[pd.DataFrame]) -> float:
    """Global piecewise cap = median post-knee life over the training bearings."""
    return float(np.median([len(df) - 1 - detect_knee(df) for df in train_dfs]))
