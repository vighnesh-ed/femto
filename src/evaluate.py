"""Metrics (MAE / RMSE / R^2 on RUL in minutes, official PHM 2012 score) and report plots.

Prediction results are passed around as a DataFrame with one row per prediction (results_frame):
    bearing, snapshot_idx, rul_true_min, rul_pred_min [, rul_std_min]
where rul_true_min is the TRUE linear RUL, whatever the model was trained on, and the optional
rul_std_min is a predictive standard deviation (e.g. MC-dropout).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.labels import detect_knee, health_indicator, smooth, vibration_rms
from src.loader import PROJECT_ROOT, SNAPSHOT_PERIOD_S
from src.windows import WindowSet

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SNAPSHOTS_TO_MIN = SNAPSHOT_PERIOD_S / 60
LIFE_STAGES = [0.0, 0.25, 0.5, 0.75, 1.0]


def results_frame(ws: WindowSet, pred: np.ndarray, std: np.ndarray | None = None) -> pd.DataFrame:
    """Predictions in snapshots -> results DataFrame in minutes. Negative RUL is clipped to 0."""
    df = pd.DataFrame({
        "bearing": ws.bearing, "snapshot_idx": ws.snapshot_idx,
        "rul_true_min": ws.rul_true * SNAPSHOTS_TO_MIN,
        "rul_pred_min": np.clip(pred, 0, None) * SNAPSHOTS_TO_MIN,
    })
    if std is not None:
        df["rul_std_min"] = std * SNAPSHOTS_TO_MIN
    return df


def phm_score(act_rul, pred_rul):
    """Official IEEE PHM 2012 scoring. Returns mean accuracy in [0, 1]; higher is better.

    Defined for act_rul > 0 only (% error). The challenge applies it once per test bearing,
    at the Test_set truncation point - not to every window of a run-to-failure curve.
    """
    act_rul = np.asarray(act_rul, dtype=float)
    pred_rul = np.asarray(pred_rul, dtype=float)
    if np.any(act_rul <= 0):
        raise ValueError("phm_score is undefined for act_rul <= 0")
    er = 100.0 * (act_rul - pred_rul) / act_rul          # % error; negative = late prediction
    a = np.where(er <= 0,
                 np.exp(-np.log(0.5) * (er / 5.0)),       # overestimate: harsh
                 np.exp( np.log(0.5) * (er / 20.0)))      # underestimate: lenient
    return float(np.mean(a))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    err = np.asarray(y_pred, dtype=float) - y_true
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "R2": float(1 - np.sum(err**2) / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(len(y_true)),
    }


def per_bearing_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """One row per bearing plus a pooled row, so one bad bearing cannot hide in a mean."""
    rows = {b: regression_metrics(g.rul_true_min, g.rul_pred_min) for b, g in results.groupby("bearing")}
    rows["pooled"] = regression_metrics(results.rul_true_min, results.rul_pred_min)
    return pd.DataFrame(rows).T


def error_by_life_stage(results: pd.DataFrame, stages=LIFE_STAGES) -> pd.DataFrame:
    """MAE (minutes) per bearing per fraction-of-life-elapsed bin: early vs late-life error."""
    life_frac = results.snapshot_idx * SNAPSHOTS_TO_MIN / (
        results.snapshot_idx * SNAPSHOTS_TO_MIN + results.rul_true_min)
    stage = pd.cut(life_frac, stages, include_lowest=True,
                   labels=[f"{a:.0%}-{b:.0%}" for a, b in zip(stages[:-1], stages[1:])])
    abs_err = (results.rul_pred_min - results.rul_true_min).abs()
    return abs_err.groupby([results.bearing, stage], observed=False).mean().unstack()


def interval_coverage(results: pd.DataFrame, z: float = 2.0) -> pd.DataFrame:
    """Share of predictions whose pred +/- z*std interval contains the true RUL, per bearing,
    overall and over the last 25% of life. ~95% for z=2 if the uncertainty is calibrated."""
    inside = (results.rul_true_min - results.rul_pred_min).abs() <= z * results.rul_std_min
    life_frac = results.snapshot_idx * SNAPSHOTS_TO_MIN / (
        results.snapshot_idx * SNAPSHOTS_TO_MIN + results.rul_true_min)
    late = life_frac >= 0.75
    rows = {b: {"coverage": inside[g.index].mean(), "coverage_last25": inside[g.index][late[g.index]].mean(),
                "mean_std_min": g.rul_std_min.mean()} for b, g in results.groupby("bearing")}
    rows["pooled"] = {"coverage": inside.mean(), "coverage_last25": inside[late].mean(),
                      "mean_std_min": results.rul_std_min.mean()}
    return pd.DataFrame(rows).T


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_degradation(dfs: dict[str, pd.DataFrame], path: Path = FIGURES_DIR / "degradation_rms.png") -> Path:
    """RMS vs time for each bearing: raw, causal-median smoothed, and detected knee."""
    n = len(dfs)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.3 * nrows), squeeze=False)
    for ax, (bearing, df) in zip(axes.flat, dfs.items()):
        hours = df.elapsed_s.to_numpy() / 3600
        rms = vibration_rms(df)
        knee = detect_knee(df)
        ax.plot(hours, rms, lw=0.5, color="0.7", label="RMS")
        ax.plot(hours, smooth(rms), lw=1.3, color="C0", label="smoothed")
        ax.axvline(hours[knee], color="C3", ls="--", lw=1, label=f"knee ({knee / (len(df) - 1):.0%} of life)")
        ax.set_yscale("log")
        ax.set_title(bearing)
        ax.set_xlabel("operating time [h]")
        ax.set_ylabel("RMS [g]")
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return _save(fig, path)


def plot_health_indicator(dfs: dict[str, pd.DataFrame], path: Path = FIGURES_DIR / "health_indicator.png") -> Path:
    """Health indicator (smoothed RMS rescaled to 0 = healthy baseline, 1 = run maximum) vs time
    for each bearing, with the detected knee. Analysis only: it uses the whole run's maximum."""
    n = len(dfs)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.0 * nrows), squeeze=False)
    for ax, (bearing, df) in zip(axes.flat, dfs.items()):
        hours = df.elapsed_s.to_numpy() / 3600
        knee = detect_knee(df)
        ax.plot(hours, health_indicator(df), lw=1.3, color="C0")
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axhline(1, color="0.7", lw=0.6)
        ax.axvline(hours[knee], color="C3", ls="--", lw=1, label=f"knee ({knee / (len(df) - 1):.0%} of life)")
        ax.set_title(bearing)
        ax.set_xlabel("operating time [h]")
        ax.set_ylabel("health indicator")
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return _save(fig, path)


def plot_rul_predictions(results: pd.DataFrame, path: Path, title: str = "",
                         cuts: dict[str, int] | None = None) -> Path:
    """Actual vs predicted RUL over operating time, one panel per bearing. Draws a +/-2 std band
    when results has rul_std_min, and a vertical line at cuts[bearing] (a snapshot index)."""
    bearings = list(results.bearing.unique())
    ncols = min(len(bearings), 4)
    nrows = int(np.ceil(len(bearings) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows), squeeze=False)
    for ax, b in zip(axes.flat, bearings):
        g = results[results.bearing == b]
        t = g.snapshot_idx * SNAPSHOT_PERIOD_S / 3600
        if "rul_std_min" in g:
            lo = np.clip(g.rul_pred_min - 2 * g.rul_std_min, 0, None)
            ax.fill_between(t, lo, g.rul_pred_min + 2 * g.rul_std_min, color="C1", alpha=0.25,
                            lw=0, label="±2σ (MC-dropout)")
        ax.plot(t, g.rul_pred_min, lw=0.7, color="C1", label="predicted")
        ax.plot(t, g.rul_true_min, lw=1.5, color="k", label="actual")
        if cuts and b in cuts:
            ax.axvline(cuts[b] * SNAPSHOT_PERIOD_S / 3600, color="C3", ls="--", lw=1, label="Test_set cut")
        ax.set_title(b)
        ax.set_xlabel("operating time [h]")
        ax.set_ylabel("RUL [min]")
        ax.legend(fontsize=8)
    for ax in axes.flat[len(bearings):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title)
    return _save(fig, path)


def plot_error_vs_life(results: pd.DataFrame, path: Path, title: str = "") -> Path:
    """Prediction error against true remaining life. Positive error = late (dangerous)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for b, g in results.groupby("bearing"):
        ax.plot(g.rul_true_min, g.rul_pred_min - g.rul_true_min, lw=0.7, label=b)
    ax.axhline(0, color="k", lw=0.8)
    ax.invert_xaxis()
    ax.set_xlabel("true RUL [min]  (failure at right)")
    ax.set_ylabel("predicted - true [min]")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return _save(fig, path)


def plot_epoch_selection(curves: pd.DataFrame, best_epoch: int, path: Path, title: str = "") -> Path:
    """Validation RMSE per held-out training bearing (select_epochs) and their mean."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for col in curves.columns.drop("mean"):
        ax.plot(curves.index, curves[col], lw=0.8, alpha=0.7, label=f"held out {col}")
    ax.plot(curves.index, curves["mean"], lw=2, color="k", label="mean")
    ax.axvline(best_epoch, color="C3", ls="--", lw=1, label=f"chosen: {best_epoch} epochs")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation RMSE [min]")
    ax.legend(fontsize=7)
    if title:
        ax.set_title(title)
    return _save(fig, path)


def main() -> None:
    """Degradation figures for the report: RMS curves and health indicators, learning and test bearings."""
    from src.features import load_features
    from src.loader import LEARNING_BEARINGS, TEST_BEARINGS

    learning = {b: load_features(b) for b in LEARNING_BEARINGS}
    test = {b: load_features(b) for b in TEST_BEARINGS}
    for path in (plot_degradation(learning, FIGURES_DIR / "degradation_rms.png"),
                 plot_degradation(test, FIGURES_DIR / "degradation_rms_test.png"),
                 plot_health_indicator(learning, FIGURES_DIR / "health_indicator.png"),
                 plot_health_indicator(test, FIGURES_DIR / "health_indicator_test.png")):
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
