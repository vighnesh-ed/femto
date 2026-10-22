"""Stage 5a: Ridge and Random Forest baselines on the default bearing split.

    python -m src.models.baselines

Trains every (model x input x target) combination on DEFAULT_TRAIN, evaluates per bearing on
DEFAULT_TEST against TRUE linear RUL, and writes reports/baselines_learning_split.csv plus
prediction / error figures for each Random Forest window model.

  input  'row'    : features of the window's last snapshot only
         'window' : the whole (W, f) window flattened
  target 'linear' : RUL = N - 1 - i
         'piecewise': linear RUL capped at the median post-knee life of the train bearings
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.evaluate import (FIGURES_DIR, SNAPSHOTS_TO_MIN, error_by_life_stage, per_bearing_metrics,
                          plot_error_vs_life, plot_rul_predictions, results_frame)
from src.features import load_features
from src.loader import PROJECT_ROOT
from src.windows import DEFAULT_TEST, DEFAULT_TRAIN, WindowSet, make_split

REPORTS_DIR = PROJECT_ROOT / "reports"


def make_model(name: str):
    if name == "ridge":
        return Ridge(alpha=1.0)  # sklearn default; untuned, since tuning needs bearing-grouped CV
    if name == "rf":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.33,
                                     n_jobs=-1, random_state=0)
    raise ValueError(name)


def model_input(ws: WindowSet, mode: str) -> np.ndarray:
    if mode == "row":
        return ws.X[:, -1, :]
    if mode == "window":
        return ws.X.reshape(len(ws), -1)
    raise ValueError(mode)


def main() -> None:
    dfs = {b: load_features(b) for b in DEFAULT_TRAIN + DEFAULT_TEST}
    train_dfs = {b: dfs[b] for b in DEFAULT_TRAIN}
    test_dfs = {b: dfs[b] for b in DEFAULT_TEST}
    print(f"train {DEFAULT_TRAIN} -> test {DEFAULT_TEST}\n")

    tables, stage_tables = [], []
    for target in ("linear", "piecewise"):
        split = make_split(train_dfs, test_dfs, target)
        train, test = split.train, split.test
        if split.cap is not None:
            print(f"piecewise cap = {split.cap:.0f} snapshots ({split.cap * SNAPSHOTS_TO_MIN:.1f} min)")
        for model_name in ("ridge", "rf"):
            for mode in ("row", "window"):
                tag = f"{model_name}/{mode}/{target}"
                t0 = time.perf_counter()
                model = make_model(model_name).fit(model_input(train, mode), train.y)
                results = results_frame(test, model.predict(model_input(test, mode)))
                m = per_bearing_metrics(results)
                print(f"{tag:<24} ({time.perf_counter() - t0:4.1f} s)  " + "  ".join(
                    f"{b}: MAE {r.MAE:6.1f} RMSE {r.RMSE:6.1f} R2 {r.R2:6.2f}" for b, r in m.iterrows()))
                tables.append(m.assign(model=model_name, input=mode, target=target))
                stage_tables.append(error_by_life_stage(results).assign(model=tag))
                if model_name == "rf" and mode == "window":
                    plot_rul_predictions(results, FIGURES_DIR / f"rul_rf_window_{target}.png",
                                         title=f"Random Forest, flattened window, {target} target")
                    plot_error_vs_life(results, FIGURES_DIR / f"error_vs_life_rf_window_{target}.png",
                                       title=f"Random Forest, flattened window, {target} target")

    table = pd.concat(tables).rename_axis("bearing").reset_index()
    table = table[["model", "input", "target", "bearing", "MAE", "RMSE", "R2", "n"]]
    out = REPORTS_DIR / "baselines_learning_split.csv"
    table.to_csv(out, index=False, float_format="%.3f")

    stages = pd.concat(stage_tables).rename_axis("bearing").reset_index()
    stages.to_csv(REPORTS_DIR / "baselines_error_by_life_stage.csv", index=False, float_format="%.1f")
    print(f"\nwrote {out.relative_to(PROJECT_ROOT)} and baselines_error_by_life_stage.csv; "
          f"figures in {FIGURES_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
