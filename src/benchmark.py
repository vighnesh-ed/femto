"""Official PHM 2012 protocol: train on the 6 learning bearings, predict the 11 test bearings.

    python -m src.benchmark                            # every model in MODELS
    python -m src.benchmark --models rf_window cnn_lstm --seeds 1

For each test bearing the model only sees the first len(Test_set) snapshots (the "cut"); its
prediction at the last window before the cut is scored against the true RUL there with the
official PHM 2012 score. Windows are causal and features are per-snapshot, so that prediction is
exactly the full-curve prediction at snapshot cut - 1. The whole Full_Test_Set run is also
predicted window by window for MAE / RMSE / R^2 and early-vs-late-life error.

Nothing from Full_Test_Set is used for training, scaling, the RUL cap or choosing epochs.

Adding a model (e.g. a teammate's GRU): write `make(n_features) -> nn.Module` and register
    MODELS["gru"] = deep_model(lambda n: GRURegressor(n))
Sklearn-style models go through sklearn_model(...). Every model returns one prediction array
per seed; metrics are averaged over seeds.

Ground truth: true RUL at the cut = len(Full_Test_Set) - len(Test_set) snapshots, confirmed
against the raw timestamps for all 11 bearings. Bearing1_4 is anomalous: its RMS jumps ~25x at
3.05 h, before its 3.16 h cut, and the recording then continues for 48 min at 5-12 g (no learning
bearing runs that long at such levels), so its data-derived RUL of 2890 s is doubtful. Published
tables have been cited with 339 s, which we could not verify. The summary therefore also gives
the PHM score without Bearing1_4.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from typing import Callable

import numpy as np
import pandas as pd
from torch import nn

from src.evaluate import (FIGURES_DIR, SNAPSHOTS_TO_MIN, error_by_life_stage, per_bearing_metrics,
                          phm_score, plot_rul_predictions, results_frame)
from src.features import load_features
from src.labels import detect_knee
from src.loader import EXPECTED_ACC_COUNTS, LEARNING_BEARINGS, PROJECT_ROOT, SNAPSHOT_PERIOD_S, TEST_BEARINGS
from src.models.baselines import make_model, model_input
from src.models.cnn_lstm import CNNLSTM
from src.models.training import TrainConfig, fit, predict, select_epochs, use_cpu_threads
from src.windows import Split, make_split

REPORTS_DIR = PROJECT_ROOT / "reports"
CUTS = {b: EXPECTED_ACC_COUNTS["Test_set"][b] for b in TEST_BEARINGS}  # snapshots the model may see
ANOMALOUS = "Bearing1_4"
STABILITY_SHIFTS = range(0, 31)  # the official cut and each snapshot of the 5 minutes before it

# A model: (split, train_dfs, target, seeds) -> list of predictions for split.test, in snapshots.
ModelRunner = Callable[[Split, dict[str, pd.DataFrame], str, int], list[np.ndarray]]


def constant_mean() -> ModelRunner:
    """Floor: always predict the mean training target. A model that cannot beat this has not
    learned anything usable about RUL."""
    def run(split, train_dfs, target, seeds):
        return [np.full(len(split.test), split.train.y.mean())]
    return run


def sklearn_model(name: str, mode: str) -> ModelRunner:
    def run(split, train_dfs, target, seeds):
        model = make_model(name).fit(model_input(split.train, mode), split.train.y)
        return [model.predict(model_input(split.test, mode))]
    return run


def deep_model(make: Callable[[int], nn.Module], max_epochs: int = 30) -> ModelRunner:
    """Epochs chosen by inner leave-one-bearing-out on the training bearings, then one model
    per seed trained on all of them (src/models/training.py). Search bound 30: on the default
    split the held-out error bottoms out within the first 5 epochs and only rises after."""
    def run(split, train_dfs, target, seeds):
        def model_fn():
            return make(split.train.X.shape[2])
        cfg = TrainConfig(epochs=max_epochs)
        epochs, _ = select_epochs(model_fn, train_dfs, target, cfg)
        print(f"    {epochs} epochs chosen")
        return [predict(fit(model_fn, split.train, split.y_scale, replace(cfg, epochs=epochs, seed=s))[0],
                        split.test.X, split.y_scale) for s in range(seeds)]
    return run


MODELS: dict[str, ModelRunner] = {
    "constant_mean": constant_mean(),
    "ridge_row": sklearn_model("ridge", "row"),
    "ridge_window": sklearn_model("ridge", "window"),
    "rf_row": sklearn_model("rf", "row"),
    "rf_window": sklearn_model("rf", "window"),
    "cnn_lstm": deep_model(lambda n: CNNLSTM(n)),
    "cnn_lstm_attn": deep_model(lambda n: CNNLSTM(n, attention=True)),
}


def cutpoint_table(results: pd.DataFrame, shift: int = 0) -> pd.DataFrame:
    """One row per test bearing: prediction at the last window before the cut (moved `shift`
    snapshots earlier), PHM %-error and accuracy A. Er < 0 = late (overestimated RUL)."""
    rows = []
    for b, cut in CUTS.items():
        r = results[(results.bearing == b) & (results.snapshot_idx == cut - 1 - shift)]
        if len(r) != 1:
            raise ValueError(f"{b}: expected exactly one window ending at snapshot {cut - 1 - shift}")
        act, pred = r.rul_true_min.iloc[0] * 60, r.rul_pred_min.iloc[0] * 60
        rows.append({"bearing": b, "act_rul_s": act, "pred_rul_s": pred,
                     "er_pct": 100 * (act - pred) / act, "A": phm_score([act], [pred])})
    return pd.DataFrame(rows)


def stable_phm_score(results: pd.DataFrame) -> float:
    """Mean PHM score over the official cut and each of the 30 snapshots (5 min) before it.

    The official score uses ONE snapshot per bearing, which makes it very noisy: rf_row scored
    0.257 at the official cut but anywhere from 0.065 to 0.257 at the cuts just before it
    (mean 0.161, vs 0.156 for constant_mean). Compare models on this number.
    """
    scores = []
    for shift in STABILITY_SHIFTS:
        t = cutpoint_table(results, shift)
        scores.append(phm_score(t.act_rul_s, t.pred_rul_s))
    return float(np.mean(scores))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PHM 2012 test-set benchmark")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--target", default="piecewise", choices=["linear", "piecewise"])
    parser.add_argument("--seeds", type=int, default=3, help="seeds per deep model")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up even when output goes to a file

    use_cpu_threads()
    train_dfs = {b: load_features(b) for b in LEARNING_BEARINGS}
    test_dfs = {b: load_features(b) for b in TEST_BEARINGS}
    split = make_split(train_dfs, test_dfs, args.target)
    cap_txt = f", cap {split.cap * SNAPSHOTS_TO_MIN:.1f} min" if split.cap is not None else ""
    print(f"train: {len(train_dfs)} learning bearings ({len(split.train)} windows); "
          f"test: {len(test_dfs)} bearings ({len(split.test)} windows); target {args.target}{cap_txt}")
    # Knee detected on the full run (analysis only): is the bearing visibly degrading at its cut?
    degrading = {b: detect_knee(df) < CUTS[b] for b, df in test_dfs.items()}
    print(f"degrading at the cut: {[b for b, d in degrading.items() if d]}; "
          f"healthy-looking: {[b for b, d in degrading.items() if not d]}\n")

    cut_rows, curve_rows, stage_rows, summary = [], [], [], []
    for name in args.models:
        t0 = time.perf_counter()
        print(f"{name}:")
        preds = MODELS[name](split, train_dfs, args.target, args.seeds)
        scores, scores_excl, scores_stable, maes = [], [], [], []
        for seed, pred in enumerate(preds):
            results = results_frame(split.test, pred)
            cut = cutpoint_table(results)
            curves = per_bearing_metrics(results)
            scores.append(phm_score(cut.act_rul_s, cut.pred_rul_s))
            ok = cut.bearing != ANOMALOUS
            scores_excl.append(phm_score(cut.act_rul_s[ok], cut.pred_rul_s[ok]))
            scores_stable.append(stable_phm_score(results))
            maes.append(curves.loc["pooled", "MAE"])
            cut_rows.append(cut.assign(model=name, seed=seed, degrading_at_cut=cut.bearing.map(degrading)))
            curve_rows.append(curves.rename_axis("bearing").reset_index().assign(model=name, seed=seed))
            if seed == 0:
                stage_rows.append(error_by_life_stage(results).rename_axis("bearing").reset_index()
                                  .assign(model=name))
                plot_rul_predictions(results, FIGURES_DIR / f"benchmark_rul_{name}_{args.target}.png",
                                     title=f"{name} on the PHM 2012 test bearings ({args.target} target)",
                                     cuts=CUTS)
        late = (pd.concat(cut_rows[-len(preds):]).er_pct < 0).mean()
        summary.append({"model": name, "phm_score": np.mean(scores), "phm_score_std": np.std(scores),
                        "phm_score_5min_mean": np.mean(scores_stable),
                        f"phm_score_excl_{ANOMALOUS}": np.mean(scores_excl), "late_share": late,
                        "full_curve_MAE_min": np.mean(maes), "runs": len(preds)})
        print(f"    PHM score {np.mean(scores):.3f} +/- {np.std(scores):.3f}, mean over last 5 min "
              f"{np.mean(scores_stable):.3f} (excl. {ANOMALOUS}: {np.mean(scores_excl):.3f})   "
              f"full-curve MAE {np.mean(maes):.1f} min   late predictions {late:.0%}   "
              f"({time.perf_counter() - t0:.0f} s)")

    tag = args.target
    pd.concat(cut_rows).to_csv(REPORTS_DIR / f"benchmark_cutpoint_{tag}.csv", index=False, float_format="%.3f")
    pd.concat(curve_rows).to_csv(REPORTS_DIR / f"benchmark_full_curves_{tag}.csv", index=False, float_format="%.3f")
    pd.concat(stage_rows).to_csv(REPORTS_DIR / f"benchmark_life_stage_{tag}.csv", index=False, float_format="%.1f")
    summary = pd.DataFrame(summary)
    summary.to_csv(REPORTS_DIR / f"benchmark_summary_{tag}.csv", index=False, float_format="%.3f")

    cut = pd.concat(cut_rows).groupby(["model", "bearing"], sort=False)[["act_rul_s", "pred_rul_s", "A"]].mean()
    pd.set_option("display.width", 200)
    print("\nprediction at the cut, seconds (mean over seeds):")
    print(cut.pred_rul_s.unstack(0).round(0).assign(actual=cut.act_rul_s.unstack(0).iloc[:, 0]).to_string())
    print(f"\n{summary.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
