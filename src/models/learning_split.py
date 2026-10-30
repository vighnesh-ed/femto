"""Validation-split run for any registered deep model: train on 1_1, 1_2, 2_1, 2_2, test on 3_1, 3_2.

    python -m src.models.learning_split                       # MLP, LSTM, GRU, 1D-CNN, 3 seeds
    python -m src.models.learning_split --models lstm --quick # 1 seed, 10 max epochs: smoke test

The same protocol as src/models/cnn_lstm.py (epochs chosen by inner leave-one-bearing-out on the
training bearings, then one model per seed trained on all of them), for the models in
src/benchmark.py DEEP_MODELS. No MC-dropout band: these models carry batch norm, which
mc_dropout_predict would put in training mode.

Writes reports/deep_learning_split_<tag>.csv (+ _per_seed), deep_error_by_life_stage_<tag>.csv,
<model>_epoch_selection_<tag>.csv and the rul_ / error_vs_life_ / epochs_ figures.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

import pandas as pd

from src.benchmark import DEEP_MODELS
from src.evaluate import (FIGURES_DIR, error_by_life_stage, per_bearing_metrics, plot_epoch_selection,
                          plot_error_vs_life, plot_rul_predictions, results_frame)
from src.features import load_features
from src.loader import PROJECT_ROOT
from src.models.training import TrainConfig, fit, predict, select_epochs, use_cpu_threads
from src.windows import DEFAULT_TEST, DEFAULT_TRAIN, make_split

REPORTS_DIR = PROJECT_ROOT / "reports"
NEW_MODELS = ["mlp", "lstm", "gru", "cnn1d"]  # the CNN-LSTM variants have their own runner


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Deep models on the default bearing split")
    parser.add_argument("--models", nargs="+", default=NEW_MODELS, choices=list(DEEP_MODELS))
    parser.add_argument("--target", default="piecewise", choices=["linear", "piecewise"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=60, help="upper bound for epoch selection")
    parser.add_argument("--quick", action="store_true", help="1 seed, 10 max epochs: smoke test")
    args = parser.parse_args(argv)
    if args.quick:
        args.seeds, args.max_epochs = 1, 10
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up even when output goes to a file

    use_cpu_threads()
    dfs = {b: load_features(b) for b in DEFAULT_TRAIN + DEFAULT_TEST}
    train_dfs = {b: dfs[b] for b in DEFAULT_TRAIN}
    split = make_split(train_dfs, {b: dfs[b] for b in DEFAULT_TEST}, args.target)
    n_features = split.train.X.shape[2]
    cfg = TrainConfig(epochs=args.max_epochs)
    tag = f"{args.target}{'_quick' if args.quick else ''}"
    print(f"train {DEFAULT_TRAIN} -> test {DEFAULT_TEST}, target {args.target}, "
          f"{len(split.train)} train windows, y_scale {split.y_scale:.0f} snapshots\n")

    metric_tables, stage_tables = [], []
    for name in args.models:
        def model_fn(name=name):
            return DEEP_MODELS[name](n_features)

        t0 = time.perf_counter()
        epochs, curves = select_epochs(model_fn, train_dfs, args.target, cfg)
        curves.to_csv(REPORTS_DIR / f"{name}_epoch_selection_{tag}.csv", float_format="%.3f")
        plot_epoch_selection(curves, epochs, FIGURES_DIR / f"epochs_{name}_{tag}.png",
                             title=f"{name}: inner leave-one-bearing-out on training bearings")
        print(f"{name}: {epochs} epochs chosen ({time.perf_counter() - t0:.0f} s)")

        for seed in range(args.seeds):
            t0 = time.perf_counter()
            model, _ = fit(model_fn, split.train, split.y_scale, replace(cfg, epochs=epochs, seed=seed))
            results = results_frame(split.test, predict(model, split.test.X, split.y_scale))
            m = per_bearing_metrics(results)
            metric_tables.append(m.assign(model=name, seed=seed, epochs=epochs))
            print(f"  seed {seed} ({time.perf_counter() - t0:4.1f} s)  " + "  ".join(
                f"{b}: MAE {r.MAE:6.1f} RMSE {r.RMSE:6.1f} R2 {r.R2:6.2f}" for b, r in m.iterrows()))
            if seed == 0:
                stage_tables.append(error_by_life_stage(results).assign(model=name))
                title = f"{name}, {args.target} target (seed 0)"
                plot_rul_predictions(results, FIGURES_DIR / f"rul_{name}_{tag}.png", title=title)
                plot_error_vs_life(results, FIGURES_DIR / f"error_vs_life_{name}_{tag}.png", title=title)

    per_seed = pd.concat(metric_tables).rename_axis("bearing").reset_index()
    per_seed.to_csv(REPORTS_DIR / f"deep_learning_split_{tag}_per_seed.csv", index=False, float_format="%.3f")
    summary = per_seed.groupby(["model", "bearing"], sort=False)[["MAE", "RMSE", "R2"]].agg(["mean", "std"])
    summary.to_csv(REPORTS_DIR / f"deep_learning_split_{tag}.csv", float_format="%.3f")
    stages = pd.concat(stage_tables).rename_axis("bearing").reset_index()
    stages.to_csv(REPORTS_DIR / f"deep_error_by_life_stage_{tag}.csv", index=False, float_format="%.1f")

    pd.set_option("display.width", 160)
    print(f"\nmean +/- std over {args.seeds} seed(s), minutes:\n{summary.round(2)}")
    print(f"\nMAE by life stage (seed 0):\n{stages.round(1).to_string(index=False)}")


if __name__ == "__main__":
    main()
