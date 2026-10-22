"""Stage 5c: 1D-CNN-LSTM for RUL, optional attention pooling, MC-dropout uncertainty.

    python -m src.models.cnn_lstm              # default split, both variants, 3 seeds
    python -m src.models.cnn_lstm --quick      # 1 seed, 10 epochs: smoke test

Input: one window of W=30 snapshots x 20 features (src/windows.py).

  Conv1d(20->32, k=3) -> ReLU -> Conv1d(32->64, k=3) -> ReLU -> Dropout   local patterns, ~50 s
  LSTM(64 -> 64)                                                          trend across the window
  pool: last hidden state, or additive attention over all W steps
  Dropout -> Linear(64 -> 1)                                              RUL / y_scale

The attention variant learns a weight per snapshot in the window; the weights are returned by
forward(..., return_attention=True) for plotting.

Protocol (main): epochs chosen by inner leave-one-bearing-out on the training bearings
(training.select_epochs), then retrained on all training bearings with several seeds. Metrics
use the deterministic prediction (dropout off); the MC-dropout std of seed 0 adds a +/-2 std
band to the plots and an interval-coverage table.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.evaluate import (FIGURES_DIR, error_by_life_stage, interval_coverage, per_bearing_metrics,
                          plot_epoch_selection, plot_error_vs_life, plot_rul_predictions, results_frame)
from src.features import load_features
from src.loader import PROJECT_ROOT, SNAPSHOT_PERIOD_S
from src.models.training import (TrainConfig, as_tensor, fit, mc_dropout_predict, predict, select_epochs,
                                 use_cpu_threads)
from src.windows import DEFAULT_TEST, DEFAULT_TRAIN, WindowSet, make_split

REPORTS_DIR = PROJECT_ROOT / "reports"
VARIANTS = {"cnn_lstm": False, "cnn_lstm_attn": True}  # name -> attention


class CNNLSTM(nn.Module):
    def __init__(self, n_features: int, conv_channels: tuple[int, ...] = (32, 64), kernel_size: int = 3,
                 hidden: int = 64, dropout: float = 0.2, attention: bool = False):
        super().__init__()
        layers, c_in = [], n_features
        for c_out in conv_channels:
            layers += [nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2), nn.ReLU()]
            c_in = c_out
        self.conv = nn.Sequential(*layers, nn.Dropout(dropout))
        self.lstm = nn.LSTM(c_in, hidden, batch_first=True)
        self.attention = (nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False))
                          if attention else None)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, W, F) -> (B, W, C)
        h, _ = self.lstm(z)                                # (B, W, H)
        if self.attention is None:
            pooled, alpha = h[:, -1], None
        else:
            alpha = torch.softmax(self.attention(h).squeeze(-1), dim=1)  # (B, W), sums to 1
            pooled = (alpha.unsqueeze(-1) * h).sum(dim=1)
        out = self.head(pooled).squeeze(-1)
        return (out, alpha) if return_attention else out


def plot_attention(model: CNNLSTM, ws: WindowSet, bearing: str, path) -> None:
    """Heatmap of the attention weight on each snapshot of the window, for every window of one
    bearing: shows which part of the last 5 minutes the model looks at as failure approaches."""
    sel = ws.bearing == bearing
    model.eval()
    with torch.no_grad():
        _, alpha = model(as_tensor(ws.X[sel]), return_attention=True)
    alpha = alpha.cpu().numpy()
    hours = ws.snapshot_idx[sel] * SNAPSHOT_PERIOD_S / 3600
    w = alpha.shape[1]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(alpha.T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[hours[0], hours[-1], -(w - 1) - 0.5, 0.5])
    fig.colorbar(im, ax=ax, label="attention weight")
    ax.set_xlabel("operating time [h]")
    ax.set_ylabel("snapshot in window\n(0 = newest)")
    ax.set_title(f"CNN-LSTM attention over the window, {bearing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CNN-LSTM (+attention) on the default bearing split")
    parser.add_argument("--target", default="piecewise", choices=["linear", "piecewise"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=60, help="upper bound for epoch selection")
    parser.add_argument("--mc-samples", type=int, default=50)
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

    metric_tables, stage_tables, coverage_tables = [], [], []
    for name, attention in VARIANTS.items():
        def model_fn(attention=attention):
            return CNNLSTM(n_features, attention=attention)

        t0 = time.perf_counter()
        epochs, curves = select_epochs(model_fn, train_dfs, args.target, cfg)
        curves.to_csv(REPORTS_DIR / f"{name}_epoch_selection_{tag}.csv", float_format="%.3f")
        plot_epoch_selection(curves, epochs, FIGURES_DIR / f"epochs_{name}_{tag}.png",
                             title=f"{name}: inner leave-one-bearing-out on training bearings")
        print(f"{name}: {epochs} epochs chosen ({time.perf_counter() - t0:.0f} s)")

        for seed in range(args.seeds):
            t0 = time.perf_counter()
            model, _ = fit(model_fn, split.train, split.y_scale, replace(cfg, epochs=epochs, seed=seed))
            pred = predict(model, split.test.X, split.y_scale)
            results = results_frame(split.test, pred)
            m = per_bearing_metrics(results)
            metric_tables.append(m.assign(model=name, seed=seed, epochs=epochs))
            print(f"  seed {seed} ({time.perf_counter() - t0:4.1f} s)  " + "  ".join(
                f"{b}: MAE {r.MAE:6.1f} RMSE {r.RMSE:6.1f} R2 {r.R2:6.2f}" for b, r in m.iterrows()))
            if seed != 0:
                continue
            _, std = mc_dropout_predict(model, split.test.X, split.y_scale, args.mc_samples)
            banded = results_frame(split.test, pred, std)
            coverage_tables.append(interval_coverage(banded).assign(model=name))
            stage_tables.append(error_by_life_stage(results).assign(model=name))
            title = f"{name}, {args.target} target (seed 0, ±2σ MC-dropout)"
            plot_rul_predictions(banded, FIGURES_DIR / f"rul_{name}_{tag}.png", title=title)
            plot_error_vs_life(results, FIGURES_DIR / f"error_vs_life_{name}_{tag}.png", title=title)
            if attention:
                for bearing in DEFAULT_TEST:
                    plot_attention(model, split.test, bearing, FIGURES_DIR / f"attention_{name}_{bearing}_{tag}.png")

    per_seed = pd.concat(metric_tables).rename_axis("bearing").reset_index()
    per_seed.to_csv(REPORTS_DIR / f"cnn_lstm_learning_split_{tag}_per_seed.csv", index=False, float_format="%.3f")
    summary = per_seed.groupby(["model", "bearing"], sort=False)[["MAE", "RMSE", "R2"]].agg(["mean", "std"])
    summary.to_csv(REPORTS_DIR / f"cnn_lstm_learning_split_{tag}.csv", float_format="%.3f")
    coverage = pd.concat(coverage_tables).rename_axis("bearing").reset_index()
    coverage.to_csv(REPORTS_DIR / f"cnn_lstm_mc_coverage_{tag}.csv", index=False, float_format="%.3f")
    stages = pd.concat(stage_tables).rename_axis("bearing").reset_index()
    stages.to_csv(REPORTS_DIR / f"cnn_lstm_error_by_life_stage_{tag}.csv", index=False, float_format="%.1f")

    pd.set_option("display.width", 160)
    print(f"\nmean +/- std over {args.seeds} seed(s), minutes:\n{summary.round(2)}")
    print(f"\nMC-dropout +/-2 std coverage (target ~0.95):\n{coverage.round(3).to_string(index=False)}")
    print(f"\nMAE by life stage (seed 0):\n{stages.round(1).to_string(index=False)}")


if __name__ == "__main__":
    main()
