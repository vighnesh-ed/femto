"""Maintenance-decision prototype: turn RUL predictions into a replace-the-bearing warning.

    python -m src.maintenance                       # RF (row) + constant floor, 11 test bearings
    python -m src.maintenance --models rf_row cnn_lstm

Rule: warn when the predicted RUL has stayed below a threshold T for `persist` consecutive
snapshots (persistence suppresses one-off dips). Each test bearing is replayed over its whole
recorded run (Full_Test_Set) as if live, and the first warning is classified:

  missed      no warning before failure                         -> unplanned breakdown
  too late    warning with less lead time than `min_lead_min`   -> not enough time to act
  early       warning more than `max_lead_min` before failure   -> bearing replaced too soon
  useful      lead time within [min_lead_min, max_lead_min]

Lead time = true RUL at the first warning. Predictions come from models trained on the
6 learning bearings only (src/benchmark.py MODELS), so the test bearings stay unseen.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.benchmark import MODELS
from src.evaluate import SNAPSHOTS_TO_MIN, results_frame
from src.features import load_features
from src.loader import LEARNING_BEARINGS, PROJECT_ROOT, TEST_BEARINGS
from src.models.training import use_cpu_threads
from src.windows import make_split

REPORTS_DIR = PROJECT_ROOT / "reports"
THRESHOLDS_MIN = [5, 10, 20, 30]
PERSIST = 3            # 30 s below threshold before warning
MIN_LEAD_MIN = 2.0     # less warning than this is too late to schedule a stop
MAX_LEAD_MIN = 60.0    # more than this throws away too much useful life


def first_warning(rul_pred_min: np.ndarray, threshold_min: float, persist: int = PERSIST) -> int | None:
    """Index of the snapshot at which the warning fires (end of the first run of `persist`
    consecutive predictions below threshold), or None if it never fires."""
    below = np.asarray(rul_pred_min) < threshold_min
    run = np.convolve(below, np.ones(persist, dtype=int), mode="valid") == persist
    hits = np.flatnonzero(run)
    return int(hits[0] + persist - 1) if hits.size else None


def classify(lead_min: float | None, min_lead: float = MIN_LEAD_MIN, max_lead: float = MAX_LEAD_MIN) -> str:
    if lead_min is None:
        return "missed"
    if lead_min < min_lead:
        return "too late"
    if lead_min > max_lead:
        return "early"
    return "useful"


def evaluate_policy(results: pd.DataFrame, thresholds=THRESHOLDS_MIN, persist: int = PERSIST) -> pd.DataFrame:
    """One row per (bearing, threshold): warning time, lead time and outcome."""
    rows = []
    for b, g in results.groupby("bearing", sort=False):
        g = g.sort_values("snapshot_idx")
        for t in thresholds:
            i = first_warning(g.rul_pred_min.to_numpy(), t, persist)
            lead = None if i is None else float(g.rul_true_min.iloc[i])
            rows.append({"bearing": b, "threshold_min": t,
                         "warn_at_h": None if i is None else g.snapshot_idx.iloc[i] * SNAPSHOTS_TO_MIN / 60,
                         "lead_min": lead, "outcome": classify(lead)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=["constant_mean", "rf_row"], choices=list(MODELS))
    parser.add_argument("--target", default="piecewise", choices=["linear", "piecewise"])
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    use_cpu_threads()
    train_dfs = {b: load_features(b) for b in LEARNING_BEARINGS}
    split = make_split(train_dfs, {b: load_features(b) for b in TEST_BEARINGS}, args.target)

    tables = []
    for name in args.models:
        print(f"{name}:")
        pred = MODELS[name](split, train_dfs, args.target, 1)[0]  # seed 0
        policy = evaluate_policy(results_frame(split.test, pred)).assign(model=name)
        tables.append(policy)
        counts = policy.groupby(["threshold_min", "outcome"]).size().unstack(fill_value=0)
        counts["median_lead_min"] = policy.groupby("threshold_min").lead_min.median()
        print(counts.to_string(), "\n")

    out = pd.concat(tables)
    out.to_csv(REPORTS_DIR / f"maintenance_policy_{args.target}.csv", index=False, float_format="%.2f")
    print(f"per-bearing detail: reports/maintenance_policy_{args.target}.csv")


if __name__ == "__main__":
    main()
