import numpy as np
import pandas as pd

from src.maintenance import classify, evaluate_policy, first_warning


def test_first_warning_needs_persistence():
    pred = np.array([50, 4, 50, 4, 4, 4, 4], dtype=float)
    assert first_warning(pred, threshold_min=5, persist=3) == 5  # one-off dip at 1 is ignored
    assert first_warning(pred, threshold_min=5, persist=1) == 1
    assert first_warning(np.full(5, 50.0), threshold_min=5) is None


def test_classify():
    assert classify(None) == "missed"
    assert classify(1.0) == "too late"
    assert classify(10.0) == "useful"
    assert classify(90.0) == "early"


def test_evaluate_policy_lead_time():
    true = np.arange(20, -1, -1, dtype=float)  # 20 .. 0 minutes
    res = pd.DataFrame({"bearing": "A", "snapshot_idx": np.arange(21), "rul_true_min": true,
                        "rul_pred_min": true})  # perfect predictions
    out = evaluate_policy(res, thresholds=[10], persist=3).iloc[0]
    assert out.lead_min == 7.0  # below 10 at 9, 8, 7 -> fires on the third, at 7 min
    assert out.outcome == "useful"
