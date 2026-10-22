import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from src.benchmark import CUTS, STABILITY_SHIFTS, cutpoint_table, stable_phm_score  # noqa: E402


def _results(pred_offset_min: float = 0.0) -> pd.DataFrame:
    """Every test bearing, snapshots up to its cut, true RUL counting down to 100 min at the cut."""
    parts = []
    for b, cut in CUTS.items():
        idx = np.arange(cut - 1 - max(STABILITY_SHIFTS), cut)
        true = 100 + (cut - 1 - idx) / 6
        parts.append(pd.DataFrame({"bearing": b, "snapshot_idx": idx, "rul_true_min": true,
                                   "rul_pred_min": true + pred_offset_min}))
    return pd.concat(parts, ignore_index=True)


def test_cutpoint_uses_last_snapshot_before_cut():
    t = cutpoint_table(_results())
    assert len(t) == len(CUTS)
    np.testing.assert_allclose(t.act_rul_s, 100 * 60)
    np.testing.assert_allclose(t.A, 1.0)


def test_stable_score_perfect_and_late():
    assert stable_phm_score(_results()) == pytest.approx(1.0)
    late, early = stable_phm_score(_results(+10)), stable_phm_score(_results(-10))
    assert late < early < 1.0  # overestimating RUL is penalised harder
