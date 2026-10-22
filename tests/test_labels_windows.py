import numpy as np
import pandas as pd
import pytest

from src.evaluate import phm_score
from src.features import FEATURE_COLUMNS, SIGNAL_FEATURES
from src.labels import detect_knee, fit_rul_cap, rul_linear, rul_piecewise
from src.windows import FeatureScaler, build_windows, sliding_windows


def _fake_bearing(n, knee=None, seed=0):
    """Feature frame with flat noisy RMS that steps up 5x at `knee`."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(1.0, 2.0, size=(n, len(SIGNAL_FEATURES)))
    rms = 0.4 + 0.01 * rng.standard_normal(n)
    if knee is not None:
        rms[knee:] *= 5
    df = pd.DataFrame(data, columns=SIGNAL_FEATURES)
    df["rms_h"] = df["rms_v"] = rms
    df.insert(0, "elapsed_s", np.arange(n) * 10)
    df.insert(0, "snapshot_idx", np.arange(n))
    return df[FEATURE_COLUMNS]


def test_rul_labels():
    assert rul_linear(5).tolist() == [4, 3, 2, 1, 0]
    assert rul_piecewise(5, cap=2).tolist() == [2, 2, 2, 1, 0]


def test_knee_found_near_step():
    knee = detect_knee(_fake_bearing(600, knee=400))
    assert 400 <= knee <= 400 + 30  # within one smoothing window


def test_no_degradation_returns_last_index():
    assert detect_knee(_fake_bearing(600)) == 599


def test_cap_is_median_post_knee_life():
    dfs = [_fake_bearing(600, knee=400, seed=s) for s in range(3)]
    cap = fit_rul_cap(dfs)
    assert 600 - 1 - 430 <= cap <= 600 - 1 - 400


def test_sliding_windows_shape_and_order():
    x = np.arange(20, dtype=float).reshape(10, 2)
    w = sliding_windows(x, window=4)
    assert w.shape == (7, 4, 2)
    np.testing.assert_array_equal(w[0], x[0:4])
    np.testing.assert_array_equal(w[-1], x[6:10])


def test_windows_labelled_at_last_step():
    train = {"A": _fake_bearing(100), "B": _fake_bearing(80, seed=1)}
    scaler = FeatureScaler().fit(list(train.values()))
    ws = build_windows(train, scaler, window=30)
    assert len(ws) == (100 - 29) + (80 - 29)
    assert ws.snapshot_idx[0] == 29 and ws.rul_true[0] == 100 - 1 - 29
    np.testing.assert_allclose(ws.X[0, -1], scaler.transform(train["A"])[29])
    capped = build_windows(train, scaler, window=30, cap=10)
    assert capped.y.max() == 10 and capped.rul_true.max() == 70


def test_scaler_uses_train_bearings_only():
    train, test = _fake_bearing(100), _fake_bearing(100, seed=1) * 0 + 50
    scaler = FeatureScaler().fit([train])
    z_train = scaler.transform(train)
    np.testing.assert_allclose(z_train.mean(axis=0), 0, atol=1e-9)
    assert scaler.transform(test).mean() > 3  # test stats did not leak into the fit


def test_phm_score():
    assert phm_score([100, 50], [100, 50]) == pytest.approx(1.0)
    late, early = phm_score([100], [110]), phm_score([100], [90])
    assert late < early  # overestimating RUL is penalised harder
    with pytest.raises(ValueError):
        phm_score([0], [1])
