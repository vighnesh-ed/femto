import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.evaluate import interval_coverage, results_frame  # noqa: E402
from src.models.cnn1d import CNN1D  # noqa: E402
from src.models.cnn_lstm import CNNLSTM  # noqa: E402
from src.models.mlp import MLP  # noqa: E402
from src.models.recurrent import RecurrentRegressor  # noqa: E402
from src.models.training import TrainConfig, fit, mc_dropout_predict, predict, select_epochs  # noqa: E402
from src.windows import WindowSet, make_split  # noqa: E402
from tests.test_labels_windows import _fake_bearing  # noqa: E402


def _toy_windows(n=256, w=10, f=4, seed=0) -> WindowSet:
    """RUL is a linear function of the last step's first feature: learnable in a few epochs."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, w, f))
    y = 50 + 20 * X[:, -1, 0]
    return WindowSet(X=X, y=y, rul_true=y, bearing=np.full(n, "A"), snapshot_idx=np.arange(n))


@pytest.mark.parametrize("attention", [False, True])
def test_cnn_lstm_shapes(attention):
    model = CNNLSTM(n_features=4, attention=attention)
    out, alpha = model(torch.randn(8, 10, 4), return_attention=True)
    assert out.shape == (8,)
    if attention:
        assert alpha.shape == (8, 10)
        np.testing.assert_allclose(alpha.sum(dim=1).detach().numpy(), 1.0, rtol=1e-5)
    else:
        assert alpha is None


@pytest.mark.parametrize("make", [lambda: RecurrentRegressor(4, "lstm"), lambda: RecurrentRegressor(4, "gru"),
                                  lambda: CNN1D(4, window=10), lambda: MLP(4, window=10)])
def test_ported_models_shapes_and_learn(make):
    assert make()(torch.randn(8, 10, 4)).shape == (8,)
    ws = _toy_windows()
    model, hist = fit(make, ws, y_scale=100, cfg=TrainConfig(epochs=15, batch_size=32, lr=3e-3))
    assert hist.train_mse.iloc[-1] < 0.5 * hist.train_mse.iloc[0]
    assert predict(model, ws.X, 100).shape == (len(ws.y),)


def test_fit_learns_and_is_deterministic():
    ws = _toy_windows()
    cfg = TrainConfig(epochs=15, batch_size=32, lr=3e-3)
    model_a, hist = fit(lambda: CNNLSTM(4, hidden=16), ws, y_scale=100, cfg=cfg, val=ws)
    model_b, _ = fit(lambda: CNNLSTM(4, hidden=16), ws, y_scale=100, cfg=cfg)
    assert hist.train_mse.iloc[-1] < 0.5 * hist.train_mse.iloc[0]
    assert {"epoch", "train_mse", "val_rmse_min"} <= set(hist.columns)
    np.testing.assert_allclose(predict(model_a, ws.X, 100), predict(model_b, ws.X, 100))


def test_mc_dropout_spread_and_restores_eval_mode():
    ws = _toy_windows(n=64)
    model = CNNLSTM(4, hidden=16, dropout=0.3)
    mean, std = mc_dropout_predict(model, ws.X, y_scale=100, n_samples=20)
    assert mean.shape == std.shape == (64,)
    assert (std > 0).all()
    assert not any(m.training for m in model.modules())  # dropout switched back off
    np.testing.assert_allclose(predict(model, ws.X, 100), predict(model, ws.X, 100))


def test_select_epochs_uses_only_training_bearings():
    train = {f"B{i}": _fake_bearing(150, knee=100, seed=i) for i in range(3)}
    cfg = TrainConfig(epochs=3, batch_size=64)
    best, curves = select_epochs(lambda: CNNLSTM(20, hidden=8), train, "piecewise", cfg, window=10)
    assert 1 <= best <= 3
    assert list(curves.columns) == ["B0", "B1", "B2", "mean"]
    assert list(curves.index) == [1, 2, 3]


def test_make_split_rejects_overlap_and_fits_on_train():
    a, b = _fake_bearing(200, knee=150), _fake_bearing(200, knee=50, seed=1)
    with pytest.raises(ValueError, match="both train and test"):
        make_split({"A": a}, {"A": a})
    split = make_split({"A": a}, {"B": b}, "piecewise")
    assert split.cap == pytest.approx(200 - 1 - 150, abs=30)  # from A's knee, not B's
    assert split.y_scale == split.cap
    assert make_split({"A": a}, {"B": b}, "linear").y_scale == 199 - 29  # longest train label


def test_results_frame_and_coverage():
    ws = WindowSet(X=np.zeros((4, 1, 1)), y=np.zeros(4), rul_true=np.array([6.0, 6, 6, 6]),
                   bearing=np.array(["A", "A", "B", "B"]), snapshot_idx=np.array([0, 1, 0, 1]))
    res = results_frame(ws, pred=np.array([-6.0, 6, 6, 18]), std=np.array([6.0, 6, 6, 1]))
    assert res.rul_pred_min.min() == 0  # negative prediction clipped
    cov = interval_coverage(res)
    assert cov.loc["A", "coverage"] == 1.0  # |0-1| <= 2*1 and |1-1| <= 2*1 (minutes)
    assert cov.loc["B", "coverage"] == 0.5  # |3-1| = 2 > 2*(1/6)
    assert isinstance(cov, pd.DataFrame)
