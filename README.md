# Bearing RUL prediction — FEMTO-ST / PHM 2012

Remaining Useful Life of rolling-element bearings from raw vibration. The results, data quirks and
corrections to the original plan are written up in [reports/REPORT.md](reports/REPORT.md).

The original FEMTO-ST download page is offline. The data is the IEEE PHM 2012 Prognostic Data
Challenge release (17 bearings, 3 operating conditions), mirrored at
[Lucky-Loek/ieee-phm-2012-data-challenge-dataset](https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset).

## Setup

```bash
pip install -r requirements.txt
# CPU-only PyTorch is enough (models are tiny) and a much smaller download:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Most people do **not** need the raw data: the feature CSVs for all 17 bearings in
`data/features/` are committed. Only whoever rebuilds features needs the raw data (about 450 MB for
the learning set plus `Full_Test_Set`):

```bash
git clone --filter=blob:none --no-checkout \
  https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset.git data/raw
cd data/raw
git sparse-checkout init --cone
git sparse-checkout set Learning_set Full_Test_Set
git checkout master
```

## Run

```bash
python -m src.features                        # raw -> data/features/<Bearing>.csv (cached)
python -m src.features --split Full_Test_Set  # the 11 test bearings
python -m src.evaluate                        # degradation curves + health indicators -> reports/figures/
python -m src.models.baselines                # Ridge / RF on the default split (train 1_x,2_x -> test 3_1,3_2)
python -m src.models.cnn_lstm                 # CNN-LSTM (+attention, MC-dropout) on the default split
python -m src.models.learning_split           # MLP / LSTM / GRU / 1D-CNN on the same split
python -m src.benchmark                       # official PHM 2012 protocol: 6 learning -> 11 test bearings
python -m src.maintenance                     # warning rule replayed on the test bearings
python -m pytest                              # 39 tests
```

Outputs go to `reports/` (CSV tables) and `reports/figures/`.

## Code

| Stage | Module | Output |
|---|---|---|
| 1 load | `src/loader.py` | `(n_snapshots, 2560, 2)` array; asserts shape and file counts |
| 2 features | `src/features.py` | `data/features/<Bearing>.csv` — **the team contract**, see `FEATURE_COLUMNS` |
| 3 labels | `src/labels.py` | knee point, health indicator, linear / piecewise RUL |
| 4 windows | `src/windows.py` | `make_split()`: `(W=30, 20)` windows, scaler + RUL cap fit on train bearings only |
| 5 baselines | `src/models/baselines.py` | Ridge, Random Forest |
| 5 deep | `src/models/training.py` | **shared** PyTorch loop: `fit`, `predict`, `select_epochs`, `mc_dropout_predict` |
| 5 deep | `src/models/cnn_lstm.py` | 1D-CNN-LSTM, optional attention pooling |
| 5 deep | `src/models/mlp.py`, `src/models/recurrent.py`, `src/models/cnn1d.py` | MLP, LSTM / GRU and 1D-CNN (architectures from the group's original Keras notebooks, ported to the shared PyTorch pipeline) |
| eval | `src/evaluate.py` | MAE / RMSE / R² in minutes, PHM 2012 score, coverage, report plots |
| eval | `src/benchmark.py` | every model on the official test protocol, PHM score at the Test_set cut |
| decision | `src/maintenance.py` | RUL < T warning rule: lead time, missed / late / early warnings |

## Adding a model

`mlp`, `lstm`, `gru` and `cnn1d` are already registered (`python -m src.benchmark --models lstm gru`).
To add another, use the shared pieces so results are comparable — same windows, targets, epoch rule
and metrics:

1. Write an `nn.Module` mapping a float32 batch `(B, 30, 20)` to `(B,)` (RUL / y_scale), as in
   `src/models/recurrent.py`.
2. Train it with `src/models/training.py` — see `src/models/cnn_lstm.py` `main()` for the full
   pattern (`make_split` → `select_epochs` → `fit` per seed → `predict` → `per_bearing_metrics`).
   Call `use_cpu_threads()` first on CPU: the default thread count makes these tiny models 10x slower.
3. Register it in `src/benchmark.py`:
   `DEEP_MODELS["gru"] = lambda n: RecurrentRegressor(n, "gru")` (the benchmark, `src.maintenance` and
   `src.models.learning_split` all read it), then
   `python -m src.benchmark --models gru rf_row`. Note: the benchmark rewrites `reports/benchmark_*.csv`
   with only the models you ask for, so run the full registry (no `--models`) before committing results.

Rules that matter: split by bearing, never by row; never fit anything (scaler, RUL cap, epoch count)
on test bearings; never train on `Full_Test_Set`; choose epochs with `select_epochs` (training
bearings only), never by looking at test results; report per-bearing metrics, not just a pooled mean.
