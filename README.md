# Bearing RUL prediction — FEMTO-ST / PHM 2012


## Setup

```bash
pip install -r requirements.txt
# CPU-only PyTorch is enough (models are tiny) and a much smaller download:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Most people do **not** need the raw data: the feature CSVs for all 17 bearings in
`data/features/` are committed. Only whoever rebuilds features needs the raw sparse checkout
(command in CLAUDE.md).

## Run

```bash
python -m src.features                        # raw -> data/features/<Bearing>.csv (cached)
python -m src.features --split Full_Test_Set  # the 11 test bearings
python -m src.models.baselines                # Ridge / RF on the default split (train 1_x,2_x -> test 3_1,3_2)
python -m src.models.cnn_lstm                 # CNN-LSTM (+attention, MC-dropout) on the default split
python -m src.benchmark                       # official PHM 2012 protocol: 6 learning -> 11 test bearings
python -m src.maintenance                     # warning rule replayed on the test bearings
python -m pytest                              # 34 tests
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
| eval | `src/evaluate.py` | MAE / RMSE / R² in minutes, PHM 2012 score, coverage, report plots |
| eval | `src/benchmark.py` | every model on the official test protocol, PHM score at the Test_set cut |
| decision | `src/maintenance.py` | RUL < T warning rule: lead time, missed / late / early warnings |

## Adding your model (GRU, 1D-CNN, ...)

Use the shared pieces so results are comparable — same windows, targets, epoch rule and metrics:

1. Write an `nn.Module` mapping a float32 batch `(B, 30, 20)` to `(B,)` (RUL / y_scale), e.g. in
   `src/models/recurrent.py`.
2. Train it with `src/models/training.py` — see `src/models/cnn_lstm.py` `main()` for the full
   pattern (`make_split` → `select_epochs` → `fit` per seed → `predict` → `per_bearing_metrics`).
   Call `use_cpu_threads()` first on CPU: the default thread count makes these tiny models 10x slower.
3. Register it in `src/benchmark.py`:
   `MODELS["gru"] = deep_model(lambda n: GRURegressor(n))`, then `python -m src.benchmark --models gru rf_row`.

Rules that matter (CLAUDE.md has the rest): split by bearing; never fit anything on test bearings;
choose epochs with `select_epochs` (training bearings only), never by looking at test results.
