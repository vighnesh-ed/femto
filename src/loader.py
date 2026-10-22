"""Stage 1: raw PHM 2012 (FEMTO-ST / PRONOSTIA) files -> arrays.

One ``acc_NNNNN.csv`` is one snapshot: 0.1 s of horizontal + vertical acceleration
sampled at 25.6 kHz (2560 rows), recorded every 10 s. Files are numbered 1..N with no
gaps, so snapshot_idx = file_number - 1 and elapsed time = snapshot_idx * 10 s.

Data quirks handled here (see CLAUDE.md):
  1. Bearing1_4 uses ';' as delimiter, everything else ',' -> sniffed per file.
  2. Microsecond column can be in scientific notation       -> everything parsed as float.
  3. No headers                                               -> header=None.
  4. Some bearings have no temp files                         -> load_temp returns None.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

FS_HZ = 25_600
SNAPSHOT_LEN = 2560
SNAPSHOT_PERIOD_S = 10

ACC_COLUMNS = ["hour", "minute", "second", "microsecond", "horiz_accel", "vert_accel"]
TEMP_COLUMNS = ["hour", "minute", "second", "decisecond", "temperature"]

LEARNING_BEARINGS = ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2"]
TEST_BEARINGS = [
    "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7",
    "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7",
    "Bearing3_3",
]

# Verified acc file counts (CLAUDE.md). The loader refuses to return a bearing that disagrees.
EXPECTED_ACC_COUNTS = {
    "Learning_set": {
        "Bearing1_1": 2803, "Bearing1_2": 871, "Bearing2_1": 911,
        "Bearing2_2": 797, "Bearing3_1": 515, "Bearing3_2": 1637,
    },
    "Test_set": {
        "Bearing1_3": 1802, "Bearing1_4": 1139, "Bearing1_5": 2302, "Bearing1_6": 2302,
        "Bearing1_7": 1502, "Bearing2_3": 1202, "Bearing2_4": 612, "Bearing2_5": 2002,
        "Bearing2_6": 572, "Bearing2_7": 172, "Bearing3_3": 352,
    },
    "Full_Test_Set": {
        "Bearing1_3": 2375, "Bearing1_4": 1428, "Bearing1_5": 2463, "Bearing1_6": 2448,
        "Bearing1_7": 2259, "Bearing2_3": 1955, "Bearing2_4": 751, "Bearing2_5": 2311,
        "Bearing2_6": 701, "Bearing2_7": 230, "Bearing3_3": 434,
    },
}


def bearing_dir(bearing: str, split: str = "Learning_set", raw_dir: Path = RAW_DIR) -> Path:
    path = Path(raw_dir) / split / bearing
    if not path.is_dir():
        raise FileNotFoundError(
            f"{path} not found. Is the sparse checkout set to include '{split}'? See CLAUDE.md."
        )
    return path


def list_files(bearing: str, kind: str = "acc", split: str = "Learning_set",
               raw_dir: Path = RAW_DIR) -> list[Path]:
    """Snapshot files of one kind ('acc' or 'temp') in time order, checked for gaps."""
    files = sorted(bearing_dir(bearing, split, raw_dir).glob(f"{kind}_*.csv"))
    numbers = [int(f.stem.split("_")[1]) for f in files]
    if numbers != list(range(1, len(files) + 1)):
        raise ValueError(f"{split}/{bearing}: {kind} files are not numbered 1..{len(files)} without gaps")
    return files


def _sniff_sep(path: Path) -> str:
    with open(path, "r") as fh:
        first = fh.readline()
    return ";" if ";" in first else ","


def read_acc_file(path: Path) -> np.ndarray:
    """One acc snapshot as a (2560, 6) float array, columns as in ACC_COLUMNS."""
    arr = pd.read_csv(path, header=None, sep=_sniff_sep(path), dtype=np.float64).to_numpy()
    if arr.shape != (SNAPSHOT_LEN, len(ACC_COLUMNS)):
        raise ValueError(f"{path}: expected shape ({SNAPSHOT_LEN}, {len(ACC_COLUMNS)}), got {arr.shape}")
    return arr


def load_acc(bearing: str, split: str = "Learning_set", raw_dir: Path = RAW_DIR,
             check_count: bool = True) -> np.ndarray:
    """All vibration snapshots of one bearing.

    Returns float64 array of shape (n_snapshots, 2560, 2); last axis = (horizontal, vertical).
    """
    files = list_files(bearing, "acc", split, raw_dir)
    expected = EXPECTED_ACC_COUNTS.get(split, {}).get(bearing)
    if check_count and expected is not None and len(files) != expected:
        raise ValueError(f"{split}/{bearing}: found {len(files)} acc files, expected {expected}")

    out = np.empty((len(files), SNAPSHOT_LEN, 2), dtype=np.float64)
    for i, f in enumerate(files):
        out[i] = read_acc_file(f)[:, 4:6]
    return out


def load_temp(bearing: str, split: str = "Learning_set", raw_dir: Path = RAW_DIR) -> pd.DataFrame | None:
    """Concatenated temperature log (10 Hz), or None if this bearing has no temp files.

    Temperature files are NOT aligned 1:1 with acc files (fewer of them, different clock),
    so use the hour/minute/second columns if you need to join against snapshots.
    """
    files = sorted(bearing_dir(bearing, split, raw_dir).glob("temp_*.csv"))
    if not files:
        return None
    frames = [
        pd.read_csv(f, header=None, sep=_sniff_sep(f), names=TEMP_COLUMNS, dtype=np.float64)
        for f in files
    ]
    return pd.concat(frames, ignore_index=True)
