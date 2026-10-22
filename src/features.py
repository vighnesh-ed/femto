"""Stage 2: snapshot -> feature row. Produces the team contract ``data/features/<Bearing>.csv``.

    python -m src.features                              # all 6 learning bearings
    python -m src.features --bearings Bearing1_1        # one bearing
    python -m src.features --split Full_Test_Set        # all 11 test bearings (full runs)

Test bearings are built from Full_Test_Set only. Test_set is a byte-for-byte prefix of
Full_Test_Set (values verified identical, incl. Bearing1_4), so the truncated model input is
simply the first EXPECTED_ACC_COUNTS['Test_set'][bearing] rows of the full CSV. Rows beyond
that are ground truth only and must never reach a training set.

Definitions (per channel, h = horizontal, v = vertical):
  rms    sqrt(mean(x^2))
  kurt   Pearson kurtosis (Gaussian = 3), the convention in vibration monitoring
  skew   sample skewness
  peak   max |x|
  crest  peak / rms
  p2p    max(x) - min(x)
  bandK  power in the K-th of 4 equal bands of 0-12.8 kHz (3.2 kHz each), from the one-sided
         power spectrum of the mean-removed snapshot, scaled so band1+..+band4 = variance (g^2).
         Mean removal keeps any sensor DC offset out of band1.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.loader import (EXPECTED_ACC_COUNTS, FS_HZ, LEARNING_BEARINGS, PROJECT_ROOT,
                        SNAPSHOT_PERIOD_S, TEST_BEARINGS, load_acc)

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
N_BANDS = 4
CHANNELS = ("h", "v")

# THE CONTRACT. Do not change without telling the team.
FEATURE_COLUMNS = [
    "snapshot_idx", "elapsed_s",
    "rms_h", "rms_v", "kurt_h", "kurt_v", "skew_h", "skew_v",
    "peak_h", "peak_v", "crest_h", "crest_v", "p2p_h", "p2p_v",
    "band1_h", "band2_h", "band3_h", "band4_h",
    "band1_v", "band2_v", "band3_v", "band4_v",
]
SIGNAL_FEATURES = FEATURE_COLUMNS[2:]  # everything except the index/time columns


def band_powers(x: np.ndarray, fs: float = FS_HZ, n_bands: int = N_BANDS) -> np.ndarray:
    """Band power of snapshots x with shape (n, L, c) -> (n, n_bands, c)."""
    n_samples = x.shape[1]
    xc = x - x.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(xc, axis=1)) ** 2 / n_samples**2
    # One-sided spectrum: double every bin except DC and (for even L) Nyquist.
    power[:, 1:(n_samples + 1) // 2] *= 2.0

    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    band_of_bin = np.minimum((freqs / (fs / 2) * n_bands).astype(int), n_bands - 1)
    return np.stack([power[:, band_of_bin == b].sum(axis=1) for b in range(n_bands)], axis=1)


def extract_features(acc: np.ndarray) -> pd.DataFrame:
    """Snapshots of shape (n, 2560, 2) -> DataFrame with exactly FEATURE_COLUMNS."""
    if acc.ndim != 3 or acc.shape[2] != len(CHANNELS):
        raise ValueError(f"expected (n_snapshots, n_samples, 2), got {acc.shape}")

    rms = np.sqrt(np.mean(acc**2, axis=1))
    peak = np.max(np.abs(acc), axis=1)
    per_channel = {
        "rms": rms,
        "kurt": stats.kurtosis(acc, axis=1, fisher=False),
        "skew": stats.skew(acc, axis=1),
        "peak": peak,
        "crest": peak / rms,
        "p2p": np.ptp(acc, axis=1),
    }
    bands = band_powers(acc)

    idx = np.arange(acc.shape[0])
    cols: dict[str, np.ndarray] = {"snapshot_idx": idx, "elapsed_s": idx * SNAPSHOT_PERIOD_S}
    for name, values in per_channel.items():
        for c, ch in enumerate(CHANNELS):
            cols[f"{name}_{ch}"] = values[:, c]
    for c, ch in enumerate(CHANNELS):
        for b in range(N_BANDS):
            cols[f"band{b + 1}_{ch}"] = bands[:, b, c]

    df = pd.DataFrame(cols)[FEATURE_COLUMNS]
    if not np.isfinite(df[SIGNAL_FEATURES].to_numpy()).all():
        raise ValueError("non-finite feature values (all-zero snapshot?)")
    return df


def feature_path(bearing: str, features_dir: Path = FEATURES_DIR) -> Path:
    return Path(features_dir) / f"{bearing}.csv"


def build_bearing(bearing: str, split: str = "Learning_set", features_dir: Path = FEATURES_DIR,
                  force: bool = False) -> Path:
    """Compute and cache one bearing's feature CSV. Skips if it already exists unless force."""
    out = feature_path(bearing, features_dir)
    if out.exists() and not force:
        print(f"{bearing}: cached at {out.relative_to(PROJECT_ROOT)}")
        return out
    t0 = time.perf_counter()
    df = extract_features(load_acc(bearing, split))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.7g")
    print(f"{bearing}: {len(df)} snapshots -> {out.relative_to(PROJECT_ROOT)} "
          f"({time.perf_counter() - t0:.1f} s)")
    return out


def load_features(bearing: str, features_dir: Path = FEATURES_DIR) -> pd.DataFrame:
    """Read one bearing's feature CSV and check it against the contract."""
    df = pd.read_csv(feature_path(bearing, features_dir))
    if list(df.columns) != FEATURE_COLUMNS:
        raise ValueError(f"{bearing}.csv columns do not match FEATURE_COLUMNS: {list(df.columns)}")
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="Learning_set", choices=["Learning_set", "Full_Test_Set"])
    parser.add_argument("--bearings", nargs="+", help="default: every bearing in the split")
    parser.add_argument("--force", action="store_true", help="recompute even if the CSV exists")
    args = parser.parse_args(argv)

    default = LEARNING_BEARINGS if args.split == "Learning_set" else TEST_BEARINGS
    for bearing in args.bearings or default:
        if bearing not in EXPECTED_ACC_COUNTS[args.split]:
            parser.error(f"{bearing} is not in {args.split}")
        build_bearing(bearing, args.split, force=args.force)


if __name__ == "__main__":
    main()
