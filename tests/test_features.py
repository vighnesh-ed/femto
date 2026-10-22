import numpy as np
import pytest

from src.features import FEATURE_COLUMNS, band_powers, extract_features
from src.loader import FS_HZ, SNAPSHOT_LEN

t = np.arange(SNAPSHOT_LEN) / FS_HZ


def _tone(freq_hz, amp=2.0):
    x = amp * np.sin(2 * np.pi * freq_hz * t)
    return np.stack([x, x], axis=-1)[None]  # (1, 2560, 2)


def test_columns_match_contract():
    assert list(extract_features(_tone(1000)).columns) == FEATURE_COLUMNS


def test_sine_statistics():
    row = extract_features(_tone(1000, amp=2.0)).iloc[0]
    assert row.rms_h == pytest.approx(2 / np.sqrt(2), rel=1e-6)
    assert row.peak_h == pytest.approx(2.0, rel=1e-3)
    assert row.crest_h == pytest.approx(np.sqrt(2), rel=1e-3)
    assert row.p2p_h == pytest.approx(4.0, rel=1e-3)
    assert row.kurt_h == pytest.approx(1.5, rel=1e-3)  # Pearson kurtosis of a sine


@pytest.mark.parametrize("freq_hz, band", [(1000, 0), (5000, 1), (8000, 2), (12000, 3)])
def test_tone_lands_in_its_band(freq_hz, band):
    bp = band_powers(_tone(freq_hz, amp=2.0))[0, :, 0]
    assert bp[band] == pytest.approx(2.0, rel=1e-6)  # power of a sine = amp^2 / 2
    assert np.delete(bp, band).sum() < 1e-9


def test_band_powers_sum_to_variance():
    x = np.random.default_rng(0).normal(0.3, 1.0, size=(5, SNAPSHOT_LEN, 2))
    np.testing.assert_allclose(band_powers(x).sum(axis=1), x.var(axis=1), rtol=1e-10)


def test_elapsed_time():
    df = extract_features(np.repeat(_tone(1000), 3, axis=0))
    assert df.snapshot_idx.tolist() == [0, 1, 2]
    assert df.elapsed_s.tolist() == [0, 10, 20]
