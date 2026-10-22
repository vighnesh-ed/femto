import numpy as np
import pytest

from src.loader import SNAPSHOT_LEN, list_files, load_acc, load_temp, read_acc_file


def _write_acc(path, sep=",", micro_fmt="{:d}", eol="\n", n=SNAPSHOT_LEN):
    rows = [sep.join(["9", "39", "39", micro_fmt.format(65664 + 39 * i), f"{0.001 * i:.3f}", f"{-0.002 * i:.3f}"])
            for i in range(n)]
    path.write_text(eol.join(rows) + eol)


def _bearing(tmp_path, n_files, **kw):
    d = tmp_path / "Learning_set" / "BearingX"
    d.mkdir(parents=True)
    for i in range(1, n_files + 1):
        _write_acc(d / f"acc_{i:05d}.csv", **kw)
    return d


def test_semicolon_and_scientific_notation(tmp_path):
    # Bearing1_4 in Full_Test_Set: ';' delimiter, microseconds like 4.2504e+05
    p = tmp_path / "acc_00001.csv"
    _write_acc(p, sep=";", micro_fmt="{:.4e}")
    arr = read_acc_file(p)
    assert arr.shape == (SNAPSHOT_LEN, 6)
    assert arr[0, 3] == pytest.approx(65664, rel=1e-4)
    assert arr[10, 4] == pytest.approx(0.010)


def test_crlf_comma_file(tmp_path):
    p = tmp_path / "acc_00001.csv"
    _write_acc(p, eol="\r\n")
    assert read_acc_file(p).shape == (SNAPSHOT_LEN, 6)


def test_wrong_shape_raises(tmp_path):
    p = tmp_path / "acc_00001.csv"
    _write_acc(p, n=100)
    with pytest.raises(ValueError, match="expected shape"):
        read_acc_file(p)


def test_load_acc_keeps_vibration_channels(tmp_path):
    _bearing(tmp_path, 3)
    acc = load_acc("BearingX", raw_dir=tmp_path)
    assert acc.shape == (3, SNAPSHOT_LEN, 2)
    assert acc[0, 10, 1] == pytest.approx(-0.020)


def test_gap_in_numbering_raises(tmp_path):
    d = _bearing(tmp_path, 3)
    (d / "acc_00002.csv").unlink()
    with pytest.raises(ValueError, match="without gaps"):
        list_files("BearingX", raw_dir=tmp_path)


def test_missing_temperature_is_none(tmp_path):
    _bearing(tmp_path, 1)
    assert load_temp("BearingX", raw_dir=tmp_path) is None
