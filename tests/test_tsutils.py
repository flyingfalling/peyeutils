import numpy as np
import pandas as pd
import pytest

from peyeutils.utils.tsutils import (
    rle,
    inverse_rle,
    get_dilated_nan_mask,
    dilate_nans,
    get_sample_rate,
    not_enough_data,
    interpolate_df_to_samplerate,
)


def test_rle_roundtrip_via_inverse_rle():
    x = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0])
    vals, starts, lens = rle(x)
    assert np.array_equal(vals, [1.0, 2.0, 3.0])
    assert np.array_equal(starts, [0, 2, 5])
    assert np.array_equal(lens, [2, 3, 1])
    assert np.array_equal(inverse_rle(vals, starts, lens), x)


def test_rle_treats_adjacent_nans_as_separate_runs():
    # NaN != NaN, so consecutive NaNs each start their own run of length 1.
    # This is documented/intentional (see the comment in rle()); pinning it
    # down here so a future "fix" doesn't silently change downstream event
    # detection that relies on this behavior.
    x = np.array([0.0, np.nan, np.nan, 0.0])
    vals, starts, lens = rle(x)
    assert np.array_equal(starts, [0, 1, 2, 3])
    assert np.array_equal(lens, [1, 1, 1, 1])


def test_rle_empty_array():
    vals, starts, lens = rle(np.array([]))
    assert len(vals) == 0 and len(starts) == 0 and len(lens) == 0


def test_rle_withnan_false_raises_on_nonfinite():
    with pytest.raises(Exception):
        rle(np.array([1.0, np.nan, 2.0]), withnan=False)


def test_get_dilated_nan_mask_expands_by_iterations():
    arr = np.array([0.0, 0.0, np.nan, 0.0, 0.0])
    mask = get_dilated_nan_mask(arr, iterations=1)
    assert np.array_equal(mask, [False, True, True, True, False])


def test_dilate_nans_only_touches_requested_columns():
    df = pd.DataFrame({"x": [0.0, 0.0, np.nan, 0.0, 0.0], "y": [1.0] * 5})
    params = dict(samplerate_hzsec=100, dilate_nan_win_sec=0.01)  # 1 sample
    out = dilate_nans(df, ["x"], params)
    assert out["x"].isna().sum() == 3  # 1 original + 1 dilated each side
    assert out["y"].isna().sum() == 0


def test_dilate_nans_requires_params():
    df = pd.DataFrame({"x": [0.0, np.nan, 0.0]})
    with pytest.raises(Exception):
        dilate_nans(df, ["x"], {})


def test_get_sample_rate_regular():
    t = np.arange(0, 1, 0.01)  # 100 Hz
    assert get_sample_rate(t) == pytest.approx(100.0)


def test_get_sample_rate_irregular_raises():
    t = np.array([0.0, 0.01, 0.05, 0.06])  # inconsistent spacing
    with pytest.raises(ValueError):
        get_sample_rate(t)


def test_not_enough_data_flags_mostly_nan_series():
    mostly_nan = np.array([np.nan] * 10 + [1.0])
    assert not_enough_data(mostly_nan, minpct=0.5, minsamps=5) is True


def test_not_enough_data_accepts_full_series():
    full = np.array([1.0] * 20)
    assert not_enough_data(full, minpct=0.5, minsamps=5) is False


def test_interpolate_df_to_samplerate_respects_explicit_endsec():
    # Regression test: this function had `en = ensec` instead of
    # `en = endsec` -- passing an explicit endsec used to raise NameError.
    n = 50
    t_ms = np.arange(n) * 10.0  # native 100 Hz, expressed in milliseconds
    df = pd.DataFrame({"Tmsec": t_ms, "x": np.sin(t_ms / 100.0)})

    out = interpolate_df_to_samplerate(
        df, tcol="Tmsec", targ_srhzsec=100, tcolunit_s=1e-3, endsec=300.0,
    )
    assert out["Tmsec"].min() == pytest.approx(0.0)
    assert out["Tmsec"].max() == pytest.approx(300.0)


def test_interpolate_df_to_samplerate_respects_explicit_startsec():
    n = 50
    t_ms = np.arange(n) * 10.0
    df = pd.DataFrame({"Tmsec": t_ms, "x": np.sin(t_ms / 100.0)})

    out = interpolate_df_to_samplerate(
        df, tcol="Tmsec", targ_srhzsec=100, tcolunit_s=1e-3, startsec=100.0,
    )
    assert out["Tmsec"].min() == pytest.approx(100.0)
