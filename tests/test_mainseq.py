import matplotlib

matplotlib.use("Agg")  # headless backend for CI / no-display environments

import numpy as np
import pandas as pd
import pytest

from peyeutils.eyemovements import mainseq as ms


def test_mainseq_ampldur_linear_bounds_are_inclusive_linear():
    ampl = np.array([0.0, 10.0])
    dur = np.array([0.020, 0.020])  # exactly at the "over" bound for both points at these params
    result = ms.mainseq_ampldur_linear(
        ampl, dur,
        over_err_intercep_sec=0.020, over_err_slope_secdeg=0.0,
        under_err_intercep_sec=0.0, under_err_slope_secdeg=0.0,
    )
    assert result.tolist() == [True, True]


def test_mainseq_ampldur_linear_rejects_outside_bounds():
    ampl = np.array([0.0])
    dur = np.array([1.0])  # way too long a duration for a tiny saccade
    result = ms.mainseq_ampldur_linear(
        ampl, dur,
        over_err_intercep_sec=0.050, over_err_slope_secdeg=0.005,
        under_err_intercep_sec=0.010, under_err_slope_secdeg=0.001,
    )
    assert result.tolist() == [False]


def test_chen2021_error_gain_zero_accepts_everything_without_crashing():
    # Regression test: getparams_...() computes (1/error_gain), and it used
    # to be called BEFORE the `error_gain <= 0` shortcut check, so passing
    # the documented "accept everything" sentinel of 0 raised
    # ZeroDivisionError instead of returning all-True.
    ampl = pd.Series([1.0, 5.0, 10.0, 16.0, 20.0])
    dur = pd.Series([0.020, 0.045, 0.070, 0.100, 0.140])
    result = ms.mainseq_ampldur_linear_95pctl_human_chen2021(ampl, dur, error_gain=0)
    assert list(result) == [True] * len(ampl)


def test_chen2021_wplot_always_returns_a_2tuple():
    # Regression test: the error_gain<=0 shortcut in the _wplot variant used
    # to return a bare array instead of (result, graphics), which breaks any
    # caller doing `result, graphics = mainseq_..._wplot(...)`.
    ampl = pd.Series([1.0, 5.0, 10.0])
    dur = pd.Series([0.020, 0.045, 0.070])

    res, g = ms.mainseq_ampldur_linear_95pctl_human_chen2021_wplot(ampl, dur, error_gain=0)
    assert list(res) == [True, True, True]
    assert g is None

    res2, g2 = ms.mainseq_ampldur_linear_95pctl_human_chen2021_wplot(ampl, dur, error_gain=1)
    assert len(res2) == 3
    assert g2 is not None


def test_chen2021_normal_saccade_is_accepted_and_extreme_one_rejected():
    # A duration wildly outside the plausible main-sequence range for its
    # amplitude should be flagged as not-main-sequence.
    ampl = pd.Series([5.0, 5.0])
    dur = pd.Series([0.045, 5.0])  # second one is absurdly long
    result = ms.mainseq_ampldur_linear_95pctl_human_chen2021(ampl, dur, error_gain=1)
    assert list(result) == [True, False]
