import math

import numpy as np
import pytest

import peyeutils as pu


def test_linsteps_basic():
    steps = pu.utils.linsteps(0, 10, 2)
    assert np.allclose(steps, [0, 2, 4, 6, 8, 10])


def test_linsteps_pastend_false_matches_floor():
    steps = pu.utils.linsteps(0, 9, 2, pastend=False)
    # floor((9-0)/2) = 4 -> indices 0..4 -> [0,2,4,6,8]
    assert np.allclose(steps, [0, 2, 4, 6, 8])


def test_allnan_true_for_all_nan():
    assert pu.utils.allnan([np.nan, np.nan, np.nan]) is True


def test_allnan_false_if_any_finite():
    assert pu.utils.allnan([np.nan, 1.0, np.nan]) is False


def test_l2dist_matches_manual_euclidean_distance():
    # This is a regression test for a real bug: l2dist used math.sqrt
    # without importing `math`, which raised NameError on every call.
    d = pu.utils.l2dist(0, 0, 3, 4)
    assert d == pytest.approx(5.0)


def test_l2dist_matches_l2distvec():
    x1, y1, x2, y2 = 1.0, 2.0, 4.0, 6.0
    assert pu.utils.l2dist(x1, y1, x2, y2) == pytest.approx(
        pu.utils.l2distvec(x1, y1, x2, y2)
    )


def test_l2distvec_is_vectorized():
    x1 = np.array([0.0, 0.0])
    y1 = np.array([0.0, 0.0])
    x2 = np.array([3.0, 6.0])
    y2 = np.array([4.0, 8.0])
    result = pu.utils.l2distvec(x1, y1, x2, y2)
    assert np.allclose(result, [5.0, 10.0])
