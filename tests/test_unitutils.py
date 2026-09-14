import math

import pytest

import peyeutils as pu
from peyeutils.utils import unitutils as uu


def test_deg_to_rad_matches_math_radians():
    # Regression test: deg_to_rad was called by dist_at_angle_deg /
    # deg_from_mid_to_meter but was never defined anywhere in the codebase,
    # so both functions raised NameError on every call.
    assert uu.deg_to_rad(180) == pytest.approx(math.pi)
    assert uu.deg_to_rad(0) == pytest.approx(0.0)


def test_deg_from_mid_to_meter_zero_angle_is_zero_offset():
    assert uu.deg_from_mid_to_meter(1.0, 0.0) == pytest.approx(0.0)


def test_dist_at_angle_deg_zero_angle_is_mid_dist():
    assert uu.dist_at_angle_deg(0.75, 0.0) == pytest.approx(0.75)


def test_deg_to_meter_is_alias_for_deg_from_mid_to_meter():
    assert uu.deg_to_meter is uu.deg_from_mid_to_meter
    assert uu.wid_at_angle_deg is uu.deg_from_mid_to_meter


def test_flatscreen_dva_and_deg_from_mid_to_meter_are_consistent():
    # flatscreen_dva gives the FULL visual angle subtended by a width wm at
    # distance dm; deg_from_mid_to_meter gives the physical offset for a
    # given HALF-angle from center. Half of flatscreen_dva fed back through
    # deg_from_mid_to_meter should recover half the original width.
    dm = 0.6  # meters
    wm = 0.3  # meters (e.g. a 30cm-wide monitor)
    full_angle_deg = uu.flatscreen_dva(dm, wm)
    recovered_half_width = uu.deg_from_mid_to_meter(dm, full_angle_deg / 2.0)
    assert recovered_half_width == pytest.approx(wm / 2.0, rel=1e-6)


def test_get_center_dva_per_meter_matches_manual_small_angle_calc():
    dm = 1.0
    ppm = 1000.0
    dva_per_m = uu.get_center_dva_per_meter(dm, ppm, reference_width_meters=0.01)
    expected = math.degrees(uu.meter_to_rad(dm, 0.01)) / 0.01
    assert dva_per_m == pytest.approx(expected)


def test_get_center_dva_per_meter_raises_on_bad_input():
    # Regression test: this used to call exit(1), which kills the whole
    # interpreter/notebook instead of raising a catchable error.
    with pytest.raises(ValueError):
        uu.get_center_dva_per_meter(dm=-1.0, ppm=1000.0)
    with pytest.raises(ValueError):
        uu.get_center_dva_per_meter(dm=1.0, ppm=0.0)


def test_get_center_dva_per_meter_raises_when_too_close_to_screen():
    with pytest.raises(ValueError):
        uu.get_center_dva_per_meter(dm=0.001, ppm=1000.0)


def test_msec_sec_roundtrip():
    assert uu.sec_to_msec(uu.msec_to_sec(1234.0)) == pytest.approx(1234.0)
