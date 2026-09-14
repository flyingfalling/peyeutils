import numpy as np
import pandas as pd
import pytest

from peyeutils.eyemovements import saccadr as sc


def test_shift_elements_forward():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.array_equal(sc.shift_elements(arr, 2, 0.0), [0.0, 0.0, 1.0, 2.0, 3.0])


def test_shift_elements_backward():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.array_equal(sc.shift_elements(arr, -2, 0.0), [3.0, 4.0, 5.0, 0.0, 0.0])


def test_shift_elements_zero_is_identity():
    arr = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(sc.shift_elements(arr, 0, -1.0), arr)


def test_default_saccadr_params_has_expected_keys():
    params = sc.default_saccadr_params()
    for key in ("nh_max_vel_degsec", "om_vel_thresh_degsec", "ek_vel_thresh_lambda",
                "saccadr_min_sep_sec", "saccadr_min_dur_sec"):
        assert key in params


def test_sd_via_median_estimator_reasonable_for_normal_data():
    rng = np.random.RandomState(0)
    x = rng.normal(0, 2.0, 10000)
    sd = sc.sd_via_median_estimator(x)
    assert 0 < sd < 2.0  # robust estimator underestimates true SD by design, but must be positive+finite


def test_sd_via_median_estimator_raises_on_degenerate_constant_input():
    with pytest.raises(Exception):
        sc.sd_via_median_estimator(np.array([5.0] * 100))


def _make_staircase_gaze_trace(sr_hz=250.0, dur_sec=8.0, step_times=(1.5, 3.0, 4.5, 6.0),
                                levels=(0, 10, 20, 30, 40), noise_sd=0.03, seed=42):
    rng = np.random.RandomState(seed)
    dt = 1.0 / sr_hz
    n = int(dur_sec * sr_hz)
    t = np.arange(n) * dt
    x = np.zeros(n)
    level_idx = 0
    for i, ti in enumerate(t):
        while level_idx < len(step_times) and ti >= step_times[level_idx]:
            level_idx += 1
        x[i] = levels[level_idx]
    x = x + rng.normal(0, noise_sd, size=n)
    y = rng.normal(0, noise_sd, size=n)
    return pd.DataFrame({"Tsec": t, "x": x, "y": y})


def test_saccadr_detects_synthetic_stepwise_saccades():
    # Regression test for a real crash: with the default params (om_usepca=True),
    # method_om's KMeans/silhouette clustering used a cluster-count upper bound
    # that could equal the number of candidate saccades (3 or 4), which
    # sklearn's silhouette_score rejects ("Valid values are 2 to n_samples-1").
    # A clean 4-saccade synthetic trace reproduces exactly that count.
    sr = 250.0
    df = _make_staircase_gaze_trace(sr_hz=sr)
    params = sc.default_saccadr_params()
    params["samplerate_hzsec"] = sr

    sdf, evdf = sc.saccadr_detect_saccades(
        df, params, tsecname="Tsec", xname="x", yname="y", eyecol="eye"
    )

    saccs = evdf[evdf["label"] == "SACC"].sort_values("stsec").reset_index(drop=True)
    assert len(saccs.index) == 4

    expected_times = [1.5, 3.0, 4.5, 6.0]
    for expected_t, (_, row) in zip(expected_times, saccs.iterrows()):
        assert row["stsec"] == pytest.approx(expected_t, abs=0.05)
        assert row["ampldva"] == pytest.approx(10.0, abs=1.0)


def test_saccadr_handles_three_candidate_saccades_without_crashing():
    # Same bug as above, at the other crashing boundary (exactly 3 candidates).
    sr = 250.0
    df = _make_staircase_gaze_trace(
        sr_hz=sr, dur_sec=6.5, step_times=(1.5, 3.0, 4.5), levels=(0, 10, 20, 30),
    )
    params = sc.default_saccadr_params()
    params["samplerate_hzsec"] = sr

    sdf, evdf = sc.saccadr_detect_saccades(
        df, params, tsecname="Tsec", xname="x", yname="y", eyecol="eye"
    )
    saccs = evdf[evdf["label"] == "SACC"]
    assert len(saccs.index) == 3


def test_saccadr_handles_two_candidate_saccades_without_crashing():
    # Regression test for a related crash: when method_om finds fewer than 3
    # candidate saccades, it short-circuits clustering with a bare ndarray
    # standing in for a fitted KMeans model, but the code unconditionally
    # accessed `.labels_` on it afterwards -- an attribute plain ndarrays
    # don't have. AttributeError on every 1- or 2-saccade trial.
    sr = 250.0
    df = _make_staircase_gaze_trace(
        sr_hz=sr, dur_sec=5.0, step_times=(1.5, 3.0), levels=(0, 10, 20),
    )
    params = sc.default_saccadr_params()
    params["samplerate_hzsec"] = sr

    sdf, evdf = sc.saccadr_detect_saccades(
        df, params, tsecname="Tsec", xname="x", yname="y", eyecol="eye"
    )
    saccs = evdf[evdf["label"] == "SACC"]
    assert len(saccs.index) == 2
