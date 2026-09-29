"""
End-to-end smoke test for the top-level documented entry point,
peyeutils.preproc_and_compute_events, on fully synthetic gaze data.

This ties together several of the fixes made during review:
  - it wasn't even reachable as `peyeutils.preproc_and_compute_events`
    (only `preproc_peyefv_edf` was exported from __init__.py, even though
    peyeutils.py's own header comment calls these "front-end (exported)
    functions for convenience");
  - the mainseq error_gain<=0 ordering bug (crashed via getparams' 1/error_gain);
  - the method_om silhouette/KMeans cluster-count-vs-sample-count crash
    (hit whenever exactly 3 or 4 saccades are found, which happens here).

No eye tracker or real recording is needed: the input is a synthetic
"staircase" gaze trace with clean instantaneous jumps standing in for
saccades, which is exactly what a new user reading the README would want
to try first.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import peyeutils as pu


def _make_staircase_gaze_df(sr_hz=250.0, dur_sec=8.0, step_times=(1.5, 3.0, 4.5, 6.0),
                             levels=(0, 10, 20, 30, 40), noise_sd=0.03, seed=1, eye="R"):
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
    return pd.DataFrame({"Tsec": t, "x": x, "y": y, "eye": eye})


def test_preproc_and_compute_events_is_exported_at_top_level():
    assert pu.preproc_and_compute_events is pu.peyeutils.preproc_and_compute_events


def test_preproc_and_compute_events_end_to_end_on_synthetic_data():
    sr = 250.0
    df = _make_staircase_gaze_df(sr_hz=sr)

    sdf, ev, nogooddata = pu.preproc_and_compute_events(
        df, tcol="Tsec", xcol="x", ycol="y", sr_hzsec=sr, mainseq_err_gain=1.5,
    )

    assert nogooddata is False
    assert not ev.empty

    counts = ev["label"].value_counts()
    assert counts.get("SACC", 0) == 4
    assert counts.get("ISI", 0) >= 1
