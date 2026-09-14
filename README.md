# P(E)Y(E)UTILS
(C) Richard Veale 2024-

Python "eye" utils -- utilities and tools for eye tracking, visual
stimulation, and neural recording/stimulation tasks/systems.

In principle, will be compatible with data from both wearable and remote
(fixed) eye-trackers.

## remote (stationary) gaze trackers:
- SR Research EYELINK 1000+
- Tobii Pro Spark/Fusion/Spectrum
- Arrington Research ViewPoint EyeTracker

## wearable (glasses) gaze trackers:
- Tobii Glasses 2/3
- Tobii Glasses X
- Pupil Invisible
- Pupil Neon
- Pupil Core
- SMI Eye Tracking Glasses.

Most modern eye-tracking is video-based (videooculography, VOG), as all the listed products above.

There also exist other methods more common in physiology research such as
electrooculography (EOG -- electrodes around the eyes) or scleral search coils
(induced electrical currents in conductive coils sutured to sclera of eyeball
or embedded in contact lenses). These functions should work for all gaze
time-series data.


# peyeutils includes three main pieces:

## 1) Algorithms and utilities for eye tracking including functions for:

- preprocessing (smoothing) of data
- labelling/removing blinks and artifacts
- labelling events such as saccades, pursuits, fixations
- conversion to physical units (degrees, meters) and relative units (stimulus)
- statistics of gaze
- plotting/summarization of gaze
- AHRS/head motion (gyroscope/accelerometer/magnetometer)
- Comparions with predictive models of gaze (e.g. saliency)
- Storage and querying and joining of large amounts of multi-sensor/multimodal data (physiology data, gaze data, experimental conditions, stimuli)

## 2) peye freeviewing (peyefv) -- Functions for running eye-tracking experiments
(via e.g. psychopy), defining stimuli, timing, and responses, with hooks for
appropriate synchronization methods (e.g. send messages to EDF files for
EYELINK)

- Color/luminance correction utilities
- Gaze contingent experiments

## 3) Controlling (wearable) gaze trackers via network connections



# Examples

Several examples are listed in the examples directory


# Quickstart

No eye tracker or recording file is needed to try the core saccade-detection
pipeline -- it works on any pandas DataFrame of (time, x, y) gaze samples at
a regular sample rate, in degrees of visual angle (dva). Here it is applied
to a synthetic gaze trace with four abrupt 10 dva "saccades":

```python
import numpy as np
import pandas as pd
import peyeutils as pu

# --- Build a synthetic gaze trace: fixations at 0/10/20/30/40 dva, ---
# --- connected by four instantaneous ("saccadic") steps.            ---
rng = np.random.RandomState(0)
sr_hzsec = 250.0
dt = 1.0 / sr_hzsec
step_times = [1.5, 3.0, 4.5, 6.0]
levels = [0, 10, 20, 30, 40]

n = int(8.0 * sr_hzsec)
t = np.arange(n) * dt
x = np.zeros(n)
level_idx = 0
for i, ti in enumerate(t):
    while level_idx < len(step_times) and ti >= step_times[level_idx]:
        level_idx += 1
    x[i] = levels[level_idx]

df = pd.DataFrame({
    "Tsec": t,
    "x": x + rng.normal(0, 0.03, size=n),  # fixational jitter
    "y": rng.normal(0, 0.03, size=n),
    "eye": "R",
})

# --- Preprocess (smoothing/blink handling) and detect saccades/fixations/ISIs ---
samples_df, events_df, nogooddata = pu.preproc_and_compute_events(
    df,
    tcol="Tsec", xcol="x", ycol="y",
    sr_hzsec=sr_hzsec,
    mainseq_err_gain=1.5,  # how strict the main-sequence sanity check on saccades is
)

print(events_df["label"].value_counts())
# SACC    4
# ISI     4
# FIXA    5
```

`events_df` has one row per detected event (`SACC`, `BLNK`, `FIXA`, `ISI`, ...)
with columns `stsec`/`ensec` (start/end time), `dursec`, `ampldva` (amplitude
in degrees of visual angle, for saccades), `angle`, and `eye`.

For lower-level building blocks used inside that pipeline -- saccade
detection alone, unit conversions, run-length encoding of event masks, etc.
-- see `tests/` for small, self-contained usage examples of each.

See `doc/known_issues_and_recommendations.md` for a list of known rough
edges and suggested improvements uncovered during a code review of this
package (Sept 2026).


# Requirements:
- numpy
- pandas
- scipy

## For video processing/image:
- opencv (cv2) -- also requires either system
libraries or python-mediated build with appropriate ffmpeg/gstreamer for codecs.



## For visualization:
- psychopy (Including system libs such as gtk, wxPython, etc. required by psychopy). Due to issues with compatibility, a development version is recommended which can handle python>3.10.

## For eyelink real-time recording:
- pylink (for EYELINK -- including SR Research eyelink SDK and system libraries https://www.sr-research.com/support/docs.php?topic=linuxsoftware)
- Tobii SDK (for tobii https://developer.tobiipro.com/)

## For eyelink after-the-fact processing (of EDF files):
- pyedfread (https://github.com/flyingfalling/pyedfread)


## Others:
- ray (multi-node parallel processing)
- Fusion for IMU (included as sub-package)


## Visualization:
- pyqt6 (?) for sync visualization. Include system qt libs.
- pyqt6 for experiment control system (multithreaded) version. Note, by default
psychopy will capture all mouse/keyboard presses? And "focus" will get messed
up if manipulation outside of user.
- matplotlib
- seaborn
- pygraph

## Sailency/Realtime

- 1) eyerevealer - C++ library for recording from multiple simultaneous
streams including realsense, tobii g2/g3, pupil, etc.
- 2) salmap_rv - C++ library (with python etc. bindings) for computing
computational saliency (attention) models on input video.

