P(E)Y(E)UTILS

python "eye" utils -- utilities and tools for eye tracking, visual
stimulation, and neural recording/stimulation tasks/systems.

(C) Richard Veale 2024-




peyeutils includes:

1) Basic utilities for eye tracking including implementation of
several algorithms for preprocessing, extracting blinks, extracting
saccades from gaze traces, etc.

2) Basic utility for recording data from EYELINK etc.


peyeutils requirements:

numpy,    # Note, version may be limited due to RAY package bugs
pandas,


For video processing/image:

opencv (cv2).
Requires either system libraries or python-mediated build with appropriate
gstreamer etc. libraries installed for codec encoders etc.


For visualization:

psychopy, psychtoolbox (py),
(Including system libs such as gtk, wxPython, etc. required by psychopy)

For eyelink real-time recording:
pylink (including system libraries for EDF libraries, i.e. edfread, etc.)

For eyelink after-the-fact processing (of EDF files):
(system libraries for edfread)


Others:

ray, Fusion (included) for IMU, 


Visualization:

pyqt6 (?) for sync visualization. Include system qt libs.
pyqt6 for experiment control system (multithreaded) version. Note, by default
psychopy will capture all mouse/keyboard presses? And "focus" will get messed
up if manipulation outside of user.



Separate:

1) eyerevealer - C++ library for recording from multiple simultaneous
streams including realsense, tobii g2/g3, pupil, etc.

2) salmap_rv - C++ library (with python etc. bindings) for computing
computational saliency (attention) models on input video.







TODO (workflow) for TASKS (pyvisstimmer)

Visual stimulation does:

0) Calibration/connection to eye trackers and other streams (external cams?)
 -- Realsense etc.?
 -- I.e. sending required sync signals?

1) presentation of visual stimuli (in coordinate frame).
 -- Selection of "task" by experimenter, start task (psychopy) and record
    exact stimulus timings/calls which were displayed to a log file.
 -- allow setting of task "parameters" (i.e. visual stimulation parameters).
    for gabor patches let them "click" on where to put it etc.?

## Issue is to automatically pop up the task (ON THE CORRECT SCREEN)
   when the GUI task button is clicked, and to MAINTAIN EYE TRACKING?
   Issue is that eye tracking connection must be recreated every time? And
   copy of EDF happens each time? I want to stream head position raw too?

## Interface with OBS stream capture or grab part of stream showing eyepos?
##  Automatically detect "bad" data as jumpy?


## Display gaze data immediately, with interaction via graph, and
   on/off of each line. Then, print to PDF (or CSV?) at current granularity?
   Let them do that after I guess? Auto-name CSV?


## REV: todo, compute optic flow (estimated rotation based on disparity (not available) and optic flow left/right).



KEYBOARD:


psychtoolbox and iohub don't care what window is active, collect all data?!

event.getKeys() only records with STIMULUS WINDOW is active.

## Problem try to "generalize" everything. Best to just say what
## VERSION (git?) of software was used for experiment, so can check
## for any bugs etc.  of that exact version...


## Would like to check OBS recording of screen from eyelink, but can't due to not being monitor if it is running on laptop...fuk.



## Todo for real:

1) take windows laptop to try to set option for eye videos?
2) python interface for start/stop tobii g3 data?
