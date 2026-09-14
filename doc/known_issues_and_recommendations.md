# Known issues and recommendations (review, Sept 2026)

This is the output of a code review pass over `src/peyeutils/`. It's split
into two parts:

1. **Bugs that were fixed** in this pass (crash-causing typos/undefined
   names, missing dependency declarations, and a handful of confirmed
   logic errors). These are covered by regression tests in `tests/`.
2. **Issues that were *not* fixed** here, because they need a design
   decision only the original author can make (change of behavior, an
   unfinished feature, a tradeoff between correctness and performance,
   etc). Those are listed below by file so nothing from the review gets
   lost.

## What was fixed

See git history / diff for the full list; in short:
- `defs.py`: `videoexts` was missing a comma (`'avi' 'mov'` silently
  concatenated into `'avimov'`), so `.avi`/`.mov` files were never
  recognized as videos.
- `peyeutils.py`: `preproc_peyefv_edf` (the package's one top-level
  exported function) referenced an uninitialized `eldict` dict and an
  undefined `badtrial` variable -- it could not run at all.
- `peyeutils.py`/`__init__.py`: `preproc_and_compute_events` -- the actual
  documented "run the whole pipeline" entry point -- wasn't exported from
  `peyeutils/__init__.py` at all (only `preproc_peyefv_edf` was).
- `utils/unitutils.py`: `deg_to_rad` was called but never defined anywhere,
  so `deg_from_mid_to_meter`/`deg_to_meter`/`dist_at_angle_deg` all raised
  `NameError`. `get_center_dva_per_meter` called `exit(1)` on bad input
  instead of raising (kills the whole process/notebook).
- `utils/nputils.py`: `l2dist` used `math.sqrt` without importing `math`.
- `utils/tsutils.py`: `interpolate_df_to_samplerate` had `en = ensec`
  instead of `en = endsec` -- crashed whenever a caller passed an explicit
  `endsec`.
- `utils/vidutils.py`: an error path referenced undefined `vidfn` instead
  of `videofn`; a sanity check used `and` where `or` was intended (so it
  only fired when *both* min and max frame-index deltas were abnormal,
  instead of either).
- `utils/clusterutils.py`: `unique_clusters` indexed a plain
  `range(len(df))` list using actual `df.index` *labels* -- broken (raises
  `IndexError` or misassigns clusters) for any non-default/non-contiguous
  index. Rewritten to map by label via `.index.map(...)`.
- `eyemovements/saccadr.py`:
  - `diff_nh` referenced undefined `sg_window_sec`/`savgol_win_samp` and
    never imported `savgol_filter`.
  - `stampe_filter` called `filter_spikes` without importing it.
  - `sd_via_median_estimator` could pass a negative value to `sqrt`,
    silently producing `NaN` instead of the "value too small" exception
    the epsilon check was meant to raise.
  - **`method_om`'s KMeans/silhouette clustering (the default
    `om_usepca=True` path) crashed on very common inputs**: whenever
    exactly 3 or 4 candidate saccades were found, `silhouette_score`
    rejected the cluster count (`n_clusters == n_samples`); whenever
    exactly 1 or 2 were found, the fallback path stored a bare `ndarray`
    where a fitted-KMeans-like `.labels_` attribute was expected next,
    raising `AttributeError`. Both fixed; see `tests/test_saccadr.py` for
    reproductions (2, 3, and 4-candidate cases).
- `eyemovements/mainseq.py`: `mainseq_ampldur_linear_95pctl_human_chen2021(...)`
  and its `_wplot` variant both computed params (which divide by
  `error_gain`) *before* checking `error_gain <= 0` -- so the documented
  "pass 0 to accept everything" sentinel crashed with `ZeroDivisionError`
  instead of working. Also, the `_wplot` variant's early-return path
  returned a bare array instead of the `(result, graphics)` tuple its
  caller in `peyeutils.py` unpacks.
- `eyemovements/isi.py`: `add_ISIs_to_events` passed `durname=dursec`
  (undefined name) instead of `durname=durname`.
- `preproc/preproc.py`: `blink_df_from_samples`'s `use_index=False` branch
  sorted by an undefined `stsec` instead of the `stcol` parameter (dormant
  under the default `use_index=True`).
- `peyefv/msgutils.py`: `import_fv_trials`'s "fixdvawid recovery" branch
  depended on a `cv_vid_file_exists()` helper and a `vidpathdict` that
  don't exist anywhere in the codebase (the call was commented out,
  leaving `w`/`h`/`fps` undefined below it). It now raises
  `NotImplementedError` immediately with a clear message instead of
  crashing deep inside the loop with a confusing `NameError` after
  partially mutating state. **This is still an unfinished feature** -- see
  below.
- `eyelink/eyelink.py`: a warning message said "ALL PX DATA IS NAN" when it
  meant "ALL HX DATA IS NAN" (copy-paste).
- `imu/imu.py`: AHRS output arrays were allocated with `np.empty` (garbage
  memory) instead of `np.full(..., np.nan)`, so timesteps skipped due to
  sensor dropout kept garbage values instead of NaN. Magnetometer burn-in
  collapsed all 3 axes into one scalar via `np.nanmean` over the whole 2D
  array instead of computing a per-axis mean.
- `tobiig3/tobiig3.py`: `.copy(0)` (shallow copy, `deep=0`) fixed to
  `.copy()`.
- `plotting/plotting.py`: a stimulus-color fallback used the string
  `'gray'`, which crashed (`TypeError`) the moment it was multiplied by
  `0.8` for any stimulus name not found in the color map (e.g. NaN names).
- `eyerevealer/binaryutils.py`: `timestamps_from_file` called `exit(0)` on
  an unreadable file instead of raising, and never closed its file handle.
- `pyproject.toml`: `scikit-learn`, `seaborn`, and `imufusion` are
  imported by the code (`eyemovements/saccadr.py`,
  `eyemovements/mainseq.py` + `peyeutils.py`, `imu/imu.py` respectively)
  but were missing from `dependencies`.

## Not fixed -- needs the author's judgment

### `eyelink/eyelink.py`
- `preproc_EL_A02_separate_samps_eye` unconditionally does
  `df.loc[df.useeye==False, ['px','py','gx','gy','hx','hy']] = np.nan`,
  assuming all six columns exist. Unlike the sibling function
  `preproc_EL_A03_remove_errors` (which guards each column with
  `if 'hx' in df.columns`), this can fabricate all-NaN `hx`/`hy` columns
  for files that never had head-tracking data, which then makes later code
  think head tracking *was* recorded (misleading "ALL HX DATA IS NAN"
  warnings). Suggest guarding per-column like the sibling function does.
- No fallback for genuinely monocular EDF files: if the left/right column
  name sets don't match exactly, it raises immediately rather than
  treating it as single-eye data.
- The `_left`/`_right` column-splitting regex is unanchored
  (`r"(.+)_(left|right)"` via `re.match`, no `$`), so a column literally
  named e.g. `foo_lefteye` would be silently mis-parsed.
- Several public functions have mutable default arguments
  (`eyes_to_use=[...]`, `nogazecols=['gx','gy']`). Not currently mutated,
  but a footgun for future edits.

### `peyefv/msgutils.py`
- The `fixdvawid` recovery path in `import_fv_trials` (see above) needs a
  `vidpathdict` (video name -> filesystem path) and a
  `cv_vid_file_exists(path) -> (w, h, fps, nframes)` helper written and
  wired up; it currently just fails loudly instead of working.
- `subfm.loc[len(subfm.index)] = newrow` (used to "append" a synthesized
  row) assumes `len(subfm.index)` isn't already a used index label --
  `subfm`'s index isn't reset, so this can silently overwrite an existing
  row instead of appending, if that integer label happens to already
  exist.
- `get_recordingsession_info` computes `recdict` (from `'REC'`-tagged
  messages) but never merges it into the returned dict -- looks like an
  unfinished merge (comment says it "overlaps" `sessdict`).
- `import_fv_blocks`'s recovery heuristic for uneven S/E block-boundary
  messages copies a start-message row to synthesize a missing end-message,
  but only patches the seconds-column (`tcol`) field, not `ELtime` --
  downstream code that reads `enrow.ELtime` for `blkend_el` gets a value
  inconsistent with the corrected `blkend_s`.
- Very widespread use of `exit(1)`/`exit(0)` for error handling throughout
  this file (and `eyerevealer/binaryutils.py`, previously) instead of
  raising -- kills the whole host process/notebook instead of letting a
  caller catch and skip a bad file. Only the one instance blocking the
  package's own top-level entry point was changed; the rest are listed
  here as a pattern worth revisiting file-wide.

### `eyemovements/`
- `blink.py`'s `merge_blinkedge_saccades` is a documented no-op stub
  (`return ev` with no logic) -- referenced in `peyeutils.py`'s comments as
  something that should eventually merge blinks/saccades with very small
  inter-event gaps.
- `saccadr.py`'s `sd_via_median_estimator` raises a fairly opaque
  `Exception("ERROR, median too small...")` whenever a gaze channel has
  near-zero velocity variance (e.g., a synthetic/calibration trace with a
  perfectly flat Y channel). This is real, reachable behavior (hit while
  writing a docstring example for `saccadr_detect_saccades`) -- a clearer
  upfront message (naming which channel/eye is degenerate) would save
  debugging time.
- `method_om`'s PCA + KMeans + silhouette clustering (the default
  `om_usepca=True`) is inherently more reliable with more candidate
  saccades to cluster over; per-trial calls with only a handful of
  candidates make the "noise vs. signal" cluster split fairly arbitrary.
  Consider batching across trials/blocks before running `method_om` when
  using PCA filtering, or defaulting `om_usepca=False` for short/per-trial
  calls.

### `utils/`
- `tsutils.py`: `remove_suspicious_repeats` assigns via chained indexing
  (`df[xn][s:e] = np.nan`), the classic pandas anti-pattern that can
  trigger `SettingWithCopyWarning` and, depending on dtype/pandas version,
  silently fail to write back to the caller's DataFrame. Should use
  `df.loc[s:e, xn] = np.nan` (mind that `.loc` slicing is inclusive of the
  end label, unlike positional slicing -- check the intended range when
  changing this).
- `tsutils.py`: `interpolate_df_to_samplerate` uses mutable default
  arguments (`truesrs=dict()`, `maxtdeltas_s=dict()`) that it *mutates*
  (`truesrs[c] = ...`). Since Python default arguments are created once,
  not per-call, these dicts persist and accumulate entries across repeated
  calls that don't pass their own -- if two calls in the same process use
  overlapping column names with different true sample rates/max deltas,
  the second call can silently reuse the first call's values. Low risk in
  a typical "one call per session" script, higher risk in a loop over many
  files/subjects.
- `interpolate_df_to_samplerate`'s `startsec`/`endsec` parameters are
  named as if they're in seconds, but are actually compared directly
  against `df[tcol]`, i.e. whatever unit `tcol` happens to be in (could be
  milliseconds). Consider renaming or documenting explicitly.
- `vidutils.py`: hardcodes a `25.0` fps assumption in "drift" calculations
  rather than using the video's own measured `fps`; `read_video_timestamps`
  divides by `fps` with no guard against `fps==0` for an unreadable video
  header.
- `fsutils.py`: `create_dir` catches `Exception` broadly and returns
  `False`, which can mask real errors (permissions, disk full) as an
  unremarkable boolean callers may not check.
- `clusterutils.py`/general: none currently known beyond what was fixed.

### `tobiig3/tobiig3.py`
- Several places check `type(x) is not float` to detect "no value" before
  indexing `x[0]`/`x[1]`/`x[2]`; a JSON `null` (Python `None`) -- a
  realistic value for a dropped/invalid gaze or IMU sample -- fails that
  check (`type(None) is not float` is True) and falls into the indexing
  branch, raising `TypeError`. Worth auditing all of these against real
  recordings that include dropout.
- `pd.merge(gazedf, imudf, on=['timestamp'], how='outer')` joins on exact
  float equality between two independently-clocked sensor streams; this
  will typically match almost nothing and instead produce a huge,
  mostly-NaN union frame. An `merge_asof`-style nearest-timestamp join (or
  resampling both to a common clock first) would likely be the intended
  behavior.
- A `cv2.VideoCapture` handle and a JSON file handle are opened without
  being released/closed in at least two functions (`read_video_timestamps`
  equivalent and `resample_tobii3_json_to_csv`).
- Lines around `self.mode = 'eyerevealer'` immediately followed by
  `if self.mode == 'eyerevealer':` (trivially true right after being set)
  look like leftover refactoring debris rather than intentional logic --
  worth a second look.
- Heavy reliance on `print()` for status/diagnostics rather than the
  `logging` module; many hardcoded thresholds/constants that would be
  better as parameters (`self.sc_widdva`, `magsr`/`othersr`, `cutoffquant`,
  `jumpcutoffsec`, ...).

### `plotting/plotting.py`
- `plot_gaze_chunks_wpupil` uses `events_df[eye_col].isin(eyes_to_plot)`
  whenever `events_df` is supplied, but `eye_col`/`eyes_to_plot` both
  default to `None` and are documented as independently optional --
  calling with `events_df` but no `eye_col`/`eyes_to_plot` raises.
- Its "legend cleanup" step reads handles/labels off whichever `ax`/`p_ax`
  survive from the *last* iteration of the inner per-chunk loop, so a
  multi-chunk figure's legend can miss labels that only appeared in
  earlier chunks (the sibling function `plot_gaze_chunks` accumulates
  handles across all chunks instead).
- Generators that `yield fig` (e.g. `plot_gaze_chunks_wpupil`) never call
  `plt.close(fig)`; a caller who doesn't explicitly close each returned
  figure will leak matplotlib figures across a long run (many trials/pages).
- The same 4-line `import pandas/numpy/pyplot/...` block is duplicated
  4 times at the top of the file (harmless but worth a cleanup pass).

### `eyerevealer/binaryutils.py`
- An unrecognized tag byte causes `read_unpack_next_tagtype` to print a
  warning and return `None`, which the caller's parse loop treats
  identically to legitimate end-of-file (`break`). A corrupted/unexpected
  tag silently truncates timestamp parsing rather than raising -- real
  data loss with only an easily-missed print statement as a trace.
- `df_from_type` builds a DataFrame from `colnames` only and never
  actually inserts the `data` argument passed to it -- looks incomplete.
  It isn't called anywhere else in the reviewed files, so it may be
  genuinely dead code; worth confirming before deleting.

### Packaging / import structure
- `utils/__init__.py` does `from .vidutils import *`, and `vidutils.py`
  does a hard top-level `import av`; `eyemovements/__init__.py` imports
  `saccadr.py`, which does a hard top-level `import sklearn`; the
  top-level `__init__.py` does `from . import imu`, and `imu.py` does a
  hard top-level `import imufusion`. All three are now correctly declared
  in `pyproject.toml`, so a normal `pip install .` works -- but it also
  means installing peyeutils at all requires video (`av`,
  `opencv-python`), ML (`scikit-learn`), and IMU-fusion (`imufusion`)
  dependencies even for a user who only wants e.g. unit conversions or
  saccade detection on already-loaded CSV data. If a lighter-weight
  install matters, consider `[project.optional-dependencies]` extras
  (`video`, `imu`, ...) plus moving those specific imports inside the
  functions that need them, rather than at module top-level.

## Suggested next steps (priority order)

1. Decide whether monocular EDF support / the `fixdvawid` video-recovery
   path in `msgutils.py` are still needed; if so, implement the missing
   `vidpathdict`/`cv_vid_file_exists` pieces.
2. Replace the `exit()` calls throughout `peyefv/msgutils.py` with raised
   exceptions, for the same reason it mattered for the package's own
   top-level entry point: a batch-processing caller currently cannot
   recover from one bad file.
3. Fix the tobiig3 gaze/IMU timestamp merge strategy (exact-equality join
   -> nearest/asof join) before trusting any binocular-IMU-combined
   analysis built on it.
4. Consider whether `om_usepca=True` should remain the saccade-detection
   default given how few candidate saccades a typical single call sees.
