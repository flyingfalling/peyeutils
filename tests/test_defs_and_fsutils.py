"""
Regression tests for src/peyeutils/defs.py and utils/fsutils.py.

These pin down a real bug that was found during review: `defs.videoexts`
was defined with a missing comma (`'avi' 'mov'`), which Python silently
concatenates into a single bogus entry `'avimov'`. That made every
`.avi` and `.mov` file get misclassified as "not a video" by
`is_filename_vid`.
"""

import peyeutils as pu


def test_videoexts_has_separate_avi_and_mov_entries():
    assert 'avi' in pu.videoexts
    assert 'mov' in pu.videoexts
    assert 'avimov' not in pu.videoexts


def test_is_filename_vid_recognizes_common_video_extensions():
    for fname in ["clip.mp4", "clip.avi", "clip.mov", "clip.MOV", "clip.mkv"]:
        assert pu.utils.is_filename_vid(fname) is True, fname


def test_is_filename_vid_rejects_non_video_extensions():
    assert pu.utils.is_filename_vid("notes.txt") is False
    assert pu.utils.is_filename_vid("photo.png") is False


def test_is_filename_img_recognizes_common_image_extensions():
    for fname in ["a.png", "a.jpg", "a.JPEG", "a.tiff"]:
        assert pu.utils.is_filename_img(fname) is True, fname


def test_is_filename_img_rejects_non_image_extensions():
    assert pu.utils.is_filename_img("clip.mp4") is False
