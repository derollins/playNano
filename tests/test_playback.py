"""Tests for the live and interactive playback window."""

import cv2
import numpy as np

from playNano.afm_stack import AFMImageStack
from playNano.playback.vis import play_stack_cv


def test_play_stack_cv_handles_flat_data(monkeypatch, tmp_path):
    """Test that play_stack_cv handles flat image data without crashing."""

    stack = AFMImageStack(
        data=np.zeros((2, 2, 2)),  # flat data
        pixel_size_nm=1.0,
        channel="height_trace",
        file_path=tmp_path,
        frame_metadata=[{"timestamp": 0}, {"timestamp": 1}],
    )

    # Patch OpenCV functions to simulate ESC press and avoid GUI
    monkeypatch.setattr(cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "setWindowProperty", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "getWindowImageRect", lambda *args, **kwargs: (0, 0, 2, 2))
    monkeypatch.setattr(cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: 27)  # ESC
    monkeypatch.setattr(cv2, "destroyWindow", lambda name: None)
    monkeypatch.setattr(cv2, "cvtColor", lambda src, code: src)
    monkeypatch.setattr(cv2, "resize", lambda src, dsize, interpolation: src)

    # Should not raise an error
    play_stack_cv(stack, fps=10.0)
