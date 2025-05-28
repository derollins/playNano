"""Tests for the playNano processing.image_processing module."""

import numpy as np
import pytest

from playNano.processing import image_processing
from playNano.processing.image_processing import flatten_afm_frame, flatten_stack


def test_flatten_afm_frame_valid():
    """Check flatten_afm_frame returns array with same shape as input."""
    frame = np.random.rand(64, 64) * 10  # Fake AFM data
    result = flatten_afm_frame(frame, filename="test_frame")
    assert isinstance(result, np.ndarray)
    assert result.shape == frame.shape


def test_flatten_afm_frame_empty():
    """Verify flatten_afm_frame returns None for empty input array."""
    frame = np.array([])
    result = flatten_afm_frame(frame)
    assert result is None


def test_flatten_afm_frame_with_nans():
    """Ensure flatten_afm_frame returns None for all-NaN arrays."""
    frame = np.full((64, 64), np.nan)
    result = flatten_afm_frame(frame)
    assert result is None


def test_flatten_afm_frame_custom_config():
    """Test flatten_afm_frame works with a custom filter configuration."""
    frame = np.random.rand(64, 64)
    custom_config = {
        "gaussian_size": 3,
        "gaussian_mode": "mirror",
        "row_alignment_quantile": 0.1,
        "threshold_method": "std_dev",
        "otsu_threshold_multiplier": 1.0,
        "threshold_std_dev": {"above": 0.2, "below": 5},
        "threshold_absolute": None,
        "remove_scars": {"run": False},
    }
    result = flatten_afm_frame(frame, filter_config=custom_config)
    assert isinstance(result, np.ndarray)


def test_flatten_stack_multiple_frames():
    """Check flatten_stack returns flattened stack with same shape."""
    stack = np.random.rand(5, 64, 64)
    result = flatten_stack(stack)
    assert result.shape == stack.shape


def test_flatten_stack_with_failure():
    """Verify flatten_stack raises ValueError if any frame fails."""
    stack = np.random.rand(3, 64, 64)
    stack[1] = np.full((64, 64), np.nan)

    with pytest.raises(ValueError):
        # Replace the logic in flatten_stack to raise on None if needed
        flatten_stack(stack)


def test_flatten_stack_zero_frames():
    """Check flatten_stack raises ValueError on empty input stack."""
    stack = np.empty((0, 64, 64))
    with pytest.raises(ValueError):
        flatten_stack(stack)


def test_flatten_stack_single_frame():
    """Ensure flatten_stack works correctly with a single-frame stack."""
    stack = np.random.rand(1, 64, 64)
    result = flatten_stack(stack)
    assert result.shape == (1, 64, 64)


def test_flatten_afm_frame_logs_filename(tmp_path, caplog):
    """Check flatten_afm_frame logs the filename and frame shape info."""
    # Arrange
    frame = np.random.rand(32, 32)
    test_filename = tmp_path / "my_test_frame"
    frame_name_str = str(test_filename.name)

    # Act
    # Ensure we listen to the exact logger used in the module
    with caplog.at_level("INFO", logger=image_processing.logger.name):
        _ = image_processing.flatten_afm_frame(frame, filename=frame_name_str)

    # Assert
    logs = "\n".join(caplog.messages)
    assert f"Flattening frame '{frame_name_str}'" in logs
    assert "shape=(32, 32)" in logs
