"""Tests for the playNano processing.image_processing module."""

import numpy as np

from playNano.processing import filters
from playNano.processing.filters import topostats_flatten


def test_topostats_topostats_flatten_valid():
    """Check topostats_flatten returns array with same shape as input."""
    frame = np.random.rand(64, 64) * 10  # Fake AFM data
    result = topostats_flatten(frame, filename="test_frame")
    assert isinstance(result, np.ndarray)
    assert result.shape == frame.shape


def test_topostats_topostats_flatten_empty():
    """Verify topostats_flatten returns None for empty input array."""
    frame = np.array([])
    result = topostats_flatten(frame)
    assert result is None


def test_topostats_flatten_with_nans():
    """Ensure topostats_flatten returns None for all-NaN arrays."""
    frame = np.full((64, 64), np.nan)
    result = topostats_flatten(frame)
    assert result is None


def test_topostats_flatten_custom_config():
    """Test topostats_flatten works with a custom filter configuration."""
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
    result = topostats_flatten(frame, filter_config=custom_config)
    assert isinstance(result, np.ndarray)


def test_topostats_flatten_logs_filename(tmp_path, caplog):
    """Check topostats_flatten logs the filename and frame shape info."""
    # Arrange
    frame = np.random.rand(32, 32)
    test_filename = tmp_path / "my_test_frame"
    frame_name_str = str(test_filename.name)

    # Act
    # Ensure we listen to the exact logger used in the module
    with caplog.at_level("INFO", logger=filters.logger.name):
        _ = filters.topostats_flatten(frame, filename=frame_name_str)

    # Assert
    logs = "\n".join(caplog.messages)
    assert f"Flattening frame '{frame_name_str}'" in logs
    assert "shape=(32, 32)" in logs
