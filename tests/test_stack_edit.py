"""Tests for playNano.processing.stack_edit module."""

import numpy as np
import pytest

from playNano.processing.stack_edit import (
    drop_frame_range,
    drop_frames,
    register_stack_edit_processing,
    select_frames,
)


@pytest.fixture
def sample_stack():
    """3D sample array (5 frames of 2x2)."""
    return np.arange(20, dtype=np.float32).reshape(5, 2, 2)


# ---------------------------------------------------------------------
# drop_frames
# ---------------------------------------------------------------------


def test_drop_frames_removes_correct_frames(sample_stack):
    """Test that correct frames are dropped."""
    result = drop_frames(sample_stack, [1, 3])
    assert result.shape[0] == 3
    # Should keep frames 0, 2, 4
    np.testing.assert_array_equal(result, sample_stack[[0, 2, 4]])


def test_drop_frames_with_unsorted_and_duplicate_indices(sample_stack):
    """Test that unsorted and duplicate indicies drop the same frames."""
    result = drop_frames(sample_stack, [3, 1, 1])
    np.testing.assert_array_equal(result, sample_stack[[0, 2, 4]])


def test_drop_frames_invalid_dimensions_raises():
    """Test that ValueError is raised by drop_frames if array not 3D."""
    arr = np.zeros((4, 4))
    with pytest.raises(ValueError, match="Expected a 3D array"):
        drop_frames(arr, [1])


def test_drop_frames_out_of_bounds_index(sample_stack):
    """Test that ValueError is raised if indices out of range."""
    with pytest.raises(ValueError, match="Indices out of range"):
        drop_frames(sample_stack, [5])


def test_drop_frames_negative_index(sample_stack):
    """Test that negative indices raise a ValueError."""
    with pytest.raises(ValueError, match="Indices out of range"):
        drop_frames(sample_stack, [-1])


# ---------------------------------------------------------------------
# drop_frame_range
# ---------------------------------------------------------------------


def test_drop_frame_range_valid(sample_stack):
    """Test that drop_frame_range returns correct indices for a valid range."""
    indices = drop_frame_range(sample_stack, 1, 3)
    assert indices == [1, 2]


@pytest.mark.parametrize(
    "start,end",
    [
        (-1, 2),  # negative start
        (1, 6),  # end out of bounds
        (3, 2),  # start >= end
        (4, 4),  # equal bounds
    ],
)
def test_drop_frame_range_invalid(sample_stack, start, end):
    """Test that a ValueError is raise when a invalid range is provided."""
    with pytest.raises(ValueError, match="Invalid range"):
        drop_frame_range(sample_stack, start, end)


def test_drop_frame_range_invalid_dimensions():
    """Test that drop_frame_range raises a ValueError if array not 3D."""
    arr = np.zeros((4, 4))
    with pytest.raises(ValueError, match="Expected a 3D array"):
        drop_frame_range(arr, 0, 1)


# ---------------------------------------------------------------------
# select_frames
# ---------------------------------------------------------------------


def test_select_frames_returns_complement(sample_stack):
    """Test that select_frames returns the complement of the keep indices."""
    drop_indices = select_frames(sample_stack, [0, 2, 4])
    assert drop_indices == [1, 3]


def test_select_frames_unsorted_or_duplicate_keep_indices(sample_stack):
    """Test that select_frames handles unsorted & duplicate keep indices correctly."""
    drop_indices = select_frames(sample_stack, [4, 2, 2, 0])
    assert drop_indices == [1, 3]


def test_select_frames_invalid_dimensions():
    """Test that select_frames raises ValueError for non-3D input arrays."""
    arr = np.zeros((4, 4))
    with pytest.raises(ValueError, match="Expected a 3D array"):
        select_frames(arr, [0])


def test_select_frames_out_of_bounds(sample_stack):
    """Test that select_frames raises ValueError for out-of-bounds indices."""
    with pytest.raises(ValueError, match="Invalid frame indices"):
        select_frames(sample_stack, [0, 6])


def test_select_frames_negative_index(sample_stack):
    """Test that select_frames raises ValueError for negative indices."""
    with pytest.raises(ValueError, match="Invalid frame indices"):
        select_frames(sample_stack, [-1])


# ---------------------------------------------------------------------
# register_stack_edit_processing
# ---------------------------------------------------------------------


def test_register_stack_edit_processing_contains_expected_keys():
    """Test that register_stack_edit_processing returns expected keys and functions."""
    registry = register_stack_edit_processing()
    assert set(registry.keys()) == {"drop_frames", "drop_frame_range", "select_frames"}
    assert registry["drop_frames"] is drop_frames
    assert registry["drop_frame_range"] is drop_frame_range
    assert registry["select_frames"] is select_frames


# ---------------------------------------------------------------------
# versioned_filter decorator integration (light test)
# ---------------------------------------------------------------------


def test_versioned_filter_metadata_attached():
    """Ensure versioning decorator attached the version metadata."""
    assert hasattr(drop_frames, "__version__")
    assert drop_frames.__version__ == "0.1.0"
    assert hasattr(drop_frame_range, "__version__")
    assert hasattr(select_frames, "__version__")
