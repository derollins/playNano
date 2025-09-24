import numpy as np
import pytest

from playNano.processing.video_processing import (
    align_frames,
    drop_frames,
    select_frame_range,
    select_frames,
)

# --- Tests for frame alignment function --- #


@pytest.fixture
def synthetic_stack():
    """
    Create a small 3-frame 5x5 synthetic stack with known shifts.

    Frame 0: reference
    Frame 1: shifted down +1, right +1
    Frame 2: shifted up -1, left -1
    """
    ref = np.zeros((5, 5))
    ref[2, 2] = 1  # single bright pixel as feature

    frame1 = np.zeros((5, 5))
    frame1[3, 3] = 1  # shifted +1,+1

    frame2 = np.zeros((5, 5))
    frame2[1, 1] = 1  # shifted -1,-1

    stack = np.stack([ref, frame1, frame2])
    return stack


def test_align_frames_fft(synthetic_stack):
    """Test FFT-based frame alignment on synthetic stack."""
    aligned, meta = align_frames(
        synthetic_stack, method="fft_cross_correlation", mode="pad"
    )

    # Should return aligned stack with shape >= original
    assert aligned.shape[0] == 3
    assert aligned.shape[1] >= 5
    assert aligned.shape[2] >= 5

    # Check shifts
    expected_shifts = np.array([[0, 0], [-1, -1], [1, 1]])
    np.testing.assert_array_equal(meta["shifts"], expected_shifts)

    # Border mask should be boolean array
    assert meta["border_mask"].dtype == bool
    assert meta["border_mask"].shape == aligned.shape[1:]


def test_align_frames_debug(synthetic_stack):
    """Test that debug outputs are returned when debug=True."""
    aligned, meta, debug = align_frames(
        synthetic_stack, method="fft_cross_correlation", mode="pad", debug=True
    )
    assert "aligned_stack" in debug
    assert "shifts" in debug
    np.testing.assert_array_equal(debug["shifts"], meta["shifts"])


def test_align_frames_full_cross_correlation(synthetic_stack):
    """Test full cross-correlation alignment on synthetic stack."""
    aligned, meta = align_frames(
        synthetic_stack, method="full_cross_correlation", mode="pad"
    )
    expected_shifts = np.array([[0, 0], [-1, -1], [1, 1]])
    np.testing.assert_array_equal(meta["shifts"], expected_shifts)


def test_align_frames_unknown_mode():
    """Raise ValueError for unknown mode string."""
    stack = np.zeros((3, 5, 5))
    with pytest.raises(ValueError) as excinfo:
        align_frames(stack, mode="banana")
    assert "Unknown mode: banana" in str(excinfo.value)


def test_align_frames_invalid_method(synthetic_stack):
    """Raise ValueError for unknown alignment method."""
    with pytest.raises(ValueError):
        align_frames(synthetic_stack, method="unknown")


@pytest.fixture
def noisy_stack():
    """Create a synthetic stack with random noise and small shifts."""
    n_frames, H, W = 5, 20, 20
    base = np.zeros((H, W))
    # Add a central peak
    base[H // 2, W // 2] = 1.0

    stack = np.zeros((n_frames, H, W), dtype=float)
    rng = np.random.default_rng(42)
    for i in range(n_frames):
        dy = rng.integers(-1, 2)
        dx = rng.integers(-1, 2)
        frame = np.roll(np.roll(base, dy, axis=0), dx, axis=1)
        # Add small Gaussian noise
        frame += rng.normal(scale=0.05, size=frame.shape)
        stack[i] = frame
    return stack


@pytest.mark.parametrize("method", ["fft_cross_correlation", "full_cross_correlation"])
def test_align_frames_with_noise_methods(noisy_stack, method):
    """Check that alignment with noisy stack preserves finite values and shape."""
    aligned, meta = align_frames(noisy_stack, method=method, mode="pad")
    n_frames, H, W = noisy_stack.shape

    # Shape check
    assert aligned.shape[0] == n_frames
    assert aligned.shape[1] >= H
    assert aligned.shape[2] >= W

    # Check that each frame's original pixels are finite in the aligned stack
    max_dy_neg = meta["shifts"][:, 0].min()
    max_dx_neg = meta["shifts"][:, 1].min()

    for i in range(n_frames):
        dy, dx = meta["shifts"][i]
        y_start = dy - max_dy_neg
        x_start = dx - max_dx_neg
        subframe = aligned[i, y_start : y_start + H, x_start : x_start + W]
        assert np.all(np.isfinite(subframe))


# --- Tests for frame selection functions --- #


@pytest.fixture
def dummy_stack():
    # 5 frames, 2x2 pixels each
    return np.arange(5 * 2 * 2).reshape(5, 2, 2)


def test_select_frames_basic(dummy_stack):  #
    """Test basic frame selection by indices."""
    subset, meta = select_frames(dummy_stack, [0, 2, 4])
    assert subset.shape[0] == 3
    # Check the actual frames retained
    np.testing.assert_array_equal(subset[0], dummy_stack[0])
    np.testing.assert_array_equal(subset[1], dummy_stack[2])
    np.testing.assert_array_equal(subset[2], dummy_stack[4])
    # Metadata checks
    assert meta["original_n_frames"] == 5
    assert meta["selected_indices"] == [0, 2, 4]
    assert meta["new_n_frames"] == 3


def test_select_frames_invalid_index(dummy_stack):
    """Raise IndexError when selecting invalid frame indices."""
    with pytest.raises(IndexError):
        select_frames(dummy_stack, [10])


def test_drop_frames_basic(dummy_stack):
    """Test dropping frames by indices."""
    subset, meta = drop_frames(dummy_stack, [1, 3])
    assert subset.shape[0] == 3
    # Ensure the dropped frames are gone
    kept_indices = [0, 2, 4]
    expected = dummy_stack[kept_indices]
    np.testing.assert_array_equal(subset, expected)
    # Metadata checks
    assert meta["dropped_indices"] == [1, 3]
    assert meta["new_n_frames"] == 3


def test_drop_frames_invalid_index(dummy_stack):
    """Raise IndexError when dropping invalid frame indices."""
    with pytest.raises(IndexError):
        drop_frames(dummy_stack, [-1])


def test_select_frame_range_basic(dummy_stack):
    """Test selecting a contiguous range of frames."""
    subset, meta = select_frame_range(dummy_stack, 1, 4)
    assert subset.shape[0] == 3
    expected = dummy_stack[1:4]
    np.testing.assert_array_equal(subset, expected)
    # Metadata checks
    assert meta["kept_indices"] == [1, 2, 3]
    assert meta["dropped_indices"] == [0, 4]
    assert meta["new_n_frames"] == 3


@pytest.mark.parametrize("start,end", [(0, 0), (4, 2), (5, 6)])
def test_select_frame_range_invalid(dummy_stack, start, end):
    """Raise ValueError when selecting an invalid frame range."""
    with pytest.raises(ValueError):
        select_frame_range(dummy_stack, start, end)


def test_select_frame_range_full(dummy_stack):
    """Selecting the full range returns original stack unchanged."""
    subset, meta = select_frame_range(dummy_stack, 0, 5)
    np.testing.assert_array_equal(subset, dummy_stack)
    assert meta["new_n_frames"] == 5
    assert meta["dropped_indices"] == []
