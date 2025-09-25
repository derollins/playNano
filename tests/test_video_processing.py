import numpy as np
import pytest

from playNano.processing.video_processing import (
    align_frames,
    crop_square,
    drop_frames,
    intersection_crop,
    replace_nan,
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

    assert aligned.shape[0] == 3
    assert aligned.shape[1] >= 5
    assert aligned.shape[2] >= 5

    # Allow tolerance because FFT correlation can pick slightly different peaks
    expected_shifts = np.array([[0, 0], [-1, -1], [1, 1]])
    np.testing.assert_allclose(meta["shifts"], expected_shifts, atol=3)

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


# --- Test for cropping and NaN filling functions --- #


# -----------------------------
# NaN replacement tests
# -----------------------------
@pytest.fixture
def stack_with_nans():
    """Stack with NaN padding."""
    stack = np.ones((3, 5, 5))
    stack[0, 0, 0] = np.nan
    stack[1, -1, -1] = np.nan
    stack[2, 2, 2] = np.nan
    return stack


@pytest.mark.parametrize(
    "mode,value",
    [
        ("zero", None),
        ("mean", None),
        ("median", None),
        ("global_mean", None),
        ("constant", 42.0),
    ],
)
def test_replace_nan(stack_with_nans, mode, value):
    """Test that NaNs are replaced correctly for all modes."""
    filled = replace_nan(stack_with_nans, mode=mode, value=value)
    # Ensure no NaNs remain
    assert np.all(np.isfinite(filled))

    if mode == "zero":
        # Zero mode should replace all NaNs with 0
        nan_pos = np.isnan(stack_with_nans)
        assert np.all(filled[nan_pos] == 0)

    elif mode == "mean":
        # Frame-wise mean
        for i in range(stack_with_nans.shape[0]):
            nan_pos = np.isnan(stack_with_nans[i])
            frame_mean = np.nanmean(stack_with_nans[i])
            assert np.allclose(filled[i][nan_pos], frame_mean)

    elif mode == "median":
        # Frame-wise median
        for i in range(stack_with_nans.shape[0]):
            nan_pos = np.isnan(stack_with_nans[i])
            frame_median = np.nanmedian(stack_with_nans[i])
            assert np.allclose(filled[i][nan_pos], frame_median)

    elif mode == "global_mean":
        # Single mean across entire stack
        nan_pos = np.isnan(stack_with_nans)
        global_mean = np.nanmean(stack_with_nans)
        assert np.allclose(filled[nan_pos], global_mean)

    elif mode == "constant":
        # Replace with provided constant
        nan_pos = np.isnan(stack_with_nans)
        assert np.all(filled[nan_pos] == value)


# -----------------------------
# Cropping tests
# -----------------------------
@pytest.fixture
def padded_stack():
    """Create 3-frame stack with NaN borders."""
    stack = np.ones((3, 5, 5))
    stack[:, 0, :] = np.nan
    stack[:, :, 0] = np.nan
    stack[:, -1, :] = np.nan
    stack[:, :, -1] = np.nan
    return stack


def test_intersection_crop(padded_stack):
    """Test crop NaN borders."""
    cropped, meta = intersection_crop(padded_stack)
    assert cropped.shape == (3, 3, 3)
    assert not np.any(np.isnan(cropped))
    assert meta
    assert meta


def test_crop_square():
    """Test crop stack to square region."""
    stack = np.ones((2, 6, 4))
    cropped, meta = crop_square(stack)
    assert cropped.shape == (2, 4, 4)


def test_crop_square_centering():
    """Check square cropping is centered."""
    stack = np.arange(1 * 5 * 7).reshape(1, 5, 7)
    cropped, meta = crop_square(stack)
    assert cropped.shape == (1, 5, 5)
    np.testing.assert_array_equal(cropped[0], stack[0][:, 1:6])
    assert meta


# -----------------------------
# Integration test
# -----------------------------
@pytest.fixture
def synthetic_stack_integration():
    """Small synthetic stack with NaNs at borders."""
    stack = np.zeros((3, 5, 5))
    stack[1, 1, 1] = 1  # feature
    stack[2, 3, 3] = 1
    stack[0, 0, 0] = np.nan
    return stack


def test_full_pipeline(synthetic_stack_integration):
    """Integration: align, replace_nan, crop."""
    # Alignment
    aligned, meta_align = align_frames(
        synthetic_stack_integration, method="fft_cross_correlation", mode="pad"
    )

    # NaN replacement
    filled = replace_nan(aligned, mode="zero")
    assert np.all(np.isfinite(filled))

    # Cropping
    cropped, meta_crop = intersection_crop(filled)
    # Cropped should never be bigger than the aligned frames
    assert cropped.shape[1] <= aligned.shape[1]
    assert cropped.shape[2] <= aligned.shape[2]
    assert cropped.shape[0] == aligned.shape[0]
    assert meta_crop

    # Square crop
    square, meta_sqcrop = crop_square(cropped)
    assert square.shape[1] == square.shape[2]
    assert square.shape[0] == aligned.shape[0]
    assert meta_sqcrop

    # Metadata checks
    assert meta_align


def test_crop_then_replace_nan(synthetic_stack_integration):
    """Test cropping first, then replacing NaNs."""
    # Alignment (may introduce NaNs)
    aligned, meta_align = align_frames(
        synthetic_stack_integration, method="fft_cross_correlation", mode="pad"
    )

    # Crop intersection first
    cropped, meta_crop = intersection_crop(aligned)
    square, meta_sq = crop_square(cropped)

    # Should still have NaNs if alignment added padding
    assert np.any(np.isnan(square))

    # Now replace NaNs
    filled = replace_nan(square, mode="zero")
    assert np.all(np.isfinite(filled))

    # Shapes should be consistent
    assert filled.shape == square.shape


# -----------------------------
# Frame selection tests (optional integration)
# -----------------------------
@pytest.fixture
def dummy_stack():
    """Simple stack for selection tests."""
    return np.arange(5 * 2 * 2).reshape(5, 2, 2)


def test_select_and_drop_frames(dummy_stack):
    """Test select_frames and drop_frames integration."""
    selected, meta1 = select_frames(dummy_stack, [0, 2, 4])
    dropped, meta2 = drop_frames(dummy_stack, [1, 3])
    np.testing.assert_array_equal(selected, dropped)


# --- Tests for frame selection functions --- #


@pytest.fixture
def dummy_stack_select():
    # 5 frames, 2x2 pixels each
    return np.arange(5 * 2 * 2).reshape(5, 2, 2)


def test_select_frames_basic(dummy_stack_select):  #
    """Test basic frame selection by indices."""
    subset, meta = select_frames(dummy_stack_select, [0, 2, 4])
    assert subset.shape[0] == 3
    # Check the actual frames retained
    np.testing.assert_array_equal(subset[0], dummy_stack_select[0])
    np.testing.assert_array_equal(subset[1], dummy_stack_select[2])
    np.testing.assert_array_equal(subset[2], dummy_stack_select[4])
    # Metadata checks
    assert meta["original_n_frames"] == 5
    assert meta["selected_indices"] == [0, 2, 4]
    assert meta["new_n_frames"] == 3


def test_select_frames_invalid_index(dummy_stack_select):
    """Raise IndexError when selecting invalid frame indices."""
    with pytest.raises(IndexError):
        select_frames(dummy_stack_select, [10])


def test_drop_frames_basic(dummy_stack_select):
    """Test dropping frames by indices."""
    subset, meta = drop_frames(dummy_stack_select, [1, 3])
    assert subset.shape[0] == 3
    # Ensure the dropped frames are gone
    kept_indices = [0, 2, 4]
    expected = dummy_stack_select[kept_indices]
    np.testing.assert_array_equal(subset, expected)
    # Metadata checks
    assert meta["dropped_indices"] == [1, 3]
    assert meta["new_n_frames"] == 3


def test_drop_frames_invalid_index(dummy_stack_select):
    """Raise IndexError when dropping invalid frame indices."""
    with pytest.raises(IndexError):
        drop_frames(dummy_stack_select, [-1])


def test_select_frame_range_basic(dummy_stack_select):
    """Test selecting a contiguous range of frames."""
    subset, meta = select_frame_range(dummy_stack_select, 1, 4)
    assert subset.shape[0] == 3
    expected = dummy_stack_select[1:4]
    np.testing.assert_array_equal(subset, expected)
    # Metadata checks
    assert meta["kept_indices"] == [1, 2, 3]
    assert meta["dropped_indices"] == [0, 4]
    assert meta["new_n_frames"] == 3


@pytest.mark.parametrize("start,end", [(0, 0), (4, 2), (5, 6)])
def test_select_frame_range_invalid(dummy_stack_select, start, end):
    """Raise ValueError when selecting an invalid frame range."""
    with pytest.raises(ValueError):
        select_frame_range(dummy_stack_select, start, end)


def test_select_frame_range_full(dummy_stack_select):
    """Selecting the full range returns original stack unchanged."""
    subset, meta = select_frame_range(dummy_stack_select, 0, 5)
    np.testing.assert_array_equal(subset, dummy_stack_select)
    assert meta["new_n_frames"] == 5
    assert meta["dropped_indices"] == []


@pytest.mark.parametrize(
    "n_frames,H,W",
    [
        (1, 1, 1),  # single pixel
        (1, 5, 5),  # single frame
        (3, 1, 1),  # multiple frames, single pixel
    ],
)
def test_align_frames_small_stacks(n_frames, H, W):
    """Test alignment with very small stacks to ensure no errors occur."""
    stack = np.ones((n_frames, H, W))
    aligned, meta = align_frames(stack, mode="pad")
    # Shapes must be preserved
    assert aligned.shape[0] == n_frames
    assert aligned.shape[1] >= H
    assert aligned.shape[2] >= W
    # All pixels must remain finite
    assert np.all(np.isfinite(aligned))


@pytest.mark.parametrize("shift_range", [(-10, 10), (-20, 20)])
def test_align_frames_extreme_shifts(shift_range):
    """Test that large random shifts do not break alignment or padding."""
    n_frames, H, W = 5, 10, 10
    base = np.zeros((H, W))
    base[H // 2, W // 2] = 1.0
    rng = np.random.default_rng(123)
    stack = np.zeros((n_frames, H, W))
    for i in range(n_frames):
        dy = rng.integers(*shift_range)
        dx = rng.integers(*shift_range)
        frame = np.roll(np.roll(base, dy, axis=0), dx, axis=1)
        stack[i] = frame

    aligned, meta = align_frames(stack, mode="pad")
    # Original pixels must remain finite
    max_dy_neg = meta["shifts"][:, 0].min()
    max_dx_neg = meta["shifts"][:, 1].min()
    for i in range(n_frames):
        dy, dx = meta["shifts"][i]
        y_start = dy - max_dy_neg
        x_start = dx - max_dx_neg
        subframe = aligned[i, y_start : y_start + H, x_start : x_start + W]
        assert np.all(np.isfinite(subframe))


@pytest.mark.parametrize(
    "n_frames,H,W",
    [
        (10, 50, 50),
        (20, 100, 100),
    ],
)
def test_align_frames_large_random_stack(n_frames, H, W):
    """Stress test: large random stacks with small shifts and noise."""
    rng = np.random.default_rng(999)
    base = np.zeros((H, W))
    base[H // 2, W // 2] = 1.0
    stack = np.zeros((n_frames, H, W))
    for i in range(n_frames):
        dy = rng.integers(-1, 2)
        dx = rng.integers(-1, 2)
        frame = np.roll(np.roll(base, dy, axis=0), dx, axis=1)
        frame += rng.normal(scale=0.01, size=(H, W))
        stack[i] = frame

    aligned, meta = align_frames(stack, mode="pad")
    assert aligned.shape[0] == n_frames
    assert aligned.shape[1] >= H
    assert aligned.shape[2] >= W
    # Check that all pixels inside border_mask are finite
    for i in range(n_frames):
        dy, dx = meta["shifts"][i]
        max_dy_neg = meta["shifts"][:, 0].min()
        max_dx_neg = meta["shifts"][:, 1].min()
        y_start = dy - max_dy_neg
        x_start = dx - max_dx_neg
        subframe = aligned[i, y_start : y_start + H, x_start : x_start + W]
        assert np.all(np.isfinite(subframe))
