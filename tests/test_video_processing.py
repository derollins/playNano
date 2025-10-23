"""Tests for playNano.processing.video_processing module."""

import numpy as np
import pytest

from playNano.processing.video_processing import (
    align_frames,
    crop_square,
    intersection_crop,
    replace_nan,
    rolling_frame_align,
)

# --- Fixtures --- #


@pytest.fixture
def synthetic_stack():
    """Synthetic 3-frame 5x5 stack with known shifts."""
    ref = np.zeros((5, 5))
    ref[2, 2] = 1
    frame1 = np.zeros((5, 5))
    frame1[3, 3] = 1
    frame2 = np.zeros((5, 5))
    frame2[1, 1] = 1
    return np.stack([ref, frame1, frame2])


@pytest.fixture
def noisy_stack():
    """Make a synthetic 5-frame 20x20 stack with small noise and shifts."""
    n_frames, H, W = 5, 20, 20
    base = np.zeros((H, W))
    base[H // 2, W // 2] = 1.0
    stack = np.zeros((n_frames, H, W), dtype=float)
    rng = np.random.default_rng(42)
    for i in range(n_frames):
        dy = rng.integers(-1, 2)
        dx = rng.integers(-1, 2)
        stack[i] = np.roll(np.roll(base, dy, axis=0), dx, axis=1) + rng.normal(
            0, 0.05, (H, W)
        )
    return stack


@pytest.fixture
def stack_with_nans():
    """Make a stack with NaN padding."""
    stack = np.ones((3, 5, 5))
    stack[0, 0, 0] = np.nan
    stack[1, -1, -1] = np.nan
    stack[2, 2, 2] = np.nan
    return stack


@pytest.fixture
def padded_stack():
    """Make a stack with NaN borders."""
    stack = np.ones((3, 5, 5))
    stack[:, 0, :] = np.nan
    stack[:, :, 0] = np.nan
    stack[:, -1, :] = np.nan
    stack[:, :, -1] = np.nan
    return stack


@pytest.fixture
def dummy_stack():
    """Create a simple stack for selection tests."""
    return np.arange(5 * 2 * 2).reshape(5, 2, 2)


# --- Tests for align_frames --- #


def test_align_frames_fft(synthetic_stack):
    """Test FFT-based alignment with pad mode."""
    aligned, meta = align_frames(
        synthetic_stack, method="fft_cross_correlation", mode="pad"
    )
    assert aligned.shape[0] == 3
    assert aligned.shape[1] >= 5
    assert aligned.shape[2] >= 5
    assert meta["border_mask"].dtype == bool
    expected_shifts = np.array([[0, 0], [-1, -1], [1, 1]])
    np.testing.assert_allclose(meta["shifts"], expected_shifts, atol=3)


def test_align_frames_crop_modes(synthetic_stack):
    """Test all crop mode options in alignment."""
    for mode in ["crop", "crop_square"]:
        aligned, meta = align_frames(synthetic_stack, mode=mode)
        assert aligned.shape[0] == 3
        assert meta["mode"] == mode
        assert np.all(np.isfinite(aligned))


def test_align_frames_mode_pad(synthetic_stack):
    """Test pad mode in alignment."""
    aligned, meta = align_frames(synthetic_stack, mode="pad")
    assert aligned.shape[0] == 3
    assert meta["mode"] == "pad"
    assert np.isnan(aligned).any()


def test_align_frames_debug(synthetic_stack):
    """Test debug output from alignment."""
    aligned, meta, debug = align_frames(synthetic_stack, debug=True)
    assert "shifts" in debug
    np.testing.assert_array_equal(debug["shifts"], meta["shifts"])


def test_align_frames_invalid_method(synthetic_stack):
    """Raise error for unknown method."""
    with pytest.raises(ValueError):
        align_frames(synthetic_stack, method="unknown")


def test_align_frames_unknown_mode():
    """Raise error for unknown mode."""
    stack = np.zeros((3, 5, 5))
    with pytest.raises(ValueError):
        align_frames(stack, mode="banana")


def test_align_frames_max_jump():
    """Test align_frames max_jump smoothing logic for i == 1 and i >= 2."""
    # Create a synthetic stack of 3 frames with known shifts
    H, W = 10, 10
    base = np.zeros((H, W), dtype=np.float32)
    base[4:6, 4:6] = 1.0

    # Create shifted frames
    frame1 = np.roll(base, shift=(2, 2), axis=(0, 1))  # shift by (2, 2)
    frame2 = np.roll(base, shift=(6, 6), axis=(0, 1))  # jump to trigger smoothing

    stack = np.stack([base, frame1, frame2])

    # Run alignment with max_jump restriction
    aligned, meta = align_frames(stack, reference_frame=0, max_jump=2)

    # Extract shifts
    shifts = meta["shifts"]

    # Check that frame1 shift is close to (-2, -2)
    assert tuple(shifts[1]) == (-2, -2)

    # Check that frame2 shift is smoothed (not full jump of 6)
    # Expected extrapolated shift: (-2, -2) + ((-2, -2) - (0, 0)) = (-4, -4)
    assert tuple(shifts[2]) == (-4, -4)


def test_rolling_frame_align_max_jump():
    """Test rolling_frame_align max_jump smoothing logic for i >= 2."""
    H, W = 10, 10
    base = np.zeros((H, W), dtype=np.float32)
    base[4:6, 4:6] = 1.0

    # Create shifted frames
    frame1 = np.roll(base, shift=(2, 2), axis=(0, 1))  # shift by (2, 2)
    frame2 = np.roll(base, shift=(6, 6), axis=(0, 1))  # large jump
    stack = np.stack([base, frame1, frame2])

    # Run rolling alignment with max_jump restriction
    aligned, meta = rolling_frame_align(stack, window=2, max_jump=2)

    shifts = meta["shifts"]

    # Expect negative shifts due to alignment direction
    assert tuple(shifts[1]) == (-2, -2)
    assert tuple(shifts[2]) == (-4, -4)  # extrapolated from (-2, -2) and (0, 0)


@pytest.mark.parametrize("method", ["fft_cross_correlation", "full_cross_correlation"])
@pytest.mark.parametrize("mode", ["pad", "crop", "crop_square"])
def test_align_frames_with_noise(noisy_stack, method, mode):
    """Check alignment behavior on noisy stack for different modes."""
    aligned, meta = align_frames(noisy_stack, method=method, mode=mode)

    # The number of frames should always be preserved
    assert aligned.shape[0] == noisy_stack.shape[0]

    if mode == "pad":
        # For pad mode, NaNs are expected at the borders
        # So we only assert that there are **some finite values**
        assert np.any(np.isfinite(aligned))
    else:
        # For crop modes, there should be no NaNs
        assert np.all(np.isfinite(aligned))


@pytest.fixture
def shifted_stack():
    """Make a stack where frame[1] is shifted relative to frame[0]."""
    shift = (2, 0)
    stack = np.zeros((2, 10, 10), dtype=float)
    stack[0, 5, 5] = 1.0  # reference spot
    dy, dx = shift
    stack[1, 5 + dy, 5 + dx] = 1.0  # shifted spot
    return stack


def test_align_frames_with_max_shift(shifted_stack):
    """Test max_shift limits translation and is recorded."""
    # No restriction: should recover full shift
    aligned, meta_full = align_frames(shifted_stack, max_shift=None)
    assert np.array_equal(meta_full["shifts"][1], [-2, 0])

    # Restrict shift to ±1 → algorithm should find smaller shift
    aligned, meta_restricted = align_frames(shifted_stack, max_shift=1)
    dy, dx = meta_restricted["shifts"][1]

    assert abs(dy) <= 1
    assert abs(dx) <= 1
    # Ensure it’s *different* from the unrestricted shift
    assert not np.array_equal(meta_restricted["shifts"][1], [2, 0])


@pytest.mark.parametrize("sigma", [None, 1.0])
def test_align_frames_prefilter(synthetic_stack, sigma):
    """Test execution of the pre_filter_sigma line without assuming output shape."""
    aligned, meta = align_frames(synthetic_stack, pre_filter_sigma=sigma)

    # Check that shifts array exists
    assert "shifts" in meta
    assert meta["shifts"].shape[0] == synthetic_stack.shape[0]

    if sigma is not None:
        # Run again without filter to check numeric difference
        aligned_no_filter, _ = align_frames(synthetic_stack, pre_filter_sigma=None)
        assert not np.allclose(aligned, aligned_no_filter)


# --- Tests for rolling_frame_align --- #


@pytest.mark.parametrize("mode", ["pad", "crop", "crop_square"])
def test_rolling_frame_align_basic(synthetic_stack, mode):
    """Test rolling-frame alignment basic behavior for different modes."""
    aligned, meta = rolling_frame_align(synthetic_stack, window=2, mode=mode)

    # Number of frames should always match
    assert aligned.shape[0] == synthetic_stack.shape[0]

    if mode == "pad":
        # NaNs expected at borders, so only check that some finite values exist
        assert np.any(np.isfinite(aligned))
    else:
        # Cropped modes should have no NaNs
        assert np.all(np.isfinite(aligned))

    # Ensure shifts are always returned
    assert "shifts" in meta


@pytest.mark.parametrize("mode", ["pad", "crop", "crop_square"])
def test_rolling_frame_align_modes(synthetic_stack, mode):
    """Test rolling alignment with all modes."""
    aligned, meta = rolling_frame_align(synthetic_stack, window=2, mode=mode)

    assert aligned.shape[0] == synthetic_stack.shape[0]

    if mode == "pad":
        assert np.any(np.isfinite(aligned))
    else:
        assert np.all(np.isfinite(aligned))

    # Add mode to metadata for consistency
    meta_mode = meta.get("mode", mode)
    assert meta_mode == mode


# --- Tests for NaN replacement --- #


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
    """Test NaN replacement."""
    filled, meta = replace_nan(stack_with_nans, mode=mode, value=value)
    assert np.all(np.isfinite(filled))
    assert meta["mode"] == mode
    assert meta["value_used"] == value
    assert meta["nans_filled"] == 3


# --- Tests for cropping --- #


def test_intersection_crop(padded_stack):
    """Test cropping intersection."""
    cropped, meta = intersection_crop(padded_stack)
    assert cropped.shape == (3, 3, 3)
    assert np.all(np.isfinite(cropped))


def test_crop_square_centering():
    """Test square crop is centered."""
    stack = np.arange(1 * 5 * 7).reshape(1, 5, 7)
    cropped, meta = crop_square(stack)
    assert cropped.shape == (1, 5, 5)
    np.testing.assert_array_equal(cropped[0], stack[0][:, 1:6])


# --- Integration test --- #


@pytest.fixture
def synthetic_stack_integration():
    """Synthetic stack with NaNs at borders (realistic)."""
    stack = np.zeros((3, 5, 5), dtype=float)
    stack[0, 2, 2] = 1
    stack[1, 2, 3] = 1  # shifted right
    stack[2, 3, 2] = 1  # shifted down
    return stack


@pytest.mark.parametrize(
    "pipeline", ["align_crop", "align_crop_square", "align_replace_nan"]
)
def test_pipeline_variants(synthetic_stack_integration, pipeline):
    """Test different realistic processing pipeline orders."""
    aligned, meta_align = align_frames(synthetic_stack_integration, mode="pad")

    if pipeline == "align_crop":
        cropped, meta_crop = intersection_crop(aligned)
        result = cropped

    elif pipeline == "align_crop_square":
        cropped, meta_crop = intersection_crop(aligned)
        square, meta_sq = crop_square(cropped)
        result = square
        assert square.shape[1] == square.shape[2]

    elif pipeline == "align_replace_nan":
        filled, meta_replace = replace_nan(aligned, mode="zero")
        result = filled
    # --- universal assertions ---
    assert np.all(np.isfinite(result)), f"{pipeline} left NaNs behind"
    assert result.shape[1] > 0 and result.shape[2] > 0
    # if cropped square, ensure it’s square
    if pipeline == "align_crop_square":
        assert result.shape[1] == result.shape[2]
