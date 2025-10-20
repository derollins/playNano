"""
Video processing functions for AFM time-series (stacks of frames).

This module provides functions that operate on 3D numpy arrays
(time-series of 2D AFM frames). These include:

- Frame alignment to compensate for drift
- Cropping and padding utilities
- Temporal (time-domain) filters
- Future extensions such as spatio-temporal denoising

All functions follow a NumPy-style API: input stacks are 3D arrays
with shape (n_frames, height, width). Outputs are processed
stacks and a metadata dictionary.
"""

from collections import deque
from typing import Callable, Literal, Optional

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d, fftconvolve
from skimage.registration import phase_cross_correlation

from playNano.utils.param_utils import param_conditions
from playNano.utils.versioning import versioned_filter

# -----------------------------------------------------------------------------#
# Alignment
# -----------------------------------------------------------------------------#


@versioned_filter("0.1.0")
def align_frames(
    stack: np.ndarray,
    reference_frame: int = 0,
    method: str = "fft_cross_correlation",
    mode: str = "pad",
    debug: bool = False,
    max_shift: Optional[int] = None,
    pre_filter_sigma: Optional[float] = None,
    max_jump: Optional[int] = None,
):
    """
    Align a stack of AFM frames to a reference frame using integer-pixel shifts.

    Alignment is performed using either FFT-based or full cross-correlation.
    Jump smoothing prevents abrupt unrealistic displacements between consecutive
    frames by limiting the change in shift relative to the previous frame.

    Parameters
    ----------
    stack : np.ndarray[float]
        3D array of shape (n_frames, height, width) containing the input AFM image
        stack.
    reference_frame : int, optional
        Index of the frame to use as the alignment reference (default 0). Must be
        within [0, n_frames-1].
    method : {"fft_cross_correlation", "full_cross_correlation"}, optional
        Alignment method (default "fft_cross_correlation"). FFT-based cross-correlation
        is generally faster and uses less memory for large frames.
    mode : {"pad", "crop", "crop_square"}, optional
        How to handle borders after shifting:
        - "pad": keep all frames with NaN padding (default)
        - "crop": crop to intersection of all frames
        - "crop_square": crop to largest centered square
    debug : bool, optional
        If True, returns additional diagnostic outputs.
    max_shift : int, optional
        Maximum allowed shift in pixels. Detected shifts are clipped to this range.
    pre_filter_sigma : float, optional
        Standard deviation of Gaussian filter applied to frames before
        cross-correlation.
    max_jump : int, optional
        Maximum allowed change in shift between consecutive frames. If exceeded, the
        shift is replaced by a linear extrapolation from the previous two frames.

    Returns
    -------
    aligned_stack : np.ndarray[float]
        Aligned 3D stack of frames. Shape may be larger than input to accommodate all
        shifts.
    metadata : dict
        Dictionary containing alignment information:
        - "reference_frame": int, index of the reference frame
        - "method": str, the alignment method used
        - "mode": str, border approach used
        - "shifts": np.ndarray of shape (n_frames, 2), detected (dy, dx) shifts
        - "original_shape": tuple of (height, width)
        - "aligned_shape": tuple of (height, width) of the output canvas
        - "border_mask": np.ndarray[bool], True where valid frame pixels exist
        - "pre_filter_sigma": float or None
        - "max_shift": int or None
        - "max_jump": int or None
    debug_outputs : dict, optional
        Returned only if ``debug=True``. Contains:
        - "shifts": copy of the shifts array.

    Raises
    ------
    ValueError
        If ``stack.ndim`` is not 3.
    ValueError
        If ``method`` is not one of {"fft_cross_correlation", "full_cross_correlation"}.
    ValueError
        If ``reference_frame`` is not in the range [0, n_frames-1].

    Notes
    -----
    - Using ``fft_cross_correlation`` reduces memory usage compared to full
    cross-correlation because it leverages the FFT algorithm and avoids creating
    large full correlation matrices.
    - Padding with NaNs allows all frames to be placed without clipping, but may
    increase memory usage for large shifts.
    - The function does not interpolate subpixel shifts; all shifts are integer-valued.

    Examples
    --------
    >>> import numpy as np
    >>> from playNano.processing.video_processing import align_frames
    >>> stack = np.random.rand(10, 200, 200)  # 10 frames of 200x200 pixels
    >>> aligned_stack, metadata = align_frames(stack, reference_frame=0)
    >>> aligned_stack.shape
    (10, 210, 210)  # padded to accommodate shifts
    >>> metadata['shifts']
    array([[ 0,  0],
        [ 1, -2],
        ...])
    """
    stack = stack.astype(np.float32, copy=False)

    if stack.ndim != 3:
        raise ValueError(
            f"stack must be a 3D array (n_frames, H, W), got shape {stack.shape}"
        )

    n_frames, H, W = stack.shape

    # Validate reference_frame
    if not (0 <= reference_frame < n_frames):
        raise ValueError(
            f"reference_frame must be in [0, {n_frames-1}], got {reference_frame}"
        )

    # Preprocess reference frame
    ref = stack[reference_frame]
    if pre_filter_sigma is not None:
        ref = gaussian_filter(ref, sigma=pre_filter_sigma)
    ref0 = ref.copy()
    ref0 -= np.mean(ref0)

    shifts = np.zeros((n_frames, 2), dtype=int)

    for i in range(n_frames):
        if i == reference_frame:
            continue

        frame = stack[i]
        if pre_filter_sigma is not None:
            frame = gaussian_filter(frame, sigma=pre_filter_sigma)

        # Compute shift
        if method == "fft_cross_correlation":
            frame0 = frame.copy()
            frame0 -= np.mean(frame0)
            cc = fftconvolve(frame0[::-1, ::-1], ref0, mode="full")
            y_center, x_center = H - 1, W - 1
        elif method == "full_cross_correlation":
            cc = correlate2d(
                ref, frame, mode="full", boundary="fill", fillvalue=0
            ).astype(np.float32)
            y_center, x_center = H - 1, W - 1
        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply max_shift restriction
        if max_shift is not None:
            y_min = max(0, y_center - max_shift)
            y_max = min(cc.shape[0], y_center + max_shift + 1)
            x_min = max(0, x_center - max_shift)
            x_max = min(cc.shape[1], x_center + max_shift + 1)
            cc_window = cc[y_min:y_max, x_min:x_max]
            y_rel, x_rel = np.unravel_index(np.argmax(cc_window), cc_window.shape)
            dy = (y_min + y_rel) - y_center
            dx = (x_min + x_rel) - x_center
        else:
            y_max_idx, x_max_idx = np.unravel_index(np.argmax(cc), cc.shape)
            dy = y_max_idx - y_center
            dx = x_max_idx - x_center

        # Jump smoothing (linear extrapolation from last 2 shifts)
        if max_jump is not None:
            if i == 1:
                dy_prev, dx_prev = shifts[i - 1]
                dy = dy_prev + np.clip(dy - dy_prev, -max_jump, max_jump)
                dx = dx_prev + np.clip(dx - dx_prev, -max_jump, max_jump)
            elif i >= 2:
                prev_shift = shifts[i - 1]
                prev_prev_shift = shifts[i - 2]
                expected = prev_shift + (prev_shift - prev_prev_shift)
                if abs(dy - prev_shift[0]) > max_jump:
                    dy = int(expected[0])
                if abs(dx - prev_shift[1]) > max_jump:
                    dx = int(expected[1])

        shifts[i] = (int(dy), int(dx))

    # Pad canvas to fit all shifted frames
    max_dy_pos, max_dy_neg = shifts[:, 0].max(), shifts[:, 0].min()
    max_dx_pos, max_dx_neg = shifts[:, 1].max(), shifts[:, 1].min()
    new_H = H + max_dy_pos - max_dy_neg
    new_W = W + max_dx_pos - max_dx_neg

    # Allocate aligned stack
    aligned_stack = np.empty((n_frames, new_H, new_W), dtype=stack.dtype)
    aligned_stack.fill(np.nan)
    border_mask = np.zeros((new_H, new_W), dtype=bool)

    for i in range(n_frames):
        dy, dx = shifts[i]
        y_start = dy - max_dy_neg
        x_start = dx - max_dx_neg
        aligned_stack[i, y_start : y_start + H, x_start : x_start + W] = stack[i]
        border_mask[y_start : y_start + H, x_start : x_start + W] = True

    metadata = {
        "reference_frame": reference_frame,
        "method": method,
        "mode": mode,
        "shifts": shifts,
        "original_shape": (H, W),
        "aligned_shape": (new_H, new_W),
        "border_mask": border_mask,
        "pre_filter_sigma": pre_filter_sigma,
        "max_shift": max_shift,
        "max_jump": max_jump,
    }

    # Apply cropping according to mode
    if mode == "crop":
        aligned_stack, crop_meta = intersection_crop(aligned_stack)
        metadata["crop"] = crop_meta
        if "bounds" in crop_meta:
            y_min, y_max, x_min, x_max = crop_meta["bounds"]
            border_mask = border_mask[y_min : y_max + 1, x_min : x_max + 1]
            metadata["border_mask"] = border_mask
    elif mode == "crop_square":
        aligned_stack, crop_meta = crop_square(aligned_stack)
        metadata["crop"] = crop_meta
        H_sq, W_sq = aligned_stack.shape[1:]
        r_start, c_start = crop_meta["offset"]
        border_mask = border_mask[r_start : r_start + H_sq, c_start : c_start + W_sq]
        metadata["border_mask"] = border_mask
    elif mode != "pad":
        raise ValueError(f"Unknown mode: {mode}")

    if debug:
        return aligned_stack, metadata, {"shifts": shifts.copy()}

    return aligned_stack, metadata


@versioned_filter("0.1.0")
def rolling_frame_align(
    stack: np.ndarray,
    window: int = 5,
    mode: str = "pad",
    debug: bool = False,
    max_shift: Optional[int] = None,
    pre_filter_sigma: Optional[float] = None,
    max_jump: Optional[int] = None,
):
    """
    Align a stack of AFM frames using a rolling reference and integer pixel shifts.

    This function computes frame-to-frame shifts relative to a rolling reference
    (average of the last `window` aligned frames) using phase cross-correlation.
    Each frame is then placed on a canvas large enough to accommodate all shifts.
    Optional jump smoothing prevents sudden unrealistic displacements between
    consecutive frames, and optional Gaussian pre-filtering can improve correlation
    robustness for noisy data.

    Parameters
    ----------
    stack : np.ndarray[float]
        3D array of shape (n_frames, height, width) containing the image frames.
    window : int, optional
        Number of previous aligned frames to average when building the rolling
        reference. Default is 5.
    mode : {"pad", "crop", "crop_square"}, optional
        How to handle borders after shifting:
        - "pad": keep all frames with NaN padding (default)
        - "crop": crop to intersection of all frames
        - "crop_square": crop to largest centered square
    debug : bool, optional
        If True, returns additional diagnostic outputs such as the rolling reference
        frames. Default is False.
    max_shift : int, optional
        Maximum allowed shift in pixels along either axis. Detected shifts are clipped.
        Default is None (no clipping).
    pre_filter_sigma : float, optional
        Standard deviation of Gaussian filter applied to both reference and moving
        frames prior to cross-correlation. Helps reduce noise. Default is None.
    max_jump : int, optional
        Maximum allowed jump in pixels between consecutive frame shifts. If exceeded,
        the shift is replaced by a linear extrapolation from the previous two shifts.
        Default is None (no jump smoothing).

    Returns
    -------
    aligned_stack : np.ndarray[float]
        3D array of shape (n_frames, canvas_height, canvas_width) containing the
        aligned frames. NaN values indicate areas outside the original frames after
        alignment.
    metadata : dict
        Dictionary containing alignment information:
        - "window": int, rolling reference window used
        - "method": str, alignment method used
        - "mode": str, border approach used
        - "shifts": ndarray of shape (n_frames, 2), detected integer shifts (dy, dx)
        - "original_shape": tuple of (height, width)
        - "aligned_shape": tuple of (canvas_height, canvas_width)
        - "border_mask": ndarray of shape (canvas_height, canvas_width), True where
          valid pixels exist
        - "pre_filter_sigma": float or None
        - "max_shift": int or None
        - "max_jump": int or None
    debug_outputs : dict, optional
        Returned only if `debug=True`. Contains:
        - "shifts": copy of the detected shifts array
        - "aligned_refs": deque of indices used for rolling reference

    Raises
    ------
    ValueError
        If ``stack.ndim`` is not 3.
    ValueError
        If ``window`` < 1.

    Notes
    -----
    - The rolling reference is computed using the last `window` aligned frames,
      ignoring NaN pixels.
    - Shifts are integer-valued; no subpixel interpolation is performed.
    - Padding ensures all frames fit without clipping, but increases memory usage.
    - Internally, a deque ``aligned_refs`` tracks which patches of which frames
      contribute to the rolling reference. Each entry stores:
        (frame_index, y0c, y1c, x0c, x1c, fy0, fy1, fx0, fx1),
      i.e. both the region of the canvas updated and the corresponding slice in
      the original frame. This allows exact removal of old contributions from
      ``rolling_sum`` and ``rolling_count`` when the window is exceeded, ensuring
      consistency without recomputation.


    Examples
    --------
    >>> import numpy as np
    >>> from playNano.processing.video_processing import rolling_frame_align
    >>> stack = np.random.rand(10, 200, 200)  # 10 frames of 200x200 pixels
    >>> aligned_stack, metadata = rolling_frame_align(stack, window=3)
    >>> aligned_stack.shape
    (10, 210, 210)
    >>> metadata['shifts']
    array([[0, 0],
           [1, -1],
           ...])
    """
    stack = stack.astype(np.float32, copy=False)

    if stack.ndim != 3:
        raise ValueError(
            f"stack must be a 3D array (n_frames, H, W), got shape {stack.shape}"
        )
    if window < 1:
        raise ValueError("window must be >= 1")

    n_frames, H, W = stack.shape
    shifts = np.zeros((n_frames, 2), dtype=int)

    # Safety margin for rolling reference
    margin = max_shift if max_shift is not None else max(H, W) // 2
    canvas_H, canvas_W = H + 2 * margin, W + 2 * margin

    # Pre-filter stack once if needed
    if pre_filter_sigma is not None:
        filtered_stack = np.empty_like(stack)
        for i in range(n_frames):
            filtered_stack[i] = gaussian_filter(stack[i], sigma=pre_filter_sigma)
    else:
        filtered_stack = stack

    rolling_sum = np.zeros((canvas_H, canvas_W), dtype=np.float32)
    rolling_count = np.zeros((canvas_H, canvas_W), dtype=np.uint32)

    # Store only canvas indices for rolling reference
    aligned_refs = deque()  # each entry: (center_y, y_end, center_x, x_end)

    # Seed first frame at center
    center_y, center_x = margin, margin
    y_end, x_end = center_y + H, center_x + W

    rolling_sum[center_y:y_end, center_x:x_end] += filtered_stack[0]
    rolling_count[center_y:y_end, center_x:x_end] += 1

    # Store a full record (frame_idx + padded box + frame box)
    aligned_refs.append(
        (
            0,  # frame_idx
            center_y,
            y_end,  # padded box y
            center_x,
            x_end,  # padded box x
            0,
            H,  # frame box y
            0,
            W,  # frame box x
        )
    )

    for i in range(1, n_frames):
        # Compute rolling mean
        with np.errstate(divide="ignore", invalid="ignore"):
            denom_safe = np.where(rolling_count == 0, 1.0, rolling_count)
            ref = rolling_sum / denom_safe
            ref_mask_bool = rolling_count != 0

        frame_corr = filtered_stack[i]

        shift, _, _ = phase_cross_correlation(
            ref,
            frame_corr,
            upsample_factor=1,
            reference_mask=ref_mask_bool,
            moving_mask=np.ones_like(frame_corr, dtype=bool),
        )
        dy, dx = shift.astype(int)

        # Jump smoothing
        if max_jump is not None and i >= 2:
            prev_shift = shifts[i - 1]
            prev_prev_shift = shifts[i - 2]
            expected = prev_shift + (prev_shift - prev_prev_shift)
            if abs(dy - prev_shift[0]) > max_jump:
                dy = int(expected[0])
            if abs(dx - prev_shift[1]) > max_jump:
                dx = int(expected[1])

        # Clip by max_shift
        if max_shift is not None:
            dy = int(np.clip(dy, -max_shift, max_shift))
            dx = int(np.clip(dx, -max_shift, max_shift))

        shifts[i] = (dy, dx)

        # Compute canvas region
        y_start, x_start = center_y + dy, center_x + dx
        y_end, x_end = y_start + H, x_start + W

        # Clip to canvas
        y0c, x0c = max(0, y_start), max(0, x_start)
        y1c, x1c = min(canvas_H, y_end), min(canvas_W, x_end)

        # Corresponding patch in frame
        fy0, fx0 = y0c - y_start, x0c - x_start
        fy1, fx1 = fy0 + (y1c - y0c), fx0 + (x1c - x0c)

        if fy1 <= fy0 or fx1 <= fx0:
            continue  # skip empty patch

        patch = frame_corr[fy0:fy1, fx0:fx1]

        # Update rolling accumulators
        rolling_sum[y0c:y1c, x0c:x1c] += patch
        rolling_count[y0c:y1c, x0c:x1c] += 1
        # store frame index & frame coords
        aligned_refs.append((i, y0c, y1c, x0c, x1c, fy0, fy1, fx0, fx1))

        # Remove oldest frame from rolling sum/count if window exceeded
        if len(aligned_refs) > window:
            (
                frame_idx,
                old_y0,
                old_y1,
                old_x0,
                old_x1,
                old_fy0,
                old_fy1,
                old_fx0,
                old_fx1,
            ) = aligned_refs.popleft()
            old_patch = filtered_stack[frame_idx, old_fy0:old_fy1, old_fx0:old_fx1]
            rolling_sum[old_y0:old_y1, old_x0:old_x1] -= old_patch
            rolling_count[old_y0:old_y1, old_x0:old_x1] -= 1

    # Compute final canvas
    min_dy, max_dy = shifts[:, 0].min(), shifts[:, 0].max()
    min_dx, max_dx = shifts[:, 1].min(), shifts[:, 1].max()
    final_H, final_W = H + (max_dy - min_dy), W + (max_dx - min_dx)
    y_offset, x_offset = -min_dy, -min_dx

    aligned_stack = np.full((n_frames, final_H, final_W), np.nan, dtype=stack.dtype)
    border_mask = np.zeros((final_H, final_W), dtype=bool)

    for i in range(n_frames):
        dy, dx = shifts[i]
        y_start, x_start = y_offset + dy, x_offset + dx
        aligned_stack[i, y_start : y_start + H, x_start : x_start + W] = stack[i]
        border_mask[y_start : y_start + H, x_start : x_start + W] = True

    metadata = {
        "window": window,
        "method": "phase_cross_correlation",
        "mode": mode,
        "shifts": shifts,
        "original_shape": (H, W),
        "aligned_shape": aligned_stack.shape[1:],
        "border_mask": border_mask,
        "pre_filter_sigma": pre_filter_sigma,
        "max_shift": max_shift,
        "max_jump": max_jump,
    }

    # Apply cropping
    if mode == "crop":
        aligned_stack, crop_meta = intersection_crop(aligned_stack)
        metadata["crop"] = crop_meta
        if "bounds" in crop_meta:
            y_min, y_max, x_min, x_max = crop_meta["bounds"]
            metadata["border_mask"] = metadata["border_mask"][
                y_min : y_max + 1, x_min : x_max + 1
            ]
    elif mode == "crop_square":
        aligned_stack, crop_meta = crop_square(aligned_stack)
        metadata["crop"] = crop_meta
        H_sq, W_sq = aligned_stack.shape[1:]
        r_start, c_start = crop_meta["offset"]
        metadata["border_mask"] = metadata["border_mask"][
            r_start : r_start + H_sq, c_start : c_start + W_sq
        ]
    elif mode != "pad":
        raise ValueError(f"Unknown mode: {mode}")

    if debug:
        return (
            aligned_stack,
            metadata,
            {"shifts": shifts.copy(), "aligned_refs": list(aligned_refs)},
        )

    return aligned_stack, metadata


# -----------------------------------------------------------------------------#
# Cropping and padding helpers
# -----------------------------------------------------------------------------#


@versioned_filter("0.1.0")
def intersection_crop(stack: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Crop aligned stack to the largest common intersection region.

    Parameters
    ----------
    stack : ndarray of shape (n_frames, height, width)
        Input aligned stack with NaN padding.

    Returns
    -------
    cropped : ndarray
        Cropped stack containing only valid (non-NaN) pixels, or original stack if
        no valid pixels exist.
    meta : dict
        Metadata about cropping parameters.
    """
    valid_mask = np.all(np.isfinite(stack), axis=0)
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        # No valid pixels, return original stack
        meta = {
            "operation": "crop_intersection",
            "original_shape": stack.shape[1:],
            "new_shape": stack.shape[1:],
            "bounds": None,
            "note": "No finite pixels found, returned original stack",
        }
        return stack.copy(), meta

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    cropped = stack[:, y_min : y_max + 1, x_min : x_max + 1]

    meta = {
        "operation": "crop_intersection",
        "original_shape": stack.shape[1:],
        "new_shape": cropped.shape[1:],
        "bounds": (y_min, y_max, x_min, x_max),
    }
    return cropped, meta


@versioned_filter("0.1.0")
def crop_square(stack: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Crop aligned stack to the largest centered square region.

    Parameters
    ----------
    stack : ndarray of shape (n_frames, height, width)
        Input aligned stack.

    Returns
    -------
    cropped : ndarray
        Cropped stack with square height and width.
    meta : dict
        Metadata about cropping parameters.
    """
    cropped, inter_meta = intersection_crop(stack)
    H, W = cropped.shape[1:]
    size = min(H, W)
    # Center crop
    r_start = (H - size) // 2
    c_start = (W - size) // 2
    cropped_sq = cropped[:, r_start : r_start + size, c_start : c_start + size]

    meta = {
        "operation": "crop_square",
        "original_shape": stack.shape[1:],
        "intersection_shape": inter_meta["new_shape"],
        "new_shape": cropped_sq.shape[1:],
        "square_size": size,
        "offset": (r_start, c_start),
    }
    return cropped_sq, meta


@param_conditions(value=lambda p: p.get("mode") == "constant")
@versioned_filter("0.1.0")
def replace_nan(
    stack: np.ndarray,
    mode: Literal["zero", "mean", "median", "global_mean", "constant"] = "zero",
    value: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Replace NaN values in a 2D frame or 3D AFM image stack using various strategies.

    Primarily used in video pipelines after alignment, but also applicable to single
    frames.

    Parameters
    ----------
    stack : np.ndarray
        Input 3D array of shape (n_frames, height, width) or 2D frame (height, width)
        that may contain NaN values.
    mode : {"zero", "mean", "median", "global_mean", "constant"}, optional
        Replacement strategy. Default is "zero".
        - "zero" : Replace NaNs with 0.
        - "mean" : Replace NaNs with the mean of each frame.
        - "median" : Replace NaNs with the median of each frame.
        - "global_mean" : Replace NaNs with the mean of the entire stack.
        - "constant" : Replace NaNs with a user-specified constant `value`.
    value : float, optional
        Constant value to use when `mode="constant"`. Must be provided in that case.

    Returns
    -------
    filled : np.ndarray
        Stack of the same shape as `stack` with NaNs replaced according to `mode`.
    meta : dict
        Metadata about the NaN replacement operation (e.g., count, mode, constant used).


    Raises
    ------
    ValueError
        If `mode` is unknown or if `mode="constant"` and `value` is not provided.

    Notes
    -----
    - Frame-wise operations like "mean" and "median" compute statistics per frame
      independently.
    - Preserves the dtype of the input stack.
    """
    filled = stack.copy()
    nan_count = np.isnan(filled).sum()
    if mode == "zero":
        filled[np.isnan(filled)] = 0
    elif mode == "mean":
        for i in range(filled.shape[0]):
            frame = filled[i]
            mask = np.isnan(frame)
            if np.any(mask):
                frame_mean = np.nanmean(frame)
                frame[mask] = frame_mean
    elif mode == "median":
        for i in range(filled.shape[0]):
            frame = filled[i]
            mask = np.isnan(frame)
            if np.any(mask):
                frame_median = np.nanmedian(frame)
                frame[mask] = frame_median
    elif mode == "global_mean":
        global_mean = np.nanmean(filled)
        filled[np.isnan(filled)] = global_mean
    elif mode == "constant":
        if value is None:
            raise ValueError("Must provide 'value' for constant mode.")
        filled[np.isnan(filled)] = value
    else:
        raise ValueError(f"Unknown mode: {mode}")

    meta = {
        "nans_filled": int(nan_count),
        "mode": mode,
        "value_used": value if mode == "constant" else None,
    }

    return filled, meta


# -----------------------------------------------------------------------------#
# Temporal filters
# -----------------------------------------------------------------------------#


def temporal_median_filter(stack: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Apply median filter across the time dimension.

    Parameters
    ----------
    stack : ndarray of shape (n_frames, height, width)
        Input stack.
    window : int, optional
        Window size (number of frames). Default is 3.

    Returns
    -------
    filtered : ndarray of shape (n_frames, height, width)
        Stack after temporal median filtering.
    """
    raise NotImplementedError


def temporal_mean_filter(stack: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Apply mean filter across the time dimension.

    Parameters
    ----------
    stack : ndarray of shape (n_frames, height, width)
        Input stack.
    window : int, optional
        Window size (number of frames). Default is 3.

    Returns
    -------
    filtered : ndarray of shape (n_frames, height, width)
        Stack after temporal mean filtering.
    """
    raise NotImplementedError


# -----------------------------------------------------------------------------#
# Future placeholders (diagnostics, spatio-temporal denoising, etc.)
# -----------------------------------------------------------------------------#


def register_video_processing() -> dict[str, Callable]:
    """
    Return a dictionary of registered video processing filters.

    Keys are names of the operations, values are the functions themselves.
    These functions should take a 3D stack (n_frames, H, W) and return either
    an ndarray (filtered stack) or a tuple (stack, metadata).
    """
    return {
        "align_frames": align_frames,
        "rolling_frame_align": rolling_frame_align,
        "intersection_crop": intersection_crop,
        "crop_square": crop_square,
        "replace_nan": replace_nan,
    }
