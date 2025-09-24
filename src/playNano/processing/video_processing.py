"""
Video processing functions for AFM time-series (stacks of frames).

This module provides functions that operate on 3D numpy arrays
(time-series of 2D AFM frames). These include:

- Frame alignment to compensate for drift
- Cropping and padding utilities
- Temporal (time-domain) filters
- Future extensions such as spatio-temporal denoising

All functions follow a NumPy-style API: input stacks are 3D arrays
with shape (n_frames, height, width). Outputs are either processed
stacks, metadata dictionaries, or both.
"""

from typing import Any, Sequence

import numpy as np
from scipy.signal import correlate2d, fftconvolve

from playNano.processing.versioning import versioned_filter

# -----------------------------------------------------------------------------#
# Alignment
# -----------------------------------------------------------------------------#


@versioned_filter("0.1.0")
def align_frames(
    stack,
    reference_frame: int = 0,
    method: str = "fft_cross_correlation",
    mode: str = "pad",
    debug: bool = False,
):
    """
    Align a stack of AFM frames to a reference frame using integer-pixel shifts.

    Parameters
    ----------
    stack : np.ndarray
        3D array of shape (n_frames, height, width) containing the input AFM image
        stack.
    reference_frame : int, optional
        Index of the frame to use as the alignment reference (default is 0).
    method : {"fft_cross_correlation", "full_cross_correlation"}, optional
        Alignment method to use (default "fft_cross_correlation").
    mode : {"pad", "crop", "crop_square"}, optional
        How to handle frame borders after shifting (default "pad").
    debug : bool, optional
        If True, return additional diagnostic outputs such as residual maps.

    Returns
    -------
    aligned_stack : np.ndarray
        3D array of shape (n_frames, new_height, new_width) containing the aligned
        frames.
        Depending on `mode`, this may be padded with NaN or cropped.
    metadata : dict
        Dictionary with alignment information:

        - ``"reference_frame"`` : int
        - ``"method"`` : str
        - ``"shifts"`` : np.ndarray of shape (n_frames, 2)
        - ``"original_shape"`` : tuple of int
        - ``"aligned_shape"`` : tuple of int
        - ``"border_mask"`` : np.ndarray, bool

    Other Parameters
    ----------------
    debug_outputs : dict, optional
        Only returned if ``debug=True``. May include residual maps, overlays,
        or diagnostic plots for quality assessment.

    Notes
    -----
    - All shifts are computed on the integer grid (no subpixel refinement).
    - Padding uses NaN to preserve numeric meaning.
    - ``border_mask`` can be used downstream to exclude padded regions.
    """
    n_frames, H, W = stack.shape
    ref = stack[reference_frame]

    # Store shifts: (dy, dx) for each frame
    shifts = np.zeros((n_frames, 2), dtype=int)

    # First pass: compute shifts
    for i in range(n_frames):
        if i == reference_frame:
            shifts[i] = (0, 0)
            continue

        frame = stack[i]

        if method == "fft_cross_correlation":
            ref0 = ref - np.mean(ref)
            frame0 = frame - np.mean(frame)
            cc = fftconvolve(frame0[::-1, ::-1], ref0, mode="full")
            y_max, x_max = np.unravel_index(np.argmax(cc), cc.shape)
            dy = y_max - (H - 1)
            dx = x_max - (W - 1)
            shifts[i] = (dy, dx)

        elif method == "full_cross_correlation":
            # Brute-force integer cross-correlation (very slow for large frames)
            cc = correlate2d(ref, frame, mode="full", boundary="fill", fillvalue=0)
            y_max, x_max = np.unravel_index(np.argmax(cc), cc.shape)
            dy = y_max - (H - 1)
            dx = x_max - (W - 1)
            shifts[i] = (dy, dx)

        else:
            raise ValueError(f"Unknown alignment method: {method}")

    # Second pass: apply shifts
    if mode == "pad":
        max_dy_pos = shifts[:, 0].max()
        max_dy_neg = shifts[:, 0].min()
        max_dx_pos = shifts[:, 1].max()
        max_dx_neg = shifts[:, 1].min()
        new_H = H + max_dy_pos - max_dy_neg
        new_W = W + max_dx_pos - max_dx_neg

        aligned_stack = np.full((n_frames, new_H, new_W), np.nan, dtype=stack.dtype)
        border_mask = np.zeros((new_H, new_W), dtype=bool)

        for i in range(n_frames):
            dy, dx = shifts[i]
            y_start = dy - max_dy_neg
            x_start = dx - max_dx_neg
            aligned_stack[i, y_start : y_start + H, x_start : x_start + W] = stack[i]
            border_mask[y_start : y_start + H, x_start : x_start + W] = True

    elif mode in {"crop", "crop_square"}:
        # TODO: implement cropping modes
        raise NotImplementedError(f"Mode '{mode}' not yet implemented.")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    metadata = {
        "reference_frame": reference_frame,
        "method": method,
        "shifts": shifts,
        "original_shape": (H, W),
        "aligned_shape": aligned_stack.shape[1:],
        "border_mask": border_mask,
    }

    debug_outputs = {}
    if debug:
        debug_outputs["aligned_stack"] = aligned_stack.copy()
        debug_outputs["shifts"] = shifts.copy()
        return aligned_stack, metadata, debug_outputs

    return aligned_stack, metadata


# -----------------------------------------------------------------------------#
# Cropping and padding helpers
# -----------------------------------------------------------------------------#


def intersection_crop(stack: np.ndarray) -> np.ndarray:
    """
    Crop aligned stack to the largest common intersection region.

    Parameters
    ----------
    stack : ndarray of shape (n_frames, height, width)
        Input aligned stack with NaN padding.

    Returns
    -------
    cropped : ndarray
        Cropped stack containing only valid (non-NaN) regions.
    """
    raise NotImplementedError


def crop_square(stack: np.ndarray) -> np.ndarray:
    """
    Crop aligned stack to a square region.

    Parameters
    ----------
    stack : ndarray of shape (n_frames, height, width)
        Input aligned stack.

    Returns
    -------
    cropped : ndarray
        Cropped stack with square height and width.
    """
    raise NotImplementedError


def replace_nan(stack: np.ndarray, mode: str = "zero") -> np.ndarray:
    """
    Replace NaN padding values for visualization or export.

    Parameters
    ----------
    stack : ndarray of shape (n_frames, height, width)
        Input stack with possible NaN values.
    mode : {"zero", "mean"}, optional
        Replacement strategy:
        - "zero": replace NaNs with 0
        - "mean": replace NaNs with frame mean

    Returns
    -------
    filled : ndarray
        Stack with NaNs replaced according to `mode`.
    """
    raise NotImplementedError


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
# Drop and Select Frames- drop or select only certain frames from a stack
# Choose by index or by range
# -----------------------------------------------------------------------------#


@versioned_filter("0.1.0")
def select_frames(
    stack: np.ndarray,
    indices: Sequence[int],
):
    """
    Select a subset of frames from a 3D AFM image stack.

    Parameters
    ----------
    stack : np.ndarray
        3D array of shape (n_frames, height, width) containing the input AFM image
        stack.
    indices : sequence of int
        Indices of the frames to retain. Must be within the range ``[0, n_frames)``.

    Returns
    -------
    subset : np.ndarray
        3D array of shape (len(indices), height, width) containing only the selected
        frames.
    metadata : dict
        Dictionary with selection information:

        - ``"original_n_frames"`` : int
          Number of frames in the original stack.
        - ``"selected_indices"`` : list of int
          Indices of frames that were kept.
        - ``"new_n_frames"`` : int
          Number of frames in the output stack.

    Notes
    -----
    - Use this function to drop incomplete or corrupted frames prior to alignment
      or other processing steps.
    - The frame order is preserved according to `indices`.

    Examples
    --------
    >>> subset, meta = select_frames(stack, indices=[0, 2, 4])
    >>> subset.shape
    (3, 512, 512)
    >>> meta["selected_indices"]
    [0, 2, 4]
    """
    n_frames = stack.shape[0]

    # Validate
    indices = np.asarray(indices, dtype=int)
    if np.any(indices < 0) or np.any(indices >= n_frames):
        raise IndexError(f"indices must be in range [0, {n_frames})")

    indices = np.unique(indices)

    subset = stack[indices]

    metadata = {
        "original_n_frames": n_frames,
        "selected_indices": indices.tolist(),
        "new_n_frames": subset.shape[0],
    }

    return subset, metadata


@versioned_filter("0.1.0")
def drop_frames(
    stack: np.ndarray,
    indices: Sequence[int],
):
    """
    Drop a subset of frames from a 3D AFM image stack.

    Parameters
    ----------
    stack : np.ndarray
        3D array of shape (n_frames, height, width) containing the input AFM image
        stack.
    indices : sequence of int
        Indices of the frames to remove. Must be within the range ``[0, n_frames)``.

    Returns
    -------
    subset : np.ndarray
        3D array of shape (n_frames - len(indices), height, width) containing the
        remaining frames.
    metadata : dict
        Dictionary with selection information:

        - ``"original_n_frames"`` : int
          Number of frames in the original stack.
        - ``"dropped_indices"`` : list of int
          Indices of frames that were removed.
        - ``"new_n_frames"`` : int
          Number of frames in the output stack.

    Notes
    -----
    - Use this function to exclude incomplete or corrupted frames prior to alignment
      or other processing steps.
    - The frame order is preserved for the retained frames.

    Examples
    --------
    >>> subset, meta = drop_frames(stack, indices=[1, 3, 5])
    >>> subset.shape
    (n_frames - 3, 512, 512)
    >>> meta["dropped_indices"]
    [1, 3, 5]
    """
    n_frames = stack.shape[0]

    # Validate
    indices = np.asarray(indices, dtype=int)
    if np.any(indices < 0) or np.any(indices >= n_frames):
        raise IndexError(f"indices must be in range [0, {n_frames})")

    indices = np.unique(indices)

    mask = np.ones(n_frames, dtype=bool)
    mask[indices] = False
    subset = stack[mask]

    metadata = {
        "original_n_frames": n_frames,
        "dropped_indices": indices.tolist(),
        "new_n_frames": subset.shape[0],
    }

    return subset, metadata


@versioned_filter("0.1.0")
def select_frame_range(
    stack: np.ndarray, start: int, end: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Select a contiguous range of frames from a 3D AFM image stack.

    Parameters
    ----------
    stack : np.ndarray
        3D array of shape (n_frames, height, width) containing the input stack.
    start : int
        First frame index to include (inclusive).
    end : int
        Last frame index to include (exclusive), following Python slicing rules.

    Returns
    -------
    subset : np.ndarray
        3D array of shape (end - start, height, width) containing the selected frames.
    metadata : dict
        Dictionary with selection information:

        - ``original_n_frames`` : int
        - ``kept_indices`` : list of int
        - ``dropped_indices`` : list of int
        - ``new_n_frames`` : int

    Raises
    ------
    ValueError
        If ``start`` or ``end`` are out of bounds, or if ``start >= end``.
    """
    n_frames = stack.shape[0]

    if not (0 <= start < n_frames):
        raise ValueError(f"Start index {start} out of bounds [0, {n_frames}).")
    if not (0 < end <= n_frames):
        raise ValueError(f"End index {end} out of bounds (0, {n_frames}].")
    if start >= end:
        raise ValueError(f"Start index {start} must be less than end index {end}.")

    kept = list(range(start, end))
    dropped = [i for i in range(n_frames) if i not in kept]
    subset = stack[start:end]

    metadata = {
        "original_n_frames": n_frames,
        "kept_indices": kept,
        "dropped_indices": dropped,
        "new_n_frames": subset.shape[0],
    }

    return subset, metadata


# -----------------------------------------------------------------------------#
# Future placeholders (diagnostics, spatio-temporal denoising, etc.)
# -----------------------------------------------------------------------------#
