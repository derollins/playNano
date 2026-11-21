"""
Processing functions for whole AFM image stacks (time series of frames).

These functions take a 3D :class:`numpy.ndarray` of shape
(n_frames, height, width) and return the processed stack and a dictionary of
metadata, which the :class:`~playnano.processing.pipeline.ProcessingPipeline`
stores in the step's provenance. They cover:

- frame alignment, to correct drift (:func:`align_frames`,
  :func:`rolling_frame_align`);
- cropping the NaN borders left by alignment (:func:`intersection_crop`,
  :func:`crop_square`);
- filling NaN pixels (:func:`replace_nan`).

:func:`temporal_median_filter` and :func:`temporal_mean_filter` are planned
and not implemented yet.
"""

from collections import deque
from typing import Callable, Literal, Optional

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d, fftconvolve
from skimage.registration import phase_cross_correlation

from playnano.utils.param_utils import param_conditions
from playnano.utils.versioning import versioned_filter

# -----------------------------------------------------------------------------#
# Alignment
# -----------------------------------------------------------------------------#


@versioned_filter("0.2.0")
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
    Align every frame of a stack to one reference frame by whole-pixel shifts.

    The shift of each frame is the position of the peak of its cross-correlation
    with the reference frame. Each frame's mean is subtracted before
    correlating, so height offsets between frames do not affect the result.

    Parameters
    ----------
    stack : numpy.ndarray
        3D image stack of shape (n_frames, height, width).
    reference_frame : int, optional
        Index of the frame the others are aligned to. Default is 0.
    method : {"fft_cross_correlation", "full_cross_correlation"}, optional
        How the cross-correlation is computed. Both find the same shifts; the
        FFT method is much faster and uses less memory on large frames.
        Default is ``"fft_cross_correlation"``.
    mode : {"pad", "crop", "crop_square"}, optional
        What to do with the borders after shifting. Default is ``"pad"``.

        - ``"pad"``: keep every pixel of every frame on a larger canvas,
          filling the gaps with NaN.
        - ``"crop"``: crop to the region covered by all frames
          (:func:`intersection_crop`).
        - ``"crop_square"``: crop to the largest centred square in that region
          (:func:`crop_square`).
    debug : bool, optional
        If True, also return a third value with diagnostic output. Default is
        False.
    max_shift : int, optional
        Largest shift to search for, in pixels along each axis. Default is None
        (no limit).
    pre_filter_sigma : float, optional
        If given, frames are smoothed with a Gaussian filter of this standard
        deviation, in pixels, before correlating. The output frames are not
        smoothed. Default is None.
    max_jump : int, optional
        Largest change in shift, in pixels, allowed between consecutive frames.
        For frame 1 the change from frame 0 is clipped to this value; from
        frame 2 on, a shift that changes by more is replaced by a linear
        extrapolation of the previous two shifts. Default is None (no limit).

    Returns
    -------
    aligned_stack : numpy.ndarray
        The aligned stack, as float32. With ``mode="pad"`` it is larger than the
        input so every frame fits; with the crop modes it is smaller.
    metadata : dict
        Details of the alignment:

        - ``"reference_frame"`` (int): the reference frame used.
        - ``"method"`` (str): the ``method`` used.
        - ``"mode"`` (str): the ``mode`` used.
        - ``"shifts"`` (numpy.ndarray of int, shape (n_frames, 2)): the
          (dy, dx) shift applied to each frame, in pixels.
        - ``"original_shape"`` (tuple): (height, width) of the input frames.
        - ``"aligned_shape"`` (tuple): (height, width) of the padded canvas,
          before any cropping.
        - ``"border_mask"`` (numpy.ndarray of bool): True where at least one
          frame has data, with the same height and width as the output.
        - ``"pre_filter_sigma"``, ``"max_shift"``, ``"max_jump"``: the values
          used.
        - ``"crop"`` (dict): only with ``mode="crop"`` or ``"crop_square"``;
          the metadata from :func:`intersection_crop` or :func:`crop_square`.
    debug_outputs : dict
        Only returned if ``debug=True``: ``{"shifts": <copy of the shifts>}``.

    Raises
    ------
    ValueError
        If ``stack`` is not 3D, ``reference_frame`` is outside
        ``0 .. n_frames - 1``, ``method`` or ``mode`` is not recognised, or a
        frame has no finite pixels.

    Notes
    -----
    Shifts are whole pixels; there is no subpixel interpolation. NaN and
    infinite pixels are ignored when finding the shifts (they are filled with
    the frame mean for the correlation only) and are kept in the output.
    Version 0.2.0 subtracts each frame's mean for ``"full_cross_correlation"``
    too; before, that method found no drift on typical AFM data.

    Examples
    --------
    >>> import numpy as np
    >>> from playnano.processing.video_processing import align_frames
    >>> rng = np.random.default_rng(0)
    >>> frame = rng.random((64, 64))
    >>> moved = np.roll(frame, (2, -3), axis=(0, 1))  # 2 px down, 3 px left
    >>> aligned, meta = align_frames(np.stack([frame, moved]))
    >>> meta["shifts"]
    array([[ 0,  0],
           [-2,  3]])
    >>> aligned.shape
    (2, 66, 67)
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

    # NaN-free (and optionally smoothed) frames, used only to find the shifts
    corr_stack, _ = _prepare_correlation_frames(stack, pre_filter_sigma)

    # Preprocess reference frame
    ref0 = corr_stack[reference_frame] - np.mean(corr_stack[reference_frame])

    shifts = np.zeros((n_frames, 2), dtype=int)

    for i in range(n_frames):
        if i == reference_frame:
            continue

        frame = corr_stack[i]

        # Compute shift
        if method == "fft_cross_correlation":
            frame0 = frame.copy()
            frame0 -= np.mean(frame0)
            cc = fftconvolve(frame0[::-1, ::-1], ref0, mode="full")
            y_center, x_center = H - 1, W - 1
        elif method == "full_cross_correlation":
            # Subtract each frame's mean, as the FFT branch does. Otherwise any
            # height offset makes the zero-shift position the strongest match.
            frame0 = frame - np.mean(frame)
            cc = correlate2d(
                ref0, frame0, mode="full", boundary="fill", fillvalue=0
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
        metadata["border_mask"] = _crop_border_mask(border_mask, crop_meta)
    elif mode == "crop_square":
        aligned_stack, crop_meta = crop_square(aligned_stack)
        metadata["crop"] = crop_meta
        metadata["border_mask"] = _crop_border_mask(border_mask, crop_meta)
    elif mode != "pad":
        raise ValueError(f"Unknown mode: {mode}")

    if debug:
        return aligned_stack, metadata, {"shifts": shifts.copy()}

    return aligned_stack, metadata


@versioned_filter("0.1.1")
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
    Align each frame to the average of the frames aligned just before it.

    The first frame is the starting reference. Each later frame is aligned, by
    phase cross-correlation and whole-pixel shifts, to the average of the last
    ``window`` aligned frames, so slow changes in the sample are followed.

    Parameters
    ----------
    stack : numpy.ndarray
        3D image stack of shape (n_frames, height, width).
    window : int, optional
        Number of most recently aligned frames averaged to make the reference.
        Default is 5.
    mode : {"pad", "crop", "crop_square"}, optional
        What to do with the borders after shifting. Default is ``"pad"``.

        - ``"pad"``: keep every pixel of every frame on a larger canvas,
          filling the gaps with NaN.
        - ``"crop"``: crop to the region covered by all frames
          (:func:`intersection_crop`).
        - ``"crop_square"``: crop to the largest centred square in that region
          (:func:`crop_square`).
    debug : bool, optional
        If True, also return a third value with diagnostic output. Default is
        False.
    max_shift : int, optional
        Largest shift allowed, in pixels along each axis; larger shifts are
        clipped to it. Default is None (no limit).
    pre_filter_sigma : float, optional
        If given, frames are smoothed with a Gaussian filter of this standard
        deviation, in pixels, before correlating. The output frames are not
        smoothed. Default is None.
    max_jump : int, optional
        From frame 2 on, a shift that differs from the previous one by more than
        this many pixels is replaced by a linear extrapolation of the previous
        two shifts. Default is None (no limit).

    Returns
    -------
    aligned_stack : numpy.ndarray
        The aligned stack, as float32. With ``mode="pad"`` it is larger than the
        input so every frame fits; with the crop modes it is smaller.
    metadata : dict
        Details of the alignment:

        - ``"window"`` (int): the ``window`` used.
        - ``"method"`` (str): always ``"phase_cross_correlation"``.
        - ``"mode"`` (str): the ``mode`` used.
        - ``"shifts"`` (numpy.ndarray of int, shape (n_frames, 2)): the
          (dy, dx) shift applied to each frame, in pixels.
        - ``"original_shape"`` (tuple): (height, width) of the input frames.
        - ``"aligned_shape"`` (tuple): (height, width) of the padded canvas,
          before any cropping.
        - ``"border_mask"`` (numpy.ndarray of bool): True where at least one
          frame has data, with the same height and width as the output.
        - ``"pre_filter_sigma"``, ``"max_shift"``, ``"max_jump"``: the values
          used.
        - ``"crop"`` (dict): only with ``mode="crop"`` or ``"crop_square"``;
          the metadata from :func:`intersection_crop` or :func:`crop_square`.
    debug_outputs : dict
        Only returned if ``debug=True``:

        - ``"shifts"``: a copy of the shifts.
        - ``"aligned_refs"``: list of the frames in the final reference window,
          each as ``(frame_index, y0, y1, x0, x1, fy0, fy1, fx0, fx1)``: the
          region of the working canvas and the matching region of the frame.

    Raises
    ------
    ValueError
        If ``stack`` is not 3D, ``window`` is less than 1, ``mode`` is not
        recognised, or a frame has no finite pixels.

    Notes
    -----
    Shifts are whole pixels; there is no subpixel interpolation. NaN and
    infinite pixels are left out of the correlation and of the rolling
    reference, and are kept in the output. Version 0.1.1 crops ``border_mask``
    together with the frames in the crop modes.

    See Also
    --------
    skimage.registration.phase_cross_correlation : The correlation used.

    Examples
    --------
    >>> import numpy as np
    >>> from playnano.processing.video_processing import rolling_frame_align
    >>> rng = np.random.default_rng(0)
    >>> frame = rng.random((64, 64))
    >>> moved = np.roll(frame, (2, -3), axis=(0, 1))
    >>> aligned, meta = rolling_frame_align(np.stack([frame, moved]), window=3)
    >>> meta["shifts"]
    array([[ 0,  0],
           [-2,  3]])
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

    # NaN-free (and optionally smoothed) frames for correlation, plus a mask of
    # the finite input pixels. Only finite pixels enter the rolling reference.
    filtered_stack, valid = _prepare_correlation_frames(stack, pre_filter_sigma)

    rolling_sum = np.zeros((canvas_H, canvas_W), dtype=np.float32)
    rolling_count = np.zeros((canvas_H, canvas_W), dtype=np.uint32)

    # Store only canvas indices for rolling reference
    aligned_refs = deque()  # each entry: (center_y, y_end, center_x, x_end)

    # Seed first frame at center
    center_y, center_x = margin, margin
    y_end, x_end = center_y + H, center_x + W

    rolling_sum[center_y:y_end, center_x:x_end] += np.where(
        valid[0], filtered_stack[0], 0
    )
    rolling_count[center_y:y_end, center_x:x_end] += valid[0]

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
            moving_mask=valid[i],
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
        patch_valid = valid[i, fy0:fy1, fx0:fx1]

        # Update rolling accumulators (finite input pixels only)
        rolling_sum[y0c:y1c, x0c:x1c] += np.where(patch_valid, patch, 0)
        rolling_count[y0c:y1c, x0c:x1c] += patch_valid
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
            old_valid = valid[frame_idx, old_fy0:old_fy1, old_fx0:old_fx1]
            rolling_sum[old_y0:old_y1, old_x0:old_x1] -= np.where(
                old_valid, old_patch, 0
            )
            rolling_count[old_y0:old_y1, old_x0:old_x1] -= old_valid

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
        metadata["border_mask"] = _crop_border_mask(border_mask, crop_meta)
    elif mode == "crop_square":
        aligned_stack, crop_meta = crop_square(aligned_stack)
        metadata["crop"] = crop_meta
        metadata["border_mask"] = _crop_border_mask(border_mask, crop_meta)
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


def _normalize_pad(pad):
    """
    Normalize pad argument to (top, bottom, left, right).

    Accepts:
      - int: uniform pad on all sides
      - tuple/list of length 2: (vertical, horizontal)
      - tuple/list of length 4: (top, bottom, left, right)
    """
    if isinstance(pad, (int, np.integer)):
        return (pad, pad, pad, pad)
    if isinstance(pad, (tuple, list)):
        if len(pad) == 2:
            v, h = pad
            return (v, v, h, h)
        if len(pad) == 4:
            return tuple(pad)
    raise ValueError("pad must be int, (v, h), or (top, bottom, left, right)")


def _prepare_correlation_frames(
    stack: np.ndarray, pre_filter_sigma: Optional[float] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Make NaN-free frames for finding alignment shifts.

    A single NaN makes a whole cross-correlation NaN, so non-finite pixels are
    replaced by the mean of the frame's finite pixels. The optional Gaussian
    pre-filter runs after this, so it cannot spread NaN. The result is used
    only to find shifts; aligned output still uses the original data.

    Parameters
    ----------
    stack : ndarray, shape (n_frames, height, width)
        Input frames.
    pre_filter_sigma : float, optional
        Standard deviation of the Gaussian filter to apply to each frame.

    Returns
    -------
    frames : ndarray, shape (n_frames, height, width)
        Frames with no NaN or infinite values. This is ``stack`` itself when
        there is nothing to fill or filter.
    valid : ndarray of bool, shape (n_frames, height, width)
        True where the input pixel was finite.

    Raises
    ------
    ValueError
        If a frame has no finite pixels.
    """
    valid = np.isfinite(stack)
    all_valid = valid.all(axis=(1, 2))
    if all_valid.all() and pre_filter_sigma is None:
        return stack, valid

    frames = stack.copy()
    for i in range(stack.shape[0]):
        if not all_valid[i]:
            if not valid[i].any():
                raise ValueError(
                    f"Frame {i} has no finite pixels, so it cannot be aligned."
                )
            frames[i][~valid[i]] = frames[i][valid[i]].mean()
        if pre_filter_sigma is not None:
            frames[i] = gaussian_filter(frames[i], sigma=pre_filter_sigma)
    return frames, valid


def _crop_border_mask(border_mask: np.ndarray, crop_meta: dict) -> np.ndarray:
    """
    Crop an alignment ``border_mask`` to match a cropped stack.

    Parameters
    ----------
    border_mask : ndarray of bool, shape (height, width)
        Mask on the full aligned canvas.
    crop_meta : dict
        Metadata from :func:`intersection_crop` or :func:`crop_square`. Its
        ``requested_bounds`` are the exclusive-end bounds the stack was cut to.

    Returns
    -------
    ndarray of bool
        Mask with the same height and width as the cropped stack. Area added by
        padding beyond the canvas is False. If nothing was cropped, the mask is
        returned unchanged.
    """
    bounds = crop_meta.get("requested_bounds")
    if bounds is None:
        return border_mask
    y0, y1, x0, x1 = (int(b) for b in bounds)
    cropped, _ = _crop_with_pad(border_mask[np.newaxis], y0, y1, x0, x1, False)
    return cropped[0]


def _crop_with_pad(stack, y0, y1, x0, x1, pad_value=np.nan):
    """
    Crop stack[:, y0:y1, x0:x1] even if y0<0 or x1>W etc., padding with pad_value.

    y0, y1, x0, x1 are *exclusive-end* bounds in original coordinates.
    """
    n, H, W = stack.shape
    # Compute required padding
    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, y1 - H)
    pad_right = max(0, x1 - W)

    if pad_top or pad_bottom or pad_left or pad_right:
        stack_p = np.pad(
            stack,
            pad_width=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=pad_value,
        )
        # Shift requested box into padded coordinate system
        y0_p = y0 + pad_top
        y1_p = y1 + pad_top
        x0_p = x0 + pad_left
        x1_p = x1 + pad_left
        out = stack_p[:, y0_p:y1_p, x0_p:x1_p]
    else:
        out = stack[:, y0:y1, x0:x1]

    return out, (pad_top, pad_bottom, pad_left, pad_right)


@versioned_filter("0.2.0")
def intersection_crop(
    stack: np.ndarray, pad: int | tuple[int, ...] = 0
) -> tuple[np.ndarray, dict]:
    """
    Crop an aligned stack to the region where every frame has data.

    The crop is the smallest rectangle containing every pixel that is finite in
    all frames. Use it after alignment with ``mode="pad"`` to remove the NaN
    borders.

    Parameters
    ----------
    stack : numpy.ndarray
        3D image stack of shape (n_frames, height, width), usually the output
        of an alignment with NaN borders.
    pad : int or tuple of int, optional
        Extra pixels to add around the crop. Default is 0.

        - int: the same on every side.
        - ``(v, h)``: ``v`` at the top and bottom, ``h`` at the left and right.
        - ``(top, bottom, left, right)``: each side separately.

        Where the padded crop goes beyond the stack, the new pixels are NaN.

    Returns
    -------
    cropped : numpy.ndarray
        The cropped (and padded) stack.
    meta : dict
        Details of the crop. Bounds are ``(y0, y1, x0, x1)`` in the input's
        pixel coordinates, with ``y1`` and ``x1`` exclusive.

        - ``"operation"``: ``"crop_intersection"``.
        - ``"original_shape"``, ``"new_shape"``: (height, width) before and
          after.
        - ``"intersection_shape"``, ``"intersection_bounds"``: the region
          where all frames are finite.
        - ``"requested_bounds"``: the intersection plus ``pad``.
        - ``"applied_pad"``: (top, bottom, left, right) pixels of NaN added
          beyond the edges of the input.
        - ``"pad_param"``: ``pad`` as (top, bottom, left, right).

        If no pixel is finite in all frames, a copy of ``stack`` is returned
        unchanged, the bounds are None and ``meta["note"]`` says so.

    Raises
    ------
    ValueError
        If ``pad`` is not an int or a tuple of 2 or 4 ints.
    """
    valid_mask = np.all(np.isfinite(stack), axis=0)
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)

    H, W = stack.shape[1:]
    if not np.any(rows) or not np.any(cols):
        meta = {
            "operation": "crop_intersection",
            "original_shape": (H, W),
            "intersection_shape": (H, W),
            "new_shape": (H, W),
            "intersection_bounds": None,
            "requested_bounds": None,
            "applied_pad": (0, 0, 0, 0),
            "note": "No finite pixels found, returned original stack",
        }
        return stack.copy(), meta

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    y0, y1 = y_min, y_max + 1
    x0, x1 = x_min, x_max + 1

    pt, pb, pl, pr = _normalize_pad(pad)
    y0_req = y0 - pt
    y1_req = y1 + pb
    x0_req = x0 - pl
    x1_req = x1 + pr

    cropped, applied_pad = _crop_with_pad(stack, y0_req, y1_req, x0_req, x1_req)

    meta = {
        "operation": "crop_intersection",
        "original_shape": (H, W),
        "intersection_shape": (y1 - y0, x1 - x0),
        "new_shape": cropped.shape[1:],
        "intersection_bounds": (y0, y1, x0, x1),
        "requested_bounds": (y0_req, y1_req, x0_req, x1_req),
        "applied_pad": applied_pad,
        "pad_param": (pt, pb, pl, pr),
    }
    return cropped, meta


@versioned_filter("0.2.0")
def crop_square(
    stack: np.ndarray, pad: int | tuple[int, ...] = 0
) -> tuple[np.ndarray, dict]:
    """
    Crop an aligned stack to the largest centred square where every frame has data.

    The square is centred in the region found by :func:`intersection_crop`, and
    its side is the shorter side of that region.

    Parameters
    ----------
    stack : numpy.ndarray
        3D image stack of shape (n_frames, height, width), usually the output
        of an alignment with NaN borders.
    pad : int or tuple of int, optional
        Extra pixels to add around the crop. Default is 0.

        - int: the same on every side.
        - ``(v, h)``: ``v`` at the top and bottom, ``h`` at the left and right.
        - ``(top, bottom, left, right)``: each side separately.

        Where the padded crop goes beyond the stack, the new pixels are NaN.

    Returns
    -------
    cropped : numpy.ndarray
        The cropped (and padded) stack.
    meta : dict
        Details of the crop, with the same keys as :func:`intersection_crop`
        (``"operation"`` is ``"crop_square"``), and also:

        - ``"square_size"``: side of the square, in pixels.
        - ``"square_bounds"``: ``(y0, y1, x0, x1)`` of the square before
          padding.
        - ``"offset"``: (row, column) of the square's corner within the
          intersection region.

        If no pixel is finite in all frames, a copy of ``stack`` is returned
        unchanged, the bounds are None and ``meta["note"]`` says so.

    Raises
    ------
    ValueError
        If ``pad`` is not an int or a tuple of 2 or 4 ints.
    """
    H, W = stack.shape[1:]
    # Compute finite-pixel intersection mask (across frames)
    valid_mask = np.all(np.isfinite(stack), axis=0)
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        # No finite pixels: return original stack unchanged (original behavior)
        # offset is 0,0 (undefined in this case, but keep compatible keys)
        meta = {
            "operation": "crop_square",
            "original_shape": (H, W),
            "intersection_shape": (H, W),
            "new_shape": (H, W),
            "square_size": min(H, W),
            "offset": (0, 0),
            "intersection_bounds": None,
            "square_bounds": None,
            "requested_bounds": None,
            "applied_pad": (0, 0, 0, 0),
            "pad_param": _normalize_pad(pad),
            "note": "No finite pixels found; returned original stack",
        }
        return stack.copy(), meta

    # Tight intersection bounds in exclusive-end coordinates
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    y0_i, y1_i = y_min, y_max + 1
    x0_i, x1_i = x_min, x_max + 1
    H_i, W_i = (y1_i - y0_i), (x1_i - x0_i)

    # Largest centered square INSIDE the intersection
    size = min(H_i, W_i)
    r_start = (H_i - size) // 2  # offset within the intersection (for metadata compat)
    c_start = (W_i - size) // 2

    # Convert that to absolute image coordinates
    y0_sq = y0_i + r_start
    x0_sq = x0_i + c_start
    y1_sq = y0_sq + size
    x1_sq = x0_sq + size

    # Apply outward padding around the square
    pt, pb, pl, pr = _normalize_pad(pad)
    y0_req = y0_sq - pt
    y1_req = y1_sq + pb
    x0_req = x0_sq - pl
    x1_req = x1_sq + pr

    # Single crop directly from the original (no prior intersection cropping)
    cropped, applied_pad = _crop_with_pad(stack, y0_req, y1_req, x0_req, x1_req)

    meta = {
        "operation": "crop_square",
        "original_shape": (H, W),
        "intersection_shape": (H_i, W_i),
        "new_shape": cropped.shape[1:],
        "square_size": size,
        "offset": (r_start, c_start),
        "intersection_bounds": (y0_i, y1_i, x0_i, x1_i),
        "square_bounds": (y0_sq, y1_sq, x0_sq, x1_sq),
        "requested_bounds": (y0_req, y1_req, x0_req, x1_req),
        "applied_pad": applied_pad,  # actual pad used beyond image edges
        "pad_param": (pt, pb, pl, pr),  # requested pad
    }

    return cropped, meta


@param_conditions(value=lambda p: p.get("mode") == "constant")
@versioned_filter("0.1.0")
def replace_nan(
    stack: np.ndarray,
    mode: Literal["zero", "mean", "median", "global_mean", "constant"] = "zero",
    value: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Replace NaN pixels in a stack.

    Often used after alignment with ``mode="pad"``, before steps that do not
    accept NaN.

    Parameters
    ----------
    stack : numpy.ndarray
        3D image stack of shape (n_frames, height, width).
    mode : {"zero", "mean", "median", "global_mean", "constant"}, optional
        What to replace NaN with. Default is ``"zero"``.

        - ``"zero"``: 0.
        - ``"mean"``, ``"median"``: the mean or median of the finite pixels of
          the same frame.
        - ``"global_mean"``: the mean of the finite pixels of the whole stack.
        - ``"constant"``: ``value``.
    value : float, optional
        Value to use with ``mode="constant"``, where it is required.

    Returns
    -------
    filled : numpy.ndarray
        A copy of ``stack`` with NaN replaced, with the same shape and dtype.
    meta : dict
        ``{"nans_filled": <number of NaN pixels>, "mode": mode,
        "value_used": value}``; ``"value_used"`` is None unless
        ``mode="constant"``.

    Raises
    ------
    ValueError
        If ``mode`` is not recognised, or ``mode="constant"`` and ``value`` is
        None.
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
    Not implemented yet: median filter along the time axis.

    Planned for a future release; calling it raises ``NotImplementedError``. It
    is not registered as a processing step.

    Parameters
    ----------
    stack : numpy.ndarray
        3D image stack of shape (n_frames, height, width).
    window : int, optional
        Number of frames in the filter window. Default is 3.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError


def temporal_mean_filter(stack: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Not implemented yet: mean filter along the time axis.

    Planned for a future release; calling it raises ``NotImplementedError``. It
    is not registered as a processing step.

    Parameters
    ----------
    stack : numpy.ndarray
        3D image stack of shape (n_frames, height, width).
    window : int, optional
        Number of frames in the filter window. Default is 3.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError


# -----------------------------------------------------------------------------#
# Future placeholders (diagnostics, spatio-temporal denoising, etc.)
# -----------------------------------------------------------------------------#


def register_video_processing() -> dict[str, Callable]:
    """
    Return the built-in video processing functions.

    Each takes a 3D stack and returns either the processed stack or a tuple of
    (stack, metadata).

    Returns
    -------
    dict
        Maps each step name (str) to its function, for example
        ``"align_frames"`` to :func:`align_frames`.
    """
    return {
        "align_frames": align_frames,
        "rolling_frame_align": rolling_frame_align,
        "intersection_crop": intersection_crop,
        "crop_square": crop_square,
        "replace_nan": replace_nan,
    }
