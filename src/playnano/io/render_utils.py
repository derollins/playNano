"""
Shared frame-rendering utilities for all rendered AFM exports.

This module provides the single source of truth for how frames are
normalised, colourised, resized, and annotated across GIF, video,
and image-sequence exports. Changing appearance here affects all
three export formats consistently.

Constants
---------
MIN_FRAME_HEIGHT : int
    Frames shorter than this are upscaled; taller frames are unchanged.
REFERENCE_HEIGHT : int
    The height at which ``font_scale=1.0`` produces the default annotation
    size. Derived font scale is proportional to the actual frame height.
ANNOTATION_COLOR : tuple
    Default RGB colour for all annotations.
DEFAULT_FONT_SCALE : float
    Base font scale passed to draw_scale_and_timestamp at REFERENCE_HEIGHT.
"""

import logging

import cv2
import numpy as np

from playnano.utils.io_utils import normalize_to_uint8
from playnano.utils.time_utils import draw_scale_and_timestamp

logger = logging.getLogger(__name__)

# --- Appearance constants ---
MIN_FRAME_HEIGHT: int = 512
REFERENCE_HEIGHT: int = 512
ANNOTATION_COLOR: tuple = (255, 255, 255)
DEFAULT_FONT_SCALE: float = 2.0


def render_frame(
    frame: np.ndarray,
    cmap,
    timestamp: float,
    pixel_size_nm: float,
    scale_bar_length_nm: int,
    font_scale: float,
    draw_ts: bool,
    draw_scale: bool,
    zmin_val: float | None = None,
    zmax_val: float | None = None,
) -> np.ndarray:
    """
    Normalise, colourise, resize, and annotate a single AFM frame.

    This is the shared rendering pipeline used by all rendered exports
    (GIF, video, image sequence). All appearance decisions are made here.

    Parameters
    ----------
    frame : np.ndarray
        2D float array (H, W) representing a single AFM frame.
    cmap : matplotlib.colors.Colormap
        Colormap to apply.
    timestamp : float
        Timestamp in seconds for the annotation overlay.
    pixel_size_nm : float
        Pixel size in nanometres, used to compute the scale bar length.
    scale_bar_length_nm : int
        Desired scale bar length in nanometres.
    font_scale : float
        Base font scale. The actual font size is derived proportionally
        from the frame height relative to ``REFERENCE_HEIGHT``, so
        annotations appear the same visual size regardless of resolution.
    draw_ts : bool
        Whether to draw the timestamp overlay.
    draw_scale : bool
        Whether to draw the scale bar overlay.
    zmin_val : float or None
        Pre-computed global minimum for normalisation. If ``None``,
        per-frame normalisation is used.
    zmax_val : float or None
        Pre-computed global maximum for normalisation. If ``None``,
        per-frame normalisation is used.

    Returns
    -------
    np.ndarray
        Annotated RGB frame as a uint8 array of shape (H, W, 3).
        Height is at least ``MIN_FRAME_HEIGHT``.
    """
    # --- Normalise ---
    if zmin_val is not None and zmax_val is not None:
        if zmin_val == zmax_val:
            frame_norm = np.zeros_like(frame, dtype=np.uint8)
        else:
            clipped = np.clip(frame, zmin_val, zmax_val)
            normalised = (clipped - zmin_val) / (zmax_val - zmin_val) * 255
            normalised = np.nan_to_num(normalised, nan=0.0, posinf=255.0, neginf=0.0)
            frame_norm = np.clip(normalised, 0, 255).astype(np.uint8)
    else:
        frame_norm = normalize_to_uint8(frame)

    # --- Colourise ---
    color_frame = (cmap(frame_norm / 255.0)[..., :3] * 255).astype(np.uint8)

    # --- Resize (upscale only) ---
    target_height = max(MIN_FRAME_HEIGHT, color_frame.shape[0])
    resize_scale = target_height / color_frame.shape[0]  # only used for cv2.resize
    annotation_scale = target_height / REFERENCE_HEIGHT  # always relative to reference

    if resize_scale != 1.0:
        color_frame = cv2.resize(
            color_frame,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_CUBIC,
        )

    # --- Derived font scale: proportional to frame height ---
    derived_font_scale = font_scale * annotation_scale

    # --- Annotate ---
    frame_annotated = draw_scale_and_timestamp(
        color_frame,
        timestamp=timestamp,
        pixel_size_nm=pixel_size_nm,
        resize_scale=resize_scale,  # physical scale for bar length
        annotation_scale=annotation_scale,  # layout scale for everything else
        bar_length_nm=scale_bar_length_nm,
        font_scale=derived_font_scale,
        draw_ts=draw_ts,
        draw_scale=draw_scale,
        color=ANNOTATION_COLOR,
    )

    return frame_annotated


def resolve_stack_data(afm_stack, raw: bool) -> tuple:
    """
    Select the correct data source and metadata for a rendered export.

    Centralises the raw-vs-processed selection logic shared by all
    ``export_*`` functions.

    Parameters
    ----------
    afm_stack : AFMImageStack
        The stack to export.
    raw : bool
        If ``True``, use the unprocessed raw snapshot when available.

    Returns
    -------
    stack_data : np.ndarray
        The frame data array to render.
    meta_src : list[dict]
        Per-frame metadata (timestamps, pixel sizes).
    is_filtered : bool
        ``True`` when processed (filtered) data is being exported.
        Callers should append ``"_filtered"`` to the output name.
    """
    if raw and "raw" in afm_stack.processed:
        stack_data = afm_stack.processed["raw"]
        meta_src = afm_stack.state_backups.get(
            "frame_metadata_before_edit", afm_stack.frame_metadata
        )
        is_filtered = False
    else:
        if raw:
            logger.debug("Requested raw export on unprocessed data; using loaded data.")
        stack_data = afm_stack.data
        meta_src = afm_stack.frame_metadata
        is_filtered = "raw" in afm_stack.processed and any(
            key != "raw" for key in afm_stack.processed
        )
    return stack_data, meta_src, is_filtered
