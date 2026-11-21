"""
Frame rendering shared by the GIF, video and image sequence exports.

:func:`render_frame` scales, colours, resizes and annotates one frame, so all
three exports look the same. Module constants control the appearance:

- ``MIN_FRAME_HEIGHT`` (512): shorter frames are enlarged to this height;
  taller frames are not resized.
- ``REFERENCE_HEIGHT`` (512): the frame height at which ``font_scale`` is used
  as given; annotation size scales with the rendered frame height.
- ``ANNOTATION_COLOR`` ((255, 255, 255)): RGB colour of the annotations.
- ``DEFAULT_FONT_SCALE`` (2.0): default ``font_scale`` of the exports.
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
    Scale, colour, resize and annotate one frame for an animated export.

    Parameters
    ----------
    frame : numpy.ndarray
        2D image.
    cmap : matplotlib.colors.Colormap
        Colormap to apply.
    timestamp : float
        Time to draw, in seconds.
    pixel_size_nm : float
        Pixel size of the frame, in nm, used for the scale bar.
    scale_bar_length_nm : int
        Length of the scale bar, in nm; 0 hides it.
    font_scale : float
        Size of the annotation text for a frame ``REFERENCE_HEIGHT`` pixels
        high; scaled with the rendered frame height.
    draw_ts : bool
        Draw the timestamp.
    draw_scale : bool
        Draw the scale bar.
    zmin_val, zmax_val : float or None, optional
        Heights mapped to the bottom and top of the colormap; values outside
        are clipped and NaN is drawn in the bottom colour. If they are equal,
        the whole frame is drawn in the bottom colour. If either is None, the
        frame is scaled from its own minimum to its maximum, with NaN counted
        as 0. Default is None.

    Returns
    -------
    numpy.ndarray
        RGB image, uint8, of shape (height, width, 3). Frames shorter than
        ``MIN_FRAME_HEIGHT`` are enlarged to that height (keeping the aspect
        ratio) with cubic interpolation.
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
    Choose the data and frame metadata for an animated export.

    Parameters
    ----------
    afm_stack : AFMImageStack
        The stack to export.
    raw : bool
        If True and the stack has a ``"raw"`` snapshot, use it, with the frame
        metadata from before any frames were removed.

    Returns
    -------
    stack_data : numpy.ndarray
        The frames to render: the raw snapshot, or ``afm_stack.data``.
    meta_src : list of dict
        The matching per-frame metadata.
    is_filtered : bool
        True if processed data was chosen, so ``_filtered`` should be added to
        the output name.
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
