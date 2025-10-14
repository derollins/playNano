"""
GIF export utilities for AFM image stacks.

This module provides functions for generating animated GIFs from AFM image stacks,
with optional timestamps and scale bars. Frames can be normalized automatically or
scaled using a fixed z-range.

Dependencies
------------
- matplotlib
- numpy
- Pillow (PIL)
"""

import logging
from pathlib import Path

import numpy as np
from matplotlib import colormaps as cm
from PIL import Image

from playNano.utils.io_utils import (
    compute_zscale_range,
    normalize_to_uint8,
    prepare_output_directory,
    sanitize_output_name,
)
from playNano.utils.time_utils import draw_scale_and_timestamp

logger = logging.getLogger(__name__)


def create_gif_with_scale_and_timestamp(
    image_stack,
    pixel_size_nm,
    timestamps=None,
    scale_bar_length_nm=100,
    output_path="output",
    duration=0.5,
    cmap_name="afmhot",
    zmin: float | str | None = None,
    zmax: float | str | None = None,
    draw_ts: bool = True,
    draw_scale: bool = True,
):
    """
    Create an animated GIF from an AFM image stack with optional overlays.

    Frames are normalized, colorized using a matplotlib colormap, and annotated
    with a scale bar and timestamps before being compiled into a GIF.

    Parameters
    ----------
    image_stack : np.ndarray
        3D array of shape (N, H, W) representing the AFM image stack.

    pixel_size_nm : float
        Size of each pixel in nanometers.

    timestamps : list[float] or tuple[float], optional
        Timestamps for each frame in seconds. If ``None`` or invalid,
        frame indices are used.

    scale_bar_length_nm : int
        Length of the scale bar in nanometers. Default is 100.

    output_path : str
        Path where the GIF will be saved. Default is 'output'.

    duration : float
        Duration of each frame in seconds. Default is 0.5.

    cmap_name : str
        Name of the matplotlib colormap to apply. Default is 'afmhot'.

    zmin : float or str or None, optional
        Minimum z-value mapped to colormap low end.
        The string literal ``"auto"`` uses the 1st percentile.


    zmax : float or str or None, optional
        Maximum z-value mapped to colormap high end.
        The string literal ``"auto"`` uses the 99th percentile.


    draw_ts : bool
        Whether to draw timestamps. Default is True.

    draw_scale : bool
        Whether to draw a scale bar. Default is True.

    Raises
    ------
    ValueError
        If ``zmin`` equals ``zmax`` or ``timestamps`` have incorrect shape.

    Returns
    -------
    None

    Notes
    -----
    - Timestamps and scale bars are drawn in white.
    - Frames are normalized globally if ``zmin`` and ``zmax`` are provided;
      otherwise, per-frame.

    """
    frames = []
    cmap = cm.get_cmap(cmap_name)

    # Check if timestamps are usable
    if (
        timestamps is not None
        and isinstance(timestamps, (list, tuple))
        and len(timestamps) == len(image_stack)
    ):
        has_valid_timestamps = True
    else:
        has_valid_timestamps = False
        logger.warning(
            "Invalid timestamps provided, will use frame indices as timestamps."
        )

    if zmin is not None or zmax is not None:
        zmin_val, zmax_val = compute_zscale_range(image_stack, zmin, zmax)
    else:
        zmin_val, zmax_val = None, None

    for i, frame in enumerate(image_stack):
        # Normalize and colorize
        if zmin_val is not None and zmax_val is not None:
            # Clip to [zmin, zmax], normalize to [0, 255]
            if zmin_val == zmax_val:
                # Flat image: avoid division by zero, render as black
                frame_norm = np.zeros_like(frame, dtype=np.uint8)
            else:
                clipped = np.clip(frame, zmin_val, zmax_val)
                frame_norm = (
                    (clipped - zmin_val) / (zmax_val - zmin_val) * 255
                ).astype(np.uint8)
        else:
            frame_norm = normalize_to_uint8(frame)

        frame_norm_float = frame_norm / 255.0  # rescale to [0, 1] for cmap input
        color_frame = (cmap(frame_norm_float)[..., :3] * 255).astype(np.uint8)

        # Determine timestamp
        if has_valid_timestamps:
            try:
                timestamp = float(timestamps[i])
            except (TypeError, ValueError, IndexError):
                timestamp = i
        else:
            logger.warning(
                f"Invalid timestamps provided, using frame index {i} as timestamp."
            )
            timestamp = i
        # Add annotations
        frame_with_overlay = draw_scale_and_timestamp(
            color_frame,
            timestamp=timestamp,
            pixel_size_nm=pixel_size_nm,
            scale=1.0,
            bar_length_nm=scale_bar_length_nm,
            font_scale=frame.shape[0] / 256,
            draw_ts=draw_ts,
            draw_scale=draw_scale,
            color=(255, 255, 255),
        )

        # Convert back to PIL
        img = Image.fromarray(frame_with_overlay)
        frames.append(img)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration * 500),
        loop=0,
    )
    logger.info(f"GIF saved to {output_path}")


def export_gif(
    afm_stack,
    make_gif: bool,
    output_folder: str | None,
    output_name: str | None,
    scale_bar_nm: int | None,
    raw: bool = False,
    zmin: float | None = None,
    zmax: float | None = None,
    draw_ts: bool = True,
    draw_scale: bool = True,
) -> None:
    """
    Export an AFM image stack as an annotated GIF.

    Parameters
    ----------
    afm_stack : AFMImageStack
        AFM stack object containing raw and/or processed data.

    make_gif : bool
        Whether to generate the GIF. If ``False``, the function exits immediately.

    output_folder : str or None
        Directory to save the GIF. Defaults to ``"output"`` if ``None``.

    output_name : str or None
        Base name for the GIF file. If ``None``, derived from the stack file name.

    scale_bar_nm : int or None
        Length of the scale bar in nanometers. Defaults to 100 nm.

    raw : bool
        If ``True``, export raw (unprocessed) data; otherwise export processed data
        if available. Default is False.

    zmin : float or None, optional
        Minimum z-value mapped to colormap low end. The string literal ``"auto"``
        can also be used to automatically set the 1st percentile. ``None`` uses
        the minimum value of the data.

    zmax : float or None, optional
        Maximum z-value mapped to colormap high end. The string literal ``"auto"``
        can also be used to automatically set the 99th percentile. ``None`` uses
        the maximum value of the data.

    draw_ts : bool
        Whether to draw timestamps on each frame. Default is True.

    draw_scale : bool
        Whether to draw a scale bar on each frame. Default is True.

    Returns
    -------
    None

    Notes
    -----
    - Uses processed data if available; otherwise falls back to raw data.
    - Timestamps and pixel size are read from ``afm_stack`` metadata.
    - Output file name includes ``"_filtered"`` if processed data is exported.
    """
    if not make_gif:
        return

    out_dir = prepare_output_directory(output_folder, default="output")
    base = sanitize_output_name(output_name, Path(afm_stack.file_path).stem)

    # Determine whether to use raw or processed data
    # (allows saving of unfiltered from play mode)
    if raw is False:
        stack_data = afm_stack.data
        raw_exists = "raw" in afm_stack.processed
        filtered_exists = raw_exists and any(
            key != "raw" for key in afm_stack.processed.keys()
        )
        if filtered_exists:
            base = f"{base}_filtered"
    elif raw is True:
        if "raw" in afm_stack.processed:
            stack_data = afm_stack.processed["raw"]
        else:
            logger.debug("Requested raw export on unprocessed data; using loaded data.")
            stack_data = afm_stack.data

    gif_path = out_dir / f"{base}.gif"

    timestamps = [md["timestamp"] for md in afm_stack.frame_metadata]

    pixel_to_nm = afm_stack.pixel_size_nm

    # default scale bar
    bar_nm = scale_bar_nm if scale_bar_nm is not None else 100

    logger.debug(f"[export] Writing GIF → {gif_path}")
    create_gif_with_scale_and_timestamp(
        stack_data,
        pixel_to_nm,
        timestamps,
        output_path=gif_path,
        scale_bar_length_nm=bar_nm,
        cmap_name="afmhot",
        zmin=zmin,
        zmax=zmax,
        draw_ts=draw_ts,
        draw_scale=draw_scale,
    )
    logger.debug(f"[export] GIF written to {gif_path}")
