"""
Module for exporting a AFM image stack as a GIF.

This module provides utility functions to create animated GIFs from AFM image stacks,
with support for overlaying timestamps and scale bars. Frames can be scaled
using a fixed Z-range or normalized automatically.

Dependencies: matplotlib, numpy, PIL
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
):
    """
    Create a GIF from a stack of images with a scale bar and optional timestamps.

    This function normalises image frames, applies a colormap, adds a scale bar and
    timestamps (if provided), and compiles the frames into an animated GIF.

    Parameters
    ----------
    image_stack : np.ndarray
        3D NumPy array of shape (N, H, W), where N is the number of frames.
    pixel_size_nm : float
        Size of a pixel in nanometers.
    timestamps : list or tuple, optional
        List of timestamps corresponding to each frame (in seconds).
        If not provided, frame indices are used instead.
    scale_bar_length_nm : int, optional
        Length of the scale bar in nanometers. Default is 100.
    output_path : str, optional
        Path to save the output GIF. Default is "output.gif".
    duration : float, optional
        Duration of each frame in seconds. ???
    cmap_name : str, optional
        Name of the matplotlib colormap to use. Default is "afmhot".
    zmin : float, "auto", or None, optional
        Minimum Z-value to map to colormap 0.
        - If "auto", uses the 1st percentile of the data.
        - If None (default), uses the minimum value of the data.
        - If a float, uses the specified value.
    zmax : float, "auto", or None, optional
        Maximum Z-value to map to colormap 255.
        - If "auto", uses the 99th percentile of the data.
        - If None (default), uses the maximum value of the data.
        - If a float, uses the specified value.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If zmin == zmax or if timestamps are the wrong shape/type.

    Notes
    -----
    - Frames are normalized and colorized using the specified colormap.
    - A scale bar is drawn in the bottom-left corner of each frame.
    - Timestamps are displayed in the top-left corner if provided and valid.
    - The resulting GIF is saved to the specified output path.
    - If zmin and zmax are not provided, frames are normalized independently.
    - zmax must be higher than zmin.
    - Output is a visually scaled animated GIF with overlays for easy viewing.
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
        # Apply OpenCV overlay drawing
        frame_with_overlay = draw_scale_and_timestamp(
            color_frame.copy(),
            timestamp=timestamp,
            pixel_size_nm=pixel_size_nm,
            scale=1.0,  # no resizing in GIFs assumed
            font_scale=frame.shape[0] / 512,
            bar_length_nm=scale_bar_length_nm,
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
) -> None:
    """
    Optionally export a GIF of the AFM stack with scale bar and timestamps.

    Parameters
    ----------
    afm_stack : AFMImageStack
        The AFM stack object containing processed data.
    make_gif : bool
        Whether or not to generate a GIF.
    output_folder : str or None
        Destination folder for the GIF.
    output_name : str or None
        Optional base name override for the GIF file.
    scale_bar_nm : int or None
        Length of the scale bar in nanometers.
    zmin : float, "auto", or None, optional
        Minimum Z-value to map to colormap 0.
        - If "auto", uses the 1st percentile of the data.
        - If None (default), uses the minimum value of the data.
        - If a float, uses the specified value.
    zmax : float, "auto", or None, optional
        Maximum Z-value to map to colormap 255.
        - If "auto", uses the 99th percentile of the data.
        - If None (default), uses the maximum value of the data.
        - If a float, uses the specified value.

    Notes
    -----
    - The output file will have "_filtered" in the name if using processed data.
    - Timestamps and pixel size are automatically pulled from the stack metadata.
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
    )
    logger.debug(f"[export] GIF written to {gif_path}")
