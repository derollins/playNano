"""
Video export utilities for AFM image stacks.

This module provides functions for generating MP4 or AVI video files from AFM image
stacks, with optional timestamps and scale bars. Frames can be normalised
automatically or scaled using a fixed z-range.

The rendered frame pipeline is identical to :mod:`~playNano.io.gif_export`:
frames are colourised with a matplotlib colormap and annotated via
:func:`~playNano.utils.time_utils.draw_scale_and_timestamp` before being
written to a container format using :mod:`imageio`.

Dependencies
------------
- matplotlib
- numpy
- Pillow (PIL)
- imageio
- imageio[ffmpeg]
"""

import logging
from pathlib import Path

import cv2
import imageio
import numpy as np
from matplotlib import colormaps as cm

from playnano.utils.colormaps import DEFAULT_CMAP
from playnano.utils.io_utils import (
    compute_zscale_range,
    normalize_to_uint8,
    prepare_output_directory,
    sanitize_output_name,
)
from playnano.utils.time_utils import draw_scale_and_timestamp

logger = logging.getLogger(__name__)

# Formats supported by imageio / ffmpeg writer
_VALID_FORMATS = {"mp4", "avi"}


def create_video_with_scale_and_timestamp(
    image_stack: np.ndarray,
    pixel_sizes_nm: list,
    timestamps=None,
    scale_bar_length_nm: int = 100,
    output_path: str | Path = "output.mp4",
    fps: float = 10.0,
    cmap_name: str = DEFAULT_CMAP,
    zmin: float | str | None = None,
    zmax: float | str | None = None,
    draw_ts: bool = True,
    draw_scale: bool = True,
    codec: str | None = None,
) -> None:
    """
    Create a video file from an AFM image stack with optional overlays.

    Frames are normalised, colourised using a matplotlib colormap, and annotated
    with a scale bar and timestamps before being compiled into an MP4 or AVI file.

    Parameters
    ----------
    image_stack : np.ndarray
        3D array of shape (N, H, W) representing the AFM image stack.
    pixel_sizes_nm : list
        Per-frame pixel size in nanometres. Must have the same length as
        ``image_stack``.
    timestamps : list[float] or tuple[float], optional
        Timestamps in seconds for each frame. Frame indices are used as
        a fallback when ``None`` or invalid.
    scale_bar_length_nm : int
        Length of the scale bar in nanometres. Default is 100.
    output_path : str or Path
        Destination path including filename and extension (``".mp4"`` or
        ``".avi"``). Default is ``'output.mp4'``.
    fps : float
        Playback frame rate in frames per second. Default is 10.
    cmap_name : str
        Name of the matplotlib colormap. Default is ``'afmhot'``.
    zmin : float or str or None, optional
        Minimum z-value mapped to the low end of the colormap. The string
        literal ``"auto"`` sets this to the 1st percentile of the stack.
    zmax : float or str or None, optional
        Maximum z-value mapped to the high end of the colormap. The string
        literal ``"auto"`` sets this to the 99th percentile of the stack.
    draw_ts : bool
        Whether to draw timestamps on each frame. Default is ``True``.
    draw_scale : bool
        Whether to draw a scale bar on each frame. Default is ``True``.
    codec : str or None, optional
        Override the imageio/ffmpeg codec string (e.g. ``"libx264"``,
        ``"mpeg4"``). When ``None``, imageio chooses a sensible default for
        the requested container format.

    Raises
    ------
    ValueError
        If ``zmin`` equals ``zmax``.

    Returns
    -------
    None

    Notes
    -----
    - Frames are normalised globally when ``zmin``/``zmax`` are provided,
      otherwise per-frame.
    - RGB frames (uint8, shape H×W×3) are written via :mod:`imageio`.
    - Requires ``imageio[ffmpeg]``.
    """
    output_path = Path(output_path)
    cmap = cm.get_cmap(cmap_name)

    # Validate timestamps
    if (
        timestamps is not None
        and isinstance(timestamps, (list, tuple))
        and len(timestamps) == len(image_stack)
    ):
        has_valid_timestamps = True
    else:
        has_valid_timestamps = False
        logger.warning(
            "Invalid timestamps provided; frame indices will be used instead."
        )

    # Validate pixel sizes
    if not (
        pixel_sizes_nm is not None
        and isinstance(pixel_sizes_nm, list)
        and len(pixel_sizes_nm) == len(image_stack)
    ):
        draw_scale = False
        logger.warning("Invalid pixel_sizes_nm list; scale bar will be omitted.")

    # Global z-range (optional)
    if zmin is not None or zmax is not None:
        zmin_val, zmax_val = compute_zscale_range(image_stack, zmin, zmax)
    else:
        zmin_val, zmax_val = None, None

    # Build writer kwargs
    writer_kwargs: dict = {"fps": fps}
    if codec is not None:
        writer_kwargs["codec"] = codec

    with imageio.get_writer(
        str(output_path),
        fps=fps,
        codec=codec,
    ) as writer:
        for i, frame in enumerate(image_stack):
            # --- Normalise and colourise ---
            if zmin_val is not None and zmax_val is not None:
                if zmin_val == zmax_val:
                    frame_norm = np.zeros_like(frame, dtype=np.uint8)
                else:
                    clipped = np.clip(frame, zmin_val, zmax_val)
                    normalised = (clipped - zmin_val) / (zmax_val - zmin_val) * 255
                    normalised = np.nan_to_num(
                        normalised, nan=0.0, posinf=255.0, neginf=0.0
                    )
                    frame_norm = np.clip(normalised, 0, 255).astype(np.uint8)
            else:
                frame_norm = normalize_to_uint8(frame)

            frame_float = frame_norm / 255.0
            color_frame = (cmap(frame_float)[..., :3] * 255).astype(np.uint8)

            # --- Timestamp / scale overlay ---
            draw_pixel_size_nm = pixel_sizes_nm[i] if draw_scale else 1.0

            FONT_SCALE = 2  # tuned for 512px frames

            # Upscale BEFORE annotation
            if color_frame.shape[0] < 512:
                TARGET_HEIGHT = 512
            else:
                TARGET_HEIGHT = color_frame.shape[0]

            scale = TARGET_HEIGHT / color_frame.shape[0]

            color_frame = cv2.resize(
                color_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )

            frame_annotated = draw_scale_and_timestamp(
                color_frame,
                timestamp=float(timestamps[i]) if has_valid_timestamps else i,
                pixel_size_nm=draw_pixel_size_nm,
                scale=scale,
                bar_length_nm=scale_bar_length_nm,
                font_scale=FONT_SCALE,
                draw_ts=draw_ts,
                draw_scale=draw_scale,
                color=(255, 255, 255),
            )

            writer.append_data(frame_annotated)

    logger.info(f"Video saved to {output_path}")


def export_video(
    afm_stack,
    make_video: bool,
    output_folder: str | None,
    output_name: str | None,
    scale_bar_nm: int | None,
    fmt: str = "mp4",
    fps: float = 10.0,
    raw: bool = False,
    zmin: float | None = None,
    zmax: float | None = None,
    draw_ts: bool = True,
    draw_scale: bool = True,
    cmap_name: str = DEFAULT_CMAP,
    codec: str | None = None,
) -> None:
    """
    Export an AFM image stack as an annotated MP4 or AVI video.

    Parameters
    ----------
    afm_stack : AFMImageStack
        AFM stack object containing raw and/or processed data.
    make_video : bool
        Whether to generate the video. If ``False``, the function returns
        immediately.
    output_folder : str or None
        Directory to save the video. Defaults to ``"output"`` if ``None``.
    output_name : str or None
        Base name for the output file (without extension). Derived from the
        stack file name when ``None``.
    scale_bar_nm : int or None
        Scale bar length in nanometres. Defaults to 100 nm when ``None``.
    fmt : {"mp4", "avi"}
        Container format. Default is ``"mp4"``.
    fps : float
        Playback frame rate. Default is 10.
    raw : bool
        If ``True``, export the unprocessed raw snapshot; otherwise export
        the current (processed) data. Default is ``False``.
    zmin : float or None, optional
        Minimum z-value for colormap scaling. ``"auto"`` triggers the 1st
        percentile. ``None`` uses per-frame minimum.
    zmax : float or None, optional
        Maximum z-value for colormap scaling. ``"auto"`` triggers the 99th
        percentile. ``None`` uses per-frame maximum.
    draw_ts : bool
        Whether to draw timestamps on each frame. Default is ``True``.
    draw_scale : bool
        Whether to draw a scale bar on each frame. Default is ``True``.
    codec : str or None, optional
        Override the ffmpeg codec string. ``None`` lets imageio choose.

    Returns
    -------
    None

    Notes
    -----
    - Processed data is preferred over raw when available; ``_filtered`` is
      appended to the output stem in that case.
    - Timestamps and pixel size are read from ``afm_stack.frame_metadata``.
      When exporting raw data after an ``edit_stack`` step, the pre-edit
      metadata stored in
      ``afm_stack.state_backups['frame_metadata_before_edit']`` is used.
    - Requires ``imageio[ffmpeg]``: ``pip install imageio[ffmpeg]``.
    """
    if not make_video:
        return

    fmt = fmt.lower().lstrip(".")
    if fmt not in _VALID_FORMATS:
        raise ValueError(
            f"Unsupported video format '{fmt}'. Choose from {_VALID_FORMATS}."
        )

    out_dir = prepare_output_directory(output_folder, default="output")
    base = sanitize_output_name(output_name, Path(afm_stack.file_path).stem)

    # Choose data source
    if raw and "raw" in afm_stack.processed:
        stack_data = afm_stack.processed["raw"]
        meta_src = afm_stack.state_backups.get(
            "frame_metadata_before_edit", afm_stack.frame_metadata
        )
    else:
        if raw:
            logger.debug("Requested raw export on unprocessed data; using loaded data.")
        stack_data = afm_stack.data
        meta_src = afm_stack.frame_metadata
        filtered_exists = "raw" in afm_stack.processed and any(
            key != "raw" for key in afm_stack.processed
        )
        if filtered_exists:
            base = f"{base}_filtered"

    timestamps = [md["timestamp"] for md in meta_src]
    pixels_to_nm = [
        md.get("frame_pixel_size_nm", afm_stack.pixel_size_nm) for md in meta_src
    ]

    bar_nm = scale_bar_nm if scale_bar_nm is not None else 100
    video_path = out_dir / f"{base}.{fmt}"

    logger.debug(f"[export] Writing video → {video_path}")
    create_video_with_scale_and_timestamp(
        stack_data,
        pixels_to_nm,
        timestamps,
        output_path=video_path,
        fps=fps,
        scale_bar_length_nm=bar_nm,
        cmap_name=cmap_name,
        zmin=zmin,
        zmax=zmax,
        draw_ts=draw_ts,
        draw_scale=draw_scale,
        codec=codec,
    )
    logger.debug(f"[export] Video written to {video_path}")
