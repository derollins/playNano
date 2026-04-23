"""
Video export utilities for AFM image stacks.

This module provides functions for generating  MP4, AVI, MOV, or MKV video files from
AFM image stacks, with optional timestamps and scale bars. Frames can be normalised
automatically or scaled using a fixed z-range.

The rendered frame pipeline is identical to :mod:`~playNano.io.gif_export`:
frames are colourised with a matplotlib colormap and annotated via
:mod:`~playNano.io.render_utils` before being written to a container format using
:mod:`cv2`.

Dependencies
------------
- matplotlib
- numpy
- OpenCV (cv2)
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from matplotlib import colormaps as cm

from playnano.io.render_utils import (
    DEFAULT_FONT_SCALE,
    render_frame,
    resolve_stack_data,
)
from playnano.utils.colormaps import DEFAULT_CMAP
from playnano.utils.io_utils import (
    compute_zscale_range,
    prepare_output_directory,
    sanitize_output_name,
)

logger = logging.getLogger(__name__)

_VALID_FORMATS = {"mp4", "avi", "mov", "mkv"}

_EXT_TO_FOURCC = {
    "avi": "XVID",
    "mov": "mp4v",
    "mkv": "mp4v",
}


def create_video_with_scale_and_timestamp(
    image_stack: np.ndarray,
    pixel_sizes_nm: list,
    timestamps=None,
    scale_bar_length_nm: int = 100,
    output_path: str | Path = "output.mp4",
    fps: float = 5.0,
    cmap_name: str = DEFAULT_CMAP,
    zmin: float | str | None = None,
    zmax: float | str | None = None,
    draw_ts: bool = True,
    draw_scale: bool = True,
    font_scale: float = DEFAULT_FONT_SCALE,
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
        Playback frame rate in frames per second. Default is 5.0.
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
    font_scale : float
        Base font scale for annotations at the reference height.
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
    - RGB frames (uint8, shape H×W×3) are written via :mod:`cv2`.
    - Requires ``opencv-python-headless``.
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

    if zmin is not None or zmax is not None:
        zmin_val, zmax_val = compute_zscale_range(image_stack, zmin, zmax)
    else:
        zmin_val, zmax_val = None, None

    extension = output_path.suffix.lstrip(".").lower()
    fourcc_str = _EXT_TO_FOURCC.get(extension, "mp4v")
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)

    writer = None

    for i, frame in enumerate(image_stack):
        # Determine timestamps
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

        pixel_size_nm = pixel_sizes_nm[i] if draw_scale else 1.0

        frame_annotated = render_frame(
            frame=frame,
            cmap=cmap,
            timestamp=timestamp,
            pixel_size_nm=pixel_size_nm,
            scale_bar_length_nm=scale_bar_length_nm,
            font_scale=font_scale,
            draw_ts=draw_ts,
            draw_scale=draw_scale,
            zmin_val=zmin_val,
            zmax_val=zmax_val,
        )

        # Ensure even dimensions — required by most video codecs
        h, w = frame_annotated.shape[:2]
        if h % 2 != 0 or w % 2 != 0:
            frame_annotated = frame_annotated[: h // 2 * 2, : w // 2 * 2]
            h, w = frame_annotated.shape[:2]

        if writer is None:
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        writer.write(cv2.cvtColor(frame_annotated, cv2.COLOR_RGB2BGR))

    if writer is not None:
        writer.release()

    logger.info(f"Video saved to {output_path}")


def export_video(
    afm_stack,
    make_video: bool,
    output_folder: str | None,
    output_name: str | None,
    scale_bar_nm: int | None,
    fmt: str = "mp4",
    fps: float = 5.0,
    raw: bool = False,
    zmin: float | None = None,
    zmax: float | None = None,
    draw_ts: bool = True,
    draw_scale: bool = True,
    cmap_name: str = DEFAULT_CMAP,
    font_scale: float = DEFAULT_FONT_SCALE,
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
    fmt : {"mp4", "avi", "mov", "mkv"}
        Container format. Default is ``"mp4"``.
    fps : float
        Playback frame rate. Default is 5.0.
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
    font_scale : float
        Base font scale for annotations at the reference height.
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
    - Requires ``opencv-python-headless``: ``pip install opencv-python-headless``.
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

    stack_data, meta_src, is_filtered = resolve_stack_data(afm_stack, raw)
    if is_filtered:
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
        font_scale=font_scale,
    )
    logger.debug(f"[export] Video written to {video_path}")
