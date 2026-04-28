"""
Image-sequence export utilities for AFM image stacks.

This module provides functions for saving each frame of an AFM image stack as
an individual PNG or JPEG file inside a dedicated output folder. Frames can be
normalised automatically or scaled with a fixed z-range, and are annotated with
optional timestamps and scale bars using the shared rendering pipeline in
:mod:`~playNano.io.render_utils`

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

_VALID_IMAGE_FORMATS = {"png", "jpg", "jpeg"}

# JPEG quality (0–95). Only used when saving JPEG files.
_JPEG_QUALITY = 90


def create_image_sequence(
    image_stack: np.ndarray,
    pixel_sizes_nm: list,
    timestamps=None,
    scale_bar_length_nm: int = 100,
    output_folder: str | Path = "output_frames",
    base_name: str = "frame",
    fmt: str = "png",
    cmap_name: str = DEFAULT_CMAP,
    zmin: float | str | None = None,
    zmax: float | str | None = None,
    draw_ts: bool = True,
    draw_scale: bool = True,
    font_scale: float = DEFAULT_FONT_SCALE,
) -> Path:
    """
    Save every frame of an AFM image stack as an individual image file.

    Frames are normalised, colourised with a matplotlib colormap, and annotated
    with a scale bar and timestamp before being written to *output_folder* as
    ``<base_name>_NNNN.<fmt>``.

    Parameters
    ----------
    image_stack : np.ndarray
        3D array of shape (N, H, W) representing the AFM image stack.
    pixel_sizes_nm : list
        Per-frame pixel size in nanometres. Must have the same length as
        ``image_stack``.
    timestamps : list[float] or tuple[float], optional
        Timestamps in seconds for each frame. Frame indices are used as a
        fallback when ``None`` or invalid.
    scale_bar_length_nm : int
        Length of the scale bar in nanometres. Default is 100.
    output_folder : str or Path
        Directory in which the frame images will be saved. Created
        automatically if it does not exist.
    base_name : str
        Stem used for each file: ``<base_name>_0000.png``, etc.
        Default is ``"frame"``.
    fmt : {"png", "jpg", "jpeg"}
        Image file format. Default is ``"png"``.
    cmap_name : str
        Name of the matplotlib colormap. Default is ``'afmhot'``.
    zmin : float or str or None, optional
        Minimum z-value mapped to the low end of the colormap. ``"auto"``
        uses the 1st percentile of the full stack.
    zmax : float or str or None, optional
        Maximum z-value mapped to the high end of the colormap. ``"auto"``
        uses the 99th percentile of the full stack.
    draw_ts : bool
        Whether to draw timestamps on each frame. Default is ``True``.
    draw_scale : bool
        Whether to draw a scale bar on each frame. Default is ``True``.
    font_scale : float
        Base font scale for annotations at the reference height.

    Returns
    -------
    Path
        The resolved output folder where images were written.

    Raises
    ------
    ValueError
        If ``fmt`` is not a supported image format, or if ``zmin == zmax``.

    Notes
    -----
    - Frames are normalised globally when ``zmin``/``zmax`` are provided,
      otherwise per-frame normalisation is applied.
    - PNG files are lossless; JPEG files are saved at quality
      :data:`_JPEG_QUALITY` (default 90).
    - Zero-padded four-digit frame indices are used so files sort correctly
      up to 9 999 frames; for larger stacks the padding grows automatically.
    """
    fmt = fmt.lower().lstrip(".")
    if fmt not in _VALID_IMAGE_FORMATS:
        raise ValueError(
            f"Unsupported image format '{fmt}'. Choose from {_VALID_IMAGE_FORMATS}."
        )
    save_fmt = "jpeg" if fmt in ("jpg", "jpeg") else "png"
    ext = "jpg" if save_fmt == "jpeg" else "png"

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    cmap = cm.get_cmap(cmap_name)
    n_frames = len(image_stack)
    pad = max(4, len(str(n_frames)))

    # Validate timestamps
    if (
        timestamps is not None
        and isinstance(timestamps, (list, tuple))
        and len(timestamps) == n_frames
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
        and len(pixel_sizes_nm) == n_frames
    ):
        draw_scale = False
        logger.warning("Invalid pixel_sizes_nm list; scale bar will be omitted.")

    if zmin is not None or zmax is not None:
        zmin_val, zmax_val = compute_zscale_range(image_stack, zmin, zmax)
    else:
        zmin_val, zmax_val = None, None

    save_kwargs: dict = {}
    if save_fmt == "jpeg":
        save_kwargs["quality"] = _JPEG_QUALITY

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

        file_name = f"{base_name}_{str(i).zfill(pad)}.{ext}"
        Image.fromarray(frame_annotated).save(
            output_folder / file_name, format=save_fmt, **save_kwargs
        )

    logger.info(f"{n_frames} frames written to {output_folder}")
    return output_folder


def export_image_sequence(
    afm_stack,
    make_sequence: bool,
    output_folder: str | None,
    output_name: str | None,
    scale_bar_nm: int | None,
    fmt: str = "png",
    raw: bool = False,
    zmin: float | None = None,
    zmax: float | None = None,
    draw_ts: bool = True,
    draw_scale: bool = True,
    cmap_name: str = DEFAULT_CMAP,
    font_scale: float = DEFAULT_FONT_SCALE,
) -> None:
    """
    Export an AFM image stack as a folder of annotated PNG or JPEG images.

    Each frame is saved as a separate file named
    ``<stem>_NNNN.<fmt>`` inside a subfolder ``<output_folder>/<stem>/``.

    Parameters
    ----------
    afm_stack : AFMImageStack
        AFM stack object containing raw and/or processed data.
    make_sequence : bool
        Whether to generate the image sequence. Returns immediately when
        ``False``.
    output_folder : str or None
        Root directory for output. Defaults to ``"output"`` when ``None``.
        A subfolder named after the stack stem is created inside it.
    output_name : str or None
        Base name for the subfolder and frame files. Derived from the stack
        file name when ``None``.
    scale_bar_nm : int or None
        Scale bar length in nanometres. Defaults to 100 nm when ``None``.
    fmt : {"png", "jpg", "jpeg"}
        Image file format. Default is ``"png"``.
    raw : bool
        If ``True``, export the unprocessed raw snapshot. Default is
        ``False``.
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
    """
    if not make_sequence:
        return

    out_root = prepare_output_directory(output_folder, default="output")
    base = sanitize_output_name(output_name, Path(afm_stack.file_path).stem)

    stack_data, meta_src, is_filtered = resolve_stack_data(afm_stack, raw)
    if is_filtered:
        base = f"{base}_filtered"

    timestamps = [md["timestamp"] for md in meta_src]
    pixels_to_nm = [
        md.get("frame_pixel_size_nm", afm_stack.pixel_size_nm) for md in meta_src
    ]

    bar_nm = scale_bar_nm if scale_bar_nm is not None else 100
    frames_dir = out_root / base

    logger.debug(f"[export] Writing image sequence → {frames_dir}")
    create_image_sequence(
        stack_data,
        pixels_to_nm,
        timestamps,
        output_folder=frames_dir,
        base_name=base,
        fmt=fmt,
        scale_bar_length_nm=bar_nm,
        cmap_name=cmap_name,
        zmin=zmin,
        zmax=zmax,
        draw_ts=draw_ts,
        draw_scale=draw_scale,
        font_scale=font_scale,
    )
    logger.debug(f"[export] Image sequence written to {frames_dir}")
