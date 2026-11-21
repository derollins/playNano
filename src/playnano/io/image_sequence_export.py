"""
Export AFM image stacks as a folder of annotated PNG or JPEG images, one per frame.

Frames are coloured and annotated by :mod:`playnano.io.render_utils`, which the
GIF and video exports share, and written with Pillow.
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
    Write each frame of an image stack as an image file, with optional annotations.

    Files are named ``<base_name>_<n>.<ext>``, where ``n`` is the frame index from
    0, padded with zeros to at least 4 digits so they sort in order. Use
    :func:`export_image_sequence` to export an
    :class:`~playnano.afm_stack.AFMImageStack`.

    Parameters
    ----------
    image_stack : numpy.ndarray
        3D image stack of shape (n_frames, height, width).
    pixel_sizes_nm : list of float
        Pixel size of each frame, in nm. If it is not a list with one value per
        frame, no scale bar is drawn.
    timestamps : list or tuple of float, optional
        Time of each frame, in seconds. If it is not a list or tuple with one
        value per frame, frame numbers are drawn instead.
    scale_bar_length_nm : int, optional
        Length of the scale bar, in nm; 0 hides it. Default is 100.
    output_folder : str or pathlib.Path, optional
        Folder for the images; created if missing. Default is
        ``"output_frames"``.
    base_name : str, optional
        Start of each file name. Default is ``"frame"``.
    fmt : {"png", "jpg", "jpeg"}, optional
        Image format. PNG is lossless; JPEG files are saved at quality 90 with
        the extension ``.jpg``. Default is ``"png"``.
    cmap_name : str, optional
        Name of a registered matplotlib colormap. Default is ``"afm_brown"``.
    zmin, zmax : float or str or None, optional
        Colour scale limits, as for the ``export_*`` functions: both None scales
        each frame separately; otherwise the limits apply to every frame, with
        ``"auto"`` for the 1st or 99th percentile and None for the stack's
        minimum or maximum. Default is None.
    draw_ts : bool, optional
        Draw each frame's time on the frame. Default is True.
    draw_scale : bool, optional
        Draw a scale bar. Default is True.
    font_scale : float, optional
        Size of the annotation text for a frame 512 pixels high. Default is 2.0.

    Returns
    -------
    pathlib.Path
        The folder the images were written to.

    Raises
    ------
    ValueError
        If ``fmt`` is not supported, ``zmin`` is greater than ``zmax``, or a
        limit is not a number, ``"auto"`` or None.
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
    Export a stack as a folder of annotated PNG or JPEG images, one per frame.

    The images are written to ``<output_folder>/<output_name>/`` as
    ``<output_name>_0000.png``, ``<output_name>_0001.png`` and so on.
    ``_filtered`` is added to the name when processed data is exported, that
    is when ``raw`` is False and the stack has processed layers other than
    ``"raw"``.

    Parameters
    ----------
    afm_stack : AFMImageStack
        The stack to export.
    make_sequence : bool
        If False, do nothing.
    output_folder : str or None
        Folder for the output; created if missing. If None, ``./output``.
    output_name : str or None
        Name for the output, without extension. Any folder part and known
        extension are removed. If None or empty, the name of the file the stack
        was loaded from is used.
    scale_bar_nm : int or None
        Length of the scale bar, in nm. 0 hides it; None uses 100.
    fmt : {"png", "jpg", "jpeg"}, optional
        Image format. Default is ``"png"``.
    raw : bool, optional
        If True, export the unprocessed snapshot ``stack.processed["raw"]``,
        with the frame times from before any frames were removed. Otherwise
        export ``stack.data``. Default is False.
    zmin : float or str or None, optional
        Height mapped to the bottom of the colormap, in the units of the data
        (usually nm), or ``"auto"`` for the 1st percentile of the stack.
        Default is None.
    zmax : float or str or None, optional
        Height mapped to the top of the colormap, or ``"auto"`` for the 99th
        percentile of the stack. Default is None.

        If ``zmin`` and ``zmax`` are both None, each frame is scaled separately
        from its own minimum to its maximum. If either is set, both apply to
        every frame, and a None limit becomes the minimum or maximum of the
        whole stack. Values outside the range are clipped.
    draw_ts : bool, optional
        Draw each frame's time, in seconds, on the frame. Default is True.
    draw_scale : bool, optional
        Draw a scale bar. Default is True.
    cmap_name : str, optional
        Name of a registered matplotlib colormap. Default is ``"afm_brown"``.
    font_scale : float, optional
        Size of the annotation text for a frame 512 pixels high; scaled with the
        rendered frame height. Default is 2.0.

    Raises
    ------
    ValueError
        If ``fmt`` is not supported, or ``zmin`` is greater than ``zmax``.
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
