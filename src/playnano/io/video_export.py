"""
Export AFM image stacks as annotated videos (MP4, AVI, MOV or MKV).

Frames are coloured and annotated by :mod:`playnano.io.render_utils`, which the
GIF and image sequence exports share, and written with OpenCV
(``opencv-python-headless``). MP4, MOV and MKV files use the ``mp4v`` codec and
AVI files use ``XVID``.
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
    Write a video from an image stack, with optional time and scale annotations.

    Use :func:`export_video` to export an :class:`~playnano.afm_stack.AFMImageStack`.

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
    output_path : str or pathlib.Path, optional
        The video file to write. Its extension sets the codec: ``.avi`` uses
        ``XVID``, anything else ``mp4v``. Default is ``"output.mp4"``.
    fps : float, optional
        Frames per second. Default is 5.0.
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

    Raises
    ------
    ValueError
        If ``zmin`` is greater than ``zmax``, or a limit is not a number,
        ``"auto"`` or None.

    Notes
    -----
    Most codecs need even frame sizes, so if a rendered frame has an odd height
    or width, its last row or column is removed.
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
    Export a stack as an annotated video.

    The file is ``<output_folder>/<output_name>.<fmt>``.
    ``_filtered`` is added to the name when processed data is exported, that
    is when ``raw`` is False and the stack has processed layers other than
    ``"raw"``.

    Parameters
    ----------
    afm_stack : AFMImageStack
        The stack to export.
    make_video : bool
        If False, do nothing.
    output_folder : str or None
        Folder for the output; created if missing. If None, ``./output``.
    output_name : str or None
        Name for the output, without extension. Any folder part and known
        extension are removed. If None or empty, the name of the file the stack
        was loaded from is used.
    scale_bar_nm : int or None
        Length of the scale bar, in nm. 0 hides it; None uses 100.
    fmt : {"mp4", "avi", "mov", "mkv"}, optional
        Video format. Default is ``"mp4"``.
    fps : float, optional
        Frames per second. Default is 5.0.
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
