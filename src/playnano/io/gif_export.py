"""
Export AFM image stacks as annotated, animated GIFs.

Frames are coloured and annotated by :mod:`playnano.io.render_utils`, which the
video and image sequence exports share, and written with Pillow.
"""

import logging
from pathlib import Path

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


def create_gif_with_scale_and_timestamp(
    image_stack,
    pixel_sizes_nm,
    timestamps=None,
    scale_bar_length_nm=100,
    output_path="output",
    fps: float = 5.0,
    cmap_name=DEFAULT_CMAP,
    zmin: float | str | None = None,
    zmax: float | str | None = None,
    draw_ts: bool = True,
    draw_scale: bool = True,
    font_scale: float = DEFAULT_FONT_SCALE,
):
    """
    Write an animated GIF from an image stack, with optional time and scale annotations.

    Use :func:`export_gif` to export an :class:`~playnano.afm_stack.AFMImageStack`.

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
        The GIF file to write. Default is ``"output"``.
    fps : float, optional
        Frames per second. The frame delay is rounded down to a whole number of
        milliseconds; with 0 or less, each frame shows for 0.5 s. Default is 5.0.
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
    """
    frames = []
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
        frames.append(Image.fromarray(frame_annotated))

    duration_ms = int((1.0 / fps) * 1000) if fps > 0 else 500

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    logger.info(f"GIF saved to {output_path}")


def export_gif(
    afm_stack,
    make_gif: bool,
    output_folder: str | None,
    output_name: str | None,
    scale_bar_nm: int | None,
    cmap_name: str = DEFAULT_CMAP,
    raw: bool = False,
    fps: float = 5.0,
    zmin: float | None = None,
    zmax: float | None = None,
    draw_ts: bool = True,
    draw_scale: bool = True,
    font_scale: float = DEFAULT_FONT_SCALE,
) -> None:
    """
    Export a stack as an annotated, animated GIF.

    The file is ``<output_folder>/<output_name>.gif``.
    ``_filtered`` is added to the name when processed data is exported, that
    is when ``raw`` is False and the stack has processed layers other than
    ``"raw"``.

    Parameters
    ----------
    afm_stack : AFMImageStack
        The stack to export.
    make_gif : bool
        If False, do nothing.
    output_folder : str or None
        Folder for the output; created if missing. If None, ``./output``.
    output_name : str or None
        Name for the output, without extension. Any folder part and known
        extension are removed. If None or empty, the name of the file the stack
        was loaded from is used.
    scale_bar_nm : int or None
        Length of the scale bar, in nm. 0 hides it; None uses 100.
    cmap_name : str, optional
        Name of a registered matplotlib colormap. Default is ``"afm_brown"``.
    raw : bool, optional
        If True, export the unprocessed snapshot ``stack.processed["raw"]``,
        with the frame times from before any frames were removed. Otherwise
        export ``stack.data``. Default is False.
    fps : float, optional
        Frames per second. Default is 5.0.
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
    font_scale : float, optional
        Size of the annotation text for a frame 512 pixels high; scaled with the
        rendered frame height. Default is 2.0.

    Raises
    ------
    ValueError
        If ``zmin`` is greater than ``zmax``.
    """
    if not make_gif:
        return

    out_dir = prepare_output_directory(output_folder, default="output")
    base = sanitize_output_name(output_name, Path(afm_stack.file_path).stem)

    stack_data, meta_src, is_filtered = resolve_stack_data(afm_stack, raw)
    if is_filtered:
        base = f"{base}_filtered"

    timestamps = [md["timestamp"] for md in meta_src]
    pixels_to_nm = [
        md.get("frame_pixel_size_nm", afm_stack.pixel_size_nm) for md in meta_src
    ]

    gif_path = out_dir / f"{base}.gif"
    bar_nm = scale_bar_nm if scale_bar_nm is not None else 100

    logger.debug(f"[export] Writing GIF → {gif_path}")

    create_gif_with_scale_and_timestamp(
        stack_data,
        pixels_to_nm,
        timestamps,
        output_path=gif_path,
        scale_bar_length_nm=bar_nm,
        cmap_name=cmap_name,
        zmin=zmin,
        zmax=zmax,
        draw_ts=draw_ts,
        draw_scale=draw_scale,
        font_scale=font_scale,
        fps=fps,
    )
    logger.debug(f"[export] GIF written to {gif_path}")
