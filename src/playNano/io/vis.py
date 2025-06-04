"""Module for opening a window to interactively view and export AFM image stacks."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from matplotlib import colormaps as cm

from playNano.cli import sanitize_output_name
from playNano.io.export import save_h5_bundle, save_npz_bundle, save_ome_tiff_stack
from playNano.stack.afm_stack import AFMImageStack
from playNano.utils import draw_scale_and_timestamp, normalize_to_uint8, pad_to_square

logger = logging.getLogger(__name__)


def _make_save_dir(output_dir: Optional[str]) -> Path:
    """
    Create (if necessary) and return a Path to the desired output directory.

    Parameters
    ----------
    output_dir : str or None
        User-provided path for exports. If None or empty, defaults to "./output".

    Returns
    -------
    Path
        The (existing or newly created) directory path.

    Raises
    ------
    Exception
        If the directory cannot be created.
    """
    out = Path(output_dir or "output")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _compute_out_stem(output_name: Optional[str], default_stem: str) -> str:
    """
    Sanitize and return a base filename (no extension) for exports.

    If `output_name` is None or empty, `default_stem` is used. Otherwise,
    `sanitize_output_name` is applied to `output_name`.

    Parameters
    ----------
    output_name : str or None
        The user-provided basename (no extension).
    default_stem : str
        Fallback filename stem (usually the input file’s stem).

    Returns
    -------
    str
        A valid, sanitized base name to use for export files.
    """
    return sanitize_output_name(output_name, default_stem)


def _export_tiff(
    afm_data: AFMImageStack,
    stack: np.ndarray,
    output_dir: Path,
    base_stem: str,
    filtered: bool,
) -> None:
    """
    Export `stack` as an OME-TIFF.

    Parameters
    ----------
    afm_data : AFMImageStack
        The AFM stack (for metadata).
    stack : np.ndarray
        Array of shape (n_frames, H, W) to write.
    output_dir : Path
        Directory in which to save the TIFF.
    base_stem : str
        Base file name (no extension).
    filtered : bool
        If True, append "_filtered" to `base_stem`. Otherwise no suffix.

    Returns
    -------
    None
    """
    suffix = "_filtered" if filtered else ""
    tif_path = output_dir / f"{base_stem}{suffix}.ome.tif"
    timestamps = [md["timestamp"] for md in afm_data.frame_metadata]
    save_ome_tiff_stack(
        path=tif_path,
        stack=stack,
        pixel_size_nm=afm_data.pixel_size_nm,
        timestamps=timestamps,
        channel=afm_data.channel,
    )
    logger.info(f"[play] Exported OME-TIFF → {tif_path}")


def _export_npz(
    afm_data: AFMImageStack,
    stack: np.ndarray,
    output_dir: Path,
    base_stem: str,
    filtered: bool,
) -> None:
    """
    Export `stack` as a compressed NPZ bundle.

    Parameters
    ----------
    afm_data : AFMImageStack
        The AFM stack (for metadata).
    stack : np.ndarray
        Array of shape (n_frames, H, W) to write.
    output_dir : Path
        Directory in which to save the NPZ.
    base_stem : str
        Base file name (no extension).
    filtered : bool
        If True, append "_filtered" to `base_stem`. Otherwise no suffix.

    Returns
    -------
    None
    """
    suffix = "_filtered" if filtered else ""
    npz_path = output_dir / f"{base_stem}{suffix}"
    timestamps = [md["timestamp"] for md in afm_data.frame_metadata]
    save_npz_bundle(
        path=npz_path,
        stack=stack,
        pixel_size_nm=afm_data.pixel_size_nm,
        timestamps=timestamps,
        channel=afm_data.channel,
    )
    logger.info(f"[play] Exported NPZ → {npz_path}.npz")


def _export_h5(
    afm_data: AFMImageStack,
    stack: np.ndarray,
    output_dir: Path,
    base_stem: str,
    filtered: bool,
) -> None:
    """
    Export `stack` as an HDF5 bundle, embedding full frame_metadata.

    Parameters
    ----------
    afm_data : AFMImageStack
        The AFM stack (for metadata & frame_metadata).
    stack : np.ndarray
        Array of shape (n_frames, H, W) to write.
    output_dir : Path
        Directory in which to save the HDF5.
    base_stem : str
        Base file name (no extension).
    filtered : bool
        If True, append "_filtered" to `base_stem`. Otherwise no suffix.

    Returns
    -------
    None
    """
    suffix = "_filtered" if filtered else ""
    h5_path = output_dir / f"{base_stem}{suffix}"
    timestamps = [md["timestamp"] for md in afm_data.frame_metadata]
    save_h5_bundle(
        path=h5_path,
        stack=stack,
        pixel_size_nm=afm_data.pixel_size_nm,
        timestamps=timestamps,
        frame_metadata=afm_data.frame_metadata,
        channel=afm_data.channel,
    )
    logger.info(f"[play] Exported HDF5 → {h5_path}.h5")


def _export_gif(
    afm_data: AFMImageStack,
    stack: np.ndarray,
    output_dir: Path,
    base_stem: str,
    filtered: bool,
    scale_bar_nm: float,
) -> None:
    """
    Export `stack` as an animated GIF with scale bar and timestamp overlay.

    Parameters
    ----------
    afm_data : AFMImageStack
        The AFM stack (for metadata & pixel_size).
    stack : np.ndarray
        Array of shape (n_frames, H, W) to write.
    output_dir : Path
        Directory in which to save the GIF.
    base_stem : str
        Base file name (no extension).
    filtered : bool
        If True, append "_filtered" to `base_stem`. Otherwise no suffix.
    fps : float
        Frames per second to embed (for correct timestamp).

    Returns
    -------
    None
    """
    from playNano.io.gif_export import create_gif_with_scale_and_timestamp

    suffix = "_filtered" if filtered else ""
    gif_path = output_dir / f"{base_stem}{suffix}.gif"
    timestamps = [md["timestamp"] for md in afm_data.frame_metadata]
    create_gif_with_scale_and_timestamp(
        stack,
        afm_data.pixel_size_nm,
        timestamps,
        output_path=gif_path,
        scale_bar_length_nm=scale_bar_nm,
    )
    logger.info(f"[play] Exported GIF → {gif_path}")


def play_stack_cv(
    afm_data: AFMImageStack,
    fps: float,
    window_name: str = "AFM Stack Viewer",
    output_dir: str = "./",
    output_name: str = "",
    filter_steps: Optional[list[str]] = None,
    scale_bar_nm: int = 100,
) -> None:
    """
    Pop up an OpenCV window and play a 3D AFM stack as video.

    This function allows you to view a 3D AFM image stack as a video,
    with on-the-fly filtering & export.

    Press
      - 'f' to apply the filters in `filter_steps` (default to
      'topostats_flatten' if none provided),
      - SPACE to toggle between raw and filtered,
      - 't' to export current view as OME-TIFF,
      - 'n' to export current view as NPZ,
      - 'h' to export current view as HDF5,
      - 'g' to export current view as animated GIF,
      - ESC or 'q' to quit the window.

    Parameters
    ----------
    afm_data : AFMImageStack
        An AFMImageStack instance containing `.data`, `.processed`,
         `.pixel_size_nm`, and `.frame_metadata`.
    fps : float
        Frames per second (e.g. line_rate / image_height).
    window_name : str, optional
        Name of the OpenCV window. Defaults to "AFM Stack Viewer".
    output_dir : str, optional
        Directory path (string) in which to save any exported files. Defaults to "./".
    output_name : str, optional
        Base name (no extension) for any exported files.
        If empty, uses `afm_data.file_path.stem`.
    filter_steps : list of str, optional
        A list of filter names (in order) to apply when 'f' is pressed.
        If None or empty, defaults to ["topostats_flatten"].

    Returns
    -------
    None
        This function runs an interactive loop;
        it returns only when the user quits (ESC/'q').
    """
    delay_ms = int(50 / fps) if fps > 0 else 50

    # Determine window size based on image shape
    frame_shape = afm_data.image_shape
    square_size = frame_shape[0]  # assume square images

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, square_size, square_size)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, 1)

    idx = 0
    num_frames = afm_data.data.shape[0]
    flat_stack: Optional[np.ndarray] = None
    showing_flat = False

    # Always keep a snapshot of the raw frames
    raw_data = afm_data.processed.get("raw", afm_data.data)

    if not filter_steps:
        filter_steps = []

    cmap = cm.get_cmap("afmhot")

    while True:
        # 1) Try to get current window size; if closed, break
        try:
            win_w, win_h = (
                int(cv2.getWindowImageRect(window_name)[2]),
                int(cv2.getWindowImageRect(window_name)[3]),
            )
        except cv2.error:
            break  # window closed

        # 2) Choose raw vs. filtered frame and pad to square
        current_frame = flat_stack[idx] if showing_flat else raw_data[idx]
        square_img = pad_to_square(current_frame)

        # If window minimized, skip rendering but still catch keys
        if win_w <= 0 or win_h <= 0:
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord("q")):
                break
            continue

        # 3) Normalize and apply colormap
        norm8 = normalize_to_uint8(square_img)
        norm_float = norm8 / 255.0
        colored = (cmap(norm_float)[..., :3] * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

        # 4) Resize to fit within the window (preserve aspect)
        scale = min(win_w / frame_bgr.shape[1], win_h / frame_bgr.shape[0])
        disp_size = (int(frame_bgr.shape[1] * scale), int(frame_bgr.shape[0] * scale))
        resized = cv2.resize(frame_bgr, disp_size, interpolation=cv2.INTER_AREA)

        # 5) Create canvas and overlay scale/timestamp
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        y_off = (win_h - disp_size[1]) // 2
        x_off = (win_w - disp_size[0]) // 2

        timestamp = idx / fps

        draw_scale_and_timestamp(
            resized,
            timestamp,
            afm_data.pixel_size_nm,
            scale,
            bar_length_nm=scale_bar_nm,
        )

        canvas[
            y_off : y_off + disp_size[1], x_off : x_off + disp_size[0]  # noqa
            ] = resized
        

        cv2.imshow(window_name, canvas)

        # 6) Handle keypresses
        key = cv2.waitKey(delay_ms) & 0xFF

        # Quit (ESC or 'q')
        if key in (27, ord("q")):
            break

        # 'f': apply filters now
        elif key == ord("f"):
            steps = filter_steps if filter_steps else ["topostats_flatten"]
            logger.info(f"[play] Applying filters: {steps} …")
            flat_stack = afm_data.apply(steps)
            showing_flat = True
            raw_data = afm_data.processed["raw"]

        # SPACE: toggle raw vs. filtered (only if flat_stack exists)
        elif key == ord(" ") and flat_stack is not None:
            showing_flat = not showing_flat

        # 't': export current view as OME-TIFF
        elif key == ord("t"):
            save_dir = _make_save_dir(output_dir)
            base = _compute_out_stem(output_name, afm_data.file_path.stem)
            stack_view = raw_data if not showing_flat else flat_stack
            _export_tiff(
                afm_data, stack_view, save_dir, base_stem=base, filtered=showing_flat
            )

        # 'n': export current view as NPZ
        elif key == ord("n"):
            save_dir = _make_save_dir(output_dir)
            base = _compute_out_stem(output_name, afm_data.file_path.stem)
            stack_view = raw_data if not showing_flat else flat_stack
            _export_npz(
                afm_data, stack_view, save_dir, base_stem=base, filtered=showing_flat
            )

        # 'h': export current view as HDF5
        elif key == ord("h"):
            save_dir = _make_save_dir(output_dir)
            base = _compute_out_stem(output_name, afm_data.file_path.stem)
            stack_view = raw_data if not showing_flat else flat_stack
            _export_h5(
                afm_data, stack_view, save_dir, base_stem=base, filtered=showing_flat
            )

        # 'g': export current view as GIF
        elif key == ord("g"):
            save_dir = _make_save_dir(output_dir)
            base = _compute_out_stem(output_name, afm_data.file_path.stem)
            stack_view = raw_data if not showing_flat else flat_stack
            _export_gif(
                afm_data,
                stack_view,
                save_dir,
                base_stem=base,
                filtered=showing_flat,
                scale_bar_nm=scale_bar_nm,
            )

        # Advance to next frame
        idx = (idx + 1) % num_frames

    # Clean up
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        logger.debug("Window closed manually.")
