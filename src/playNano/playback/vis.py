"""Module for opening a window to interactively view and export AFM image stacks."""

import logging
import sys
from typing import Optional

import cv2
import numpy as np
from matplotlib import colormaps as cm

from playNano.io.export import export_bundles
from playNano.io.gif_export import export_gif
from playNano.processing.pipeline import ProcessingPipeline
from playNano.stack.afm_stack import AFMImageStack
from playNano.utils.io_utils import (
    normalize_to_uint8,
    pad_to_square,
    prepare_output_directory,
)
from playNano.utils.time_utils import draw_scale_and_timestamp

logger = logging.getLogger(__name__)

default_steps_with_kwargs = [
    ("remove_plane", {}),
    ("polynomial_flatten", {"order": 2}),
    ("mask_mean_offset", {"factor": 0.5}),
    ("row_median_align", {}),
    ("polynomial_flatten", {"order": 2}),
    ("zero_mean", {}),
]


def play_stack_cv(
    afm_data: AFMImageStack,
    fps: float,
    window_name: str = "AFM Stack Viewer",
    output_dir: str = "./",
    output_name: str = "",
    steps_with_kwargs: Optional[list[tuple[str, dict[str, object]]]] = None,
    scale_bar_nm: int = 100,
) -> None:
    """
    Pop up an OpenCV window and play a 3D AFM stack as video.

    This function allows you to view a 3D AFM image stack as a video,
    with on-the-fly filtering & export.

    Press
      - 'f' to apply the filters in `steps_with_kwargs` (default to
      'default_steps_with_kwargs' if none provided),
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
    steps_with_kwargs : list of tuples (str, object), optional
        A list of tuple containing filter names (in order) with keyword arguments
         to apply when 'f' is pressed If None or empty,
         defaults to [default_steps_with_kwargs].

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

    if not steps_with_kwargs:
        steps_with_kwargs = []

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

        canvas[y_off : y_off + disp_size[1], x_off : x_off + disp_size[0]] = resized

        cv2.imshow(window_name, canvas)

        # 6) Handle keypresses
        key = cv2.waitKey(delay_ms) & 0xFF

        # Quit (ESC or 'q')
        if key in (27, ord("q")):
            break

        # 'f': apply filters
        elif key == ord("f"):
            # If no filters configured, use default workflow
            if not steps_with_kwargs:
                logger.info("[play] No filters configured. Using default preset.")
                steps_with_kwargs = default_steps_with_kwargs

            # Apply the processing pipeline with the configured steps
            logger.info(f"[play] Applying filters: {steps_with_kwargs} …")
            processing = ProcessingPipeline(afm_data)
            for step_name, step_kwargs in steps_with_kwargs:
                if step_name == "clear":
                    processing.clear_mask()
                else:
                    try:
                        processing.add_filter(step_name, **step_kwargs)
                    except Exception as e:
                        logger.error(f"Error adding filter '{step_name}': {e}")
                        sys.exit(1)
            try:
                processing.run()
                flat_stack = afm_data.data
                raw_data = afm_data.processed["raw"]
                showing_flat = True
            except Exception as e:
                logger.error(f"Error in processing.run(): {e}")
                sys.exit(1)

        # SPACE: toggle raw vs. filtered (only if flat_stack exists)
        elif key == ord(" ") and flat_stack is not None:
            showing_flat = not showing_flat

        # 't': export current view as OME-TIFF
        elif key == ord("t"):
            save_dir = prepare_output_directory(output_dir, "output")
            raw = False
            if showing_flat is False and "raw" in afm_data.processed:
                raw = True
            export_bundles(afm_data, save_dir, output_name, ["tif"], raw)

        # 'n': export current view as NPZ
        elif key == ord("n"):
            save_dir = prepare_output_directory(output_dir, "output")
            if showing_flat:
                raw = False
            elif showing_flat is False and "raw" in afm_data.processed:
                raw = True
            export_bundles(afm_data, save_dir, output_name, ["npz"], raw)

        # 'h': export current view as HDF5
        elif key == ord("h"):
            save_dir = prepare_output_directory(output_dir, "output")
            if showing_flat:
                raw = False
            elif showing_flat is False and "raw" in afm_data.processed:
                raw = True
            export_bundles(afm_data, save_dir, output_name, ["h5"], raw)

        # 'g': export current view as GIF
        elif key == ord("g"):
            save_dir = prepare_output_directory(output_dir, "output")
            raw = False
            if showing_flat is False and "raw" in afm_data.processed:
                raw = True
            export_gif(
                afm_data,
                True,
                save_dir,
                output_name,
                scale_bar_nm=scale_bar_nm,
                raw=raw,
            )

        # Advance to next frame
        idx = (idx + 1) % num_frames

    # Clean up
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        logger.debug("Window closed manually.")
