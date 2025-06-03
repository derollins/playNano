"""Module for opening a window to show data."""

import logging
import sys

import cv2
import numpy as np
from matplotlib import colormaps as cm

from playNano.cli import prepare_output_directory, sanitize_output_name
from playNano.stack.afm_stack import AFMImageStack
from playNano.utils import draw_scale_and_timestamp, normalize_to_uint8, pad_to_square

logger = logging.getLogger(__name__)


def play_stack_cv(
    afm_data: AFMImageStack,
    fps: float,
    window_name: str = "AFM Stack Viewer",
    output_dir="./",
    output_name="",
) -> None:
    """
    Pop up an OpenCV window and play a 3D stack as a video.

    Press 'f' to (re)flatten on the fly.
    Press SPACE to toggle between raw and flattened.
    Press ESC or 'q' to quit.
    """

    # instructions = "Keys: f=apply filter, SPACE= toggle filter, e=export, q=quit"

    delay_ms = int(50 / fps) if fps > 0 else 50

    # Determine image size after padding to square
    frame_shape = afm_data.image_shape
    square_size = frame_shape[0]  # since it's square, height == width

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, square_size, square_size)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, 1)

    idx = 0
    num_frames = afm_data.data.shape[0]
    flat_stack = None
    showing_flat = False
    if "raw" not in afm_data.processed:
        raw_data = afm_data.data
    else:
        raw_data = afm_data.processed["raw"]

    cmap = cm.get_cmap("afmhot")

    while True:
        try:
            win_w, win_h = (
                int(cv2.getWindowImageRect(window_name)[2]),
                int(cv2.getWindowImageRect(window_name)[3]),
            )
        except cv2.error:
            break  # Window closed manually
        frame = flat_stack[idx] if showing_flat else raw_data[idx]
        square = pad_to_square(frame)

        # Skip frame if window is minimized (width or height is zero or negative)
        if win_w <= 0 or win_h <= 0:
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord("q")):  # ESC or q
                break
            continue

        # normalize and apply colourmap
        norm = normalize_to_uint8(square)
        norm_float = norm / 255.0
        colorized = (cmap(norm_float)[..., :3] * 255).astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        color_bgr = cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR)

        # scale to fit window
        scale = min(win_w / color_bgr.shape[1], win_h / color_bgr.shape[0])
        disp_size = (int(color_bgr.shape[1] * scale), int(color_bgr.shape[0] * scale))
        resized_colored = cv2.resize(color_bgr, disp_size, interpolation=cv2.INTER_AREA)

        # Create black canvas the size of the window
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

        # Compute top-left coordinates to center image
        y_offset = (win_h - disp_size[1]) // 2
        x_offset = (win_w - disp_size[0]) // 2

        timestamp = idx / fps
        draw_scale_and_timestamp(
            resized_colored, timestamp, afm_data.pixel_size_nm, scale
        )

        # Paste resized image into canvas
        canvas[
            y_offset : y_offset + disp_size[1],  # noqa: E203
            x_offset : x_offset + disp_size[0],  # noqa: E203
        ] = resized_colored

        # Show the final canvas
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (27, ord("q")):  # ESC or q
            break

        # 'f'': compute the flattened stack once
        elif key == ord("f"):  # 'f': flatten on the fly
            _filter = "topostats_flatten"
            flat_stack = afm_data.apply([_filter])
            showing_flat = True
            raw_data = afm_data.processed["raw"]

        # SPACE : only toggle *if* we have a flattened stack
        elif key == ord(" ") and flat_stack is not None:
            showing_flat = not showing_flat

        # 'e': Filtered image exported as GIF
        elif key == ord("e") and flat_stack is not None:
            from playNano.io.gif_export import create_gif_with_scale_and_timestamp

            # Determine and create output directory
            input_stem = afm_data.file_path.stem
            try:
                save_dir = prepare_output_directory(output_dir)
            except ValueError as e:
                logger.error(str(e))
                sys.exit(1)  # Exit with error code 1
                logger.info(f"Saving outputs to: {save_dir}")

            # Remove stem and whitepace from file name input or use flattened if none provided. # noqa
            try:
                output_stem = sanitize_output_name(output_name, input_stem)
            except ValueError as e:
                logger.error(str(e))
                sys.exit(1)  # Exit with error code 1
            timestamps = [meta["timestamp"] for meta in afm_data.frame_metadata]
            output_path = save_dir / f"{output_stem}_filtered.gif"
            create_gif_with_scale_and_timestamp(
                afm_data.data,
                afm_data.pixel_size_nm,
                timestamps,
                output_path=output_path,
            )
            print(f"Exported filtered GIF to {output_path}")

        idx = (idx + 1) % num_frames

    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        logger.debug("Window closed manually.")
