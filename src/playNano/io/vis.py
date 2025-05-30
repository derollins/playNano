"""Module for opening a window to show data."""

import cv2
import numpy as np

from playNano.processing.image_processing import flatten_stack


def pad_to_square(img: np.ndarray, border_color: int = 0) -> np.ndarray:
    """Pad a 2D grayscale image to a square canvas by centring it."""
    h, w = img.shape[:2]
    size = max(h, w)
    # create a square canvas
    canvas = np.full((size, size), border_color, dtype=img.dtype)
    # compute top-left corner
    y = (size - h) // 2
    x = (size - w) // 2
    canvas[y:y + h, x:x + w] = img
    return canvas


def play_stack_cv(
    image_stack: np.ndarray,
    pixel_size_nm: float,
    fps: float,
    window_name: str = "AFM Stack Viewer",
) -> None:
    """
    Pop up an OpenCV window and play a 3D stack as a video.

    Left-click anywhere to toggle between raw and flattened views.
    Press SPACE to (re)flatten on the fly.
    Press ESC or 'q' to quit.
    """
    delay_ms = int(50 / fps) if fps > 0 else 50

    # Determine image size after padding to square
    frame_shape = pad_to_square(image_stack[0]).shape
    square_size = frame_shape[0]  # since it's square, height == width

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, square_size, square_size)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, 1)

    idx = 0
    num_frames = image_stack.shape[0]
    flat_stack = None
    showing_flat = False

    while True:
        frame = flat_stack[idx] if showing_flat else image_stack[idx]
        square = pad_to_square(frame)
        # normalize to 0..255 and convert to BGR for display
        norm = cv2.normalize(square, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # Get current window size (width, height)
        win_w = int(cv2.getWindowImageRect(window_name)[2])
        win_h = int(cv2.getWindowImageRect(window_name)[3])

        # Determine scale factor to fit image in window while preserving aspect ratio
        scale = min(win_w / bgr.shape[1], win_h / bgr.shape[0])
        disp_size = (int(bgr.shape[1] * scale), int(bgr.shape[0] * scale))

        # Resize for display
        resized_bgr = cv2.resize(bgr, disp_size, interpolation=cv2.INTER_AREA)

        # Create black canvas the size of the window
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

        # Compute top-left coordinates to center image
        y_offset = (win_h - disp_size[1]) // 2
        x_offset = (win_w - disp_size[0]) // 2

        # Paste resized image into canvas
        canvas[
            y_offset:y_offset + disp_size[1], x_offset:x_offset + disp_size[0]
        ] = resized_bgr

        # Show the final canvas
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (27, ord("q")):  # ESC or q
            break

        # Space: compute the flattened stack once
        elif key == ord(" "):  # space: flatten on the fly
            flat_stack = flatten_stack(image_stack, pixel_to_nm=pixel_size_nm)
            showing_flat = True

        # 'f': only toggle *if* we have a flattened stack
        elif key == ord("f") and flat_stack is not None:
            showing_flat = not showing_flat

        idx = (idx + 1) % num_frames

    cv2.destroyWindow(window_name)
