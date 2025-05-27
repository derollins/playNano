"""Module for exporting a AFM image stack as a GIF."""

import logging

import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalize a float image to the uint8 [0, 255] range, handling NaNs and Infs.

    Parameters
    ----------
    image : np.ndarray
        Input image as a NumPy array of floats. May contain NaNs or infinite values.

    Returns
    -------
    np.ndarray
        Normalized image as a uint8 NumPy array with values in the range [0, 255].

    Notes
    -----
    NaNs and infinite values are replaced with 0 before normalization.
    If the image has constant values (min == max), a zero array is returned.
    """
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8)
    norm_image = (image - min_val) / (max_val - min_val) * 255
    return norm_image.astype(np.uint8)


def create_gif_with_scale_and_timestamp(
    image_stack,
    pixel_size_nm,
    timestamps=None,
    scale_bar_length_nm=100,
    output_path="output.gif",
    duration=0.5,
    cmap_name="afmhot",
):
    """
    Create a GIF from a stack of images with a scale bar and optional timestamps.

    This function normalises image frames, applies a colormap, adds a scale bar and
    timestamps (if provided), and compiles the frames into an animated GIF.

    Parameters
    ----------
    image_stack : np.ndarray
        3D NumPy array of shape (N, H, W), where N is the number of frames.
    pixel_size_nm : float
        Size of a pixel in nanometers.
    timestamps : list or tuple, optional
        List of timestamps corresponding to each frame (in seconds).
        If not provided, frame indices are used instead.
    scale_bar_length_nm : int, optional
        Length of the scale bar in nanometers. Default is 100.
    output_path : str, optional
        Path to save the output GIF. Default is "output.gif".
    duration : float, optional
        Duration of each frame in seconds. ???
    cmap_name : str, optional
        Name of the matplotlib colormap to use. Default is "afmhot".

    Returns
    -------
    None

    Notes
    -----
    - Frames are normalized and colorized using the specified colormap.
    - A scale bar is drawn in the bottom-left corner of each frame.
    - Timestamps are displayed in the top-left corner if provided and valid.
    - The resulting GIF is saved to the specified output path.
    """
    frames = []
    height_px, width_px = image_stack.shape[1:]
    scale_bar_px = int(scale_bar_length_nm / pixel_size_nm)
    cmap = cm.get_cmap(cmap_name)

    # Try to load a nicer font
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except IOError:
        font = ImageFont.load_default()

    # Check if timestamps are usable
    has_valid_timestamps = (
        timestamps is not None
        and isinstance(timestamps, (list, tuple))
        and len(timestamps) == len(image_stack)
    )

    for i, frame in enumerate(image_stack):
        # Normalize and colorize
        frame_norm = normalize_to_uint8(frame)
        frame_norm = (frame_norm - np.min(frame_norm)) / (np.ptp(frame_norm) + 1e-9)
        color_frame = (cmap(frame_norm)[..., :3] * 255).astype(np.uint8)
        img = Image.fromarray(color_frame)
        draw = ImageDraw.Draw(img)

        # Draw scale bar
        bar_x = int(width_px * 0.05)
        bar_y = int(height_px * 0.95)
        draw.line(
            [(bar_x, bar_y), (bar_x + scale_bar_px, bar_y)], fill="white", width=4
        )
        draw.text(
            (bar_x, bar_y - 20), f"{scale_bar_length_nm} nm", fill="white", font=font
        )

        # Timestamp or frame fallback
        if has_valid_timestamps:
            try:
                timestamp = float(timestamps[i])
                timestamp_text = f"t = {timestamp:.2f} s"
            except (TypeError, ValueError, IndexError):
                timestamp_text = f"Frame {i}"
        else:
            timestamp_text = f"Frame {i}"

        draw.text((10, 10), timestamp_text, fill="grey", font=font)
        frames.append(img)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration * 500),
        loop=0,
    )
    logger.info(f"GIF saved to {output_path}")
