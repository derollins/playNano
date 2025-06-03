"""Utility functions for playNano."""

import cv2
import numpy as np


def pad_to_square(img: np.ndarray, border_color: int = 0) -> np.ndarray:
    """Pad a 2D grayscale image to a square canvas by centring it."""
    h, w = img.shape[:2]
    size = max(h, w)
    canvas = np.full((size, size), border_color, dtype=img.dtype)
    y = (size - h) // 2
    x = (size - w) // 2
    canvas[y : y + h, x : x + w] = img  # noqa
    return canvas


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
    """
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def draw_scale_and_timestamp(
    image: np.ndarray,
    timestamp: float,
    pixel_size_nm: float,
    scale: float,
    bar_length_nm: int = 100,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    color: tuple = (255, 255, 255),
) -> np.ndarray:
    """
    Draw a scale bar and timestamp onto an image (in-place).

    Parameters
    ----------
    image : np.ndarray
        The image to annotate (uint8, 3 channels expected).
    timestamp : float
        Time in seconds to display.
    pixel_size_nm : float
        Size of one pixel in nanometers.
    scale : float
        Display scale factor (e.g., for resized images).
    bar_length_nm : int, optional
        Length of the scale bar in nanometers, by default 100.
    font_scale : float, optional
        Scale factor for the timestamp and label font.
    font_thickness : int, optional
        Thickness of the font lines.
    color : tuple, optional
        Color of text and scale bar in BGR format, by default white.

    Returns
    -------
    np.ndarray
        The annotated image (same array as input, modified in-place).
    """
    h, _ = image.shape[:2]
    px_per_nm = 1.0 / pixel_size_nm
    bar_length_px = int(bar_length_nm * px_per_nm * scale)
    bar_height = 5

    bar_x = 10
    bar_y = h - 20

    # Draw scale bar
    cv2.rectangle(
        image, (bar_x, bar_y), (bar_x + bar_length_px, bar_y + bar_height), color, -1
    )  # noqa
    cv2.putText(
        image,
        f"{bar_length_nm} nm",
        (bar_x, bar_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        font_thickness,
    )

    # Draw timestamp
    cv2.putText(
        image,
        f"Time: {timestamp:.2f} s",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale + 0.1,
        color,
        font_thickness + 1,
    )

    return image
