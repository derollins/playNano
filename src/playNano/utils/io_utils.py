"""Utility functions for IO operations in playNano."""

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
