"""Utility functions for IO operations in playNano."""

from pathlib import Path

import cv2
import numpy as np

INVALID_CHARS = r'\/:*?"<>|'
INVALID_FOLDER_CHARS = r'*?"<>|'


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


def sanitize_output_name(name: str, default: str) -> str:
    """
    Sanitize output file names by removing extensions and stripping whitespace.

    Parameters
    ----------
    name : str
        The output file name provided by the user.
    default : str
        Default name to use if `name` is empty or None.

    Returns
    -------
    str
        Sanitized base file name without extension.
    """
    if not name:
        return default
    name = name.strip()
    # Remove extension if any
    try:
        name = Path(name).with_suffix("").name
    except ValueError:
        return default

    if any(c in name for c in INVALID_CHARS):
        raise ValueError(f"Invalid characters in output name: {INVALID_CHARS}")

    return name


def prepare_output_directory(folder: str | None, default: str = "output") -> Path:
    """
    Validate, resolve, and create the output directory if it doesn't exist.

    Parameters
    ----------
    folder : str or None
        User-provided output folder path. If None, use `default`.
    default : str, optional
        Default folder name to use if `folder` not specified.

    Returns
    -------
    Path
        A Path object pointing to the created output directory.

    Raises
    ------
    ValueError
        If any part of the folder path contains invalid characters.
    """
    if folder is None:
        folder = default
    elif type(folder) is not str:
        try:
            folder = str(folder)
        except Exception as e:
            raise ValueError(f"Invalid folder path: {e}") from e
    folder = folder.strip() if folder else default
    folder_path = Path(folder).resolve()
    for part in folder_path.parts:
        if any(c in part for c in INVALID_FOLDER_CHARS):
            raise ValueError(
                f"Invalid characters in output folder path: {INVALID_FOLDER_CHARS}"
            )
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path
