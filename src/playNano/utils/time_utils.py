"""Utility functions for time operations and timestamps in playNano."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import cv2
import dateutil.parser
import numpy as np


def normalize_timestamps(metadata_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize timestamp data to a float in seconds.

    Given a list of per-frame metadata dicts, parse each 'timestamp' entry
    (if present) into a float (seconds). Returns a new list of dicts
    with 'timestamp' replaced by float or None.

    - ISO-format strings → parsed with dateutil.isoparse()
    - datetime objects       → .timestamp()
    - numeric (int/float)    → float()
    - missing/unparsable     → None

    Parameters
    ----------
    metadata_list : list of dict
        List of metadata dictionaries, each possibly containing a 'timestamp'.

    Returns
    -------
    list of dict
        List of metadata dicts with 'timestamp' normalized to float seconds or None.
    """
    normalized: list[dict[str, Any]] = []
    for md in metadata_list:
        new_md = dict(md)  # shallow copy so we don't mutate the original
        t = new_md.get("timestamp", None)

        if t is None:
            new_md["timestamp"] = None

        elif isinstance(t, str):
            try:
                dt = dateutil.parser.isoparse(t)
                new_md["timestamp"] = dt.timestamp()
            except Exception:
                # parsing failed
                new_md["timestamp"] = None

        elif isinstance(t, datetime):
            new_md["timestamp"] = t.timestamp()

        elif isinstance(t, (int, float)):
            new_md["timestamp"] = float(t)

        else:
            new_md["timestamp"] = None

        normalized.append(new_md)

    return normalized


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
    if bar_length_nm > 0:
        bar_length_px = int(bar_length_nm * px_per_nm * scale)
        bar_height = 5

        bar_x = 10
        bar_y = h - 20

        # Draw scale bar
        cv2.rectangle(
            image,
            (bar_x, bar_y),
            (bar_x + bar_length_px, bar_y + bar_height),
            color,
            -1,  # noqa
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
