"""
Timestamp and annotation utilities for playNano.

This module provides functions for normalising per-frame timestamps,
drawing scale bar and timestamp overlays onto AFM image frames, and
generating UTC timestamps for provenance records.

Functions
---------
normalize_timestamps
    Parse and normalise per-frame timestamp metadata to float seconds.
draw_scale_and_timestamp
    Draw a scale bar and timestamp overlay onto an AFM image frame.
utc_now_iso
    Return the current UTC time as an ISO 8601 string.
"""

from __future__ import annotations

from datetime import datetime
from importlib.resources import files

# Allow compatibility with Python 3.10
try:
    from datetime import UTC
except ImportError:
    from datetime import timezone

    UTC = timezone.utc

from typing import Any

import dateutil.parser
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
    resize_scale: float,
    annotation_scale: float,
    bar_length_nm: int = 100,
    font_scale: float = 1,
    draw_ts: bool = True,
    draw_scale: bool = True,
    color: tuple = (255, 255, 255),
) -> np.ndarray:
    """
    Draw a scale bar and/or timestamp overlay onto an AFM image frame.

    Two separate scale factors are used to decouple physical accuracy from
    visual consistency. ``resize_scale`` ensures the scale bar length correctly
    represents the physical nanometre distance regardless of how the frame was
    resized. ``annotation_scale`` ensures margins, bar thickness, and text
    positions appear proportionally identical at any output resolution.

    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3), dtype uint8.
    timestamp : float
        Timestamp in seconds to display in the top-left corner.
    pixel_size_nm : float
        Physical size of one pixel in the original (pre-resize) frame,
        in nanometres. Used to compute the scale bar length in pixels.
    resize_scale : float
        Factor by which the frame was upscaled from its original resolution
        (``output_height / original_height``). Used only for computing the
        physically accurate scale bar width.
    annotation_scale : float
        Factor relative to ``REFERENCE_HEIGHT`` used to scale all annotation
        geometry — margins, bar thickness, label gaps, and text position
        (``output_height / REFERENCE_HEIGHT``). Ensures annotations occupy
        the same visual proportion at any output resolution.
    bar_length_nm : int
        Desired scale bar length in nanometres. Default is 100.
    font_scale : float
        Multiplier applied to the base font point size (15 pt). Default is 1.
    draw_ts : bool
        Whether to draw the timestamp in the top-left corner. Default is
        ``True``.
    draw_scale : bool
        Whether to draw the scale bar and label in the bottom-left corner.
        Default is ``True``.
    color : tuple
        RGB colour for all annotations. Default is white ``(255, 255, 255)``.

    Returns
    -------
    np.ndarray
        Annotated image as a uint8 array of shape (H, W, 3).

    Notes
    -----
    - Requires the Steps-Mono font bundled in ``playnano.resources.fonts``.
      Falls back to the PIL default font if the file cannot be loaded.
    - The scale bar is only drawn when ``pixel_size_nm > 0`` and
      ``bar_length_nm > 0``.
    """
    # Convert to PIL for drawing
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    W, H = pil.size

    # ==== Font setup ====
    steps_font_path = files("playnano.resources.fonts").joinpath(
        "Steps-Mono/Steps-Mono.otf"
    )

    # compute a point size to match the GUI's QFont
    ptsize = int(15 * font_scale)
    try:
        font = ImageFont.truetype(steps_font_path, ptsize)
    except Exception:
        # fallback to default
        font = ImageFont.load_default()

    # Helper to measure text size
    def measure(text):
        """Measure text size in pixels, compatible with different PIL versions."""
        try:
            return font.getsize(text)
        except AttributeError:
            bbox = draw.textbbox((0, 0), text, font=font)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])

    # Scale all pixel margins/sizes proportionally
    margin = int(10 * annotation_scale)
    bar_h = max(1, int(5 * annotation_scale))
    bottom_margin = int(22 * annotation_scale)
    label_gap = int(9 * annotation_scale)
    ts_gap = int(2 * annotation_scale)

    # ==== DRAW TIMESTAMP ====
    if draw_ts:
        ts_text = f"{timestamp:.2f} s"
        tw, th = measure(ts_text)
        y_offset = th + ts_gap
        draw.text((margin, y_offset), ts_text, font=font, fill=color)

    # ==== DRAW SCALE BAR ====
    if draw_scale and bar_length_nm > 0 and pixel_size_nm and pixel_size_nm > 0:
        px_per_nm = 1.0 / pixel_size_nm
        raw_bar_px = bar_length_nm * px_per_nm
        bar_w = int(raw_bar_px * resize_scale)
        x0, y0 = margin, H - bottom_margin
        x1, y1 = x0 + bar_w - 1, y0 + bar_h - 1
        draw.rectangle([x0, y0, x1, y1], fill=color)
        label = f"{bar_length_nm} nm"
        lw, lh = measure(label)
        draw.text((x0, y0 - lh - label_gap), label, font=font, fill=color)

    return np.array(pil)


def utc_now_iso() -> str:
    """
    Return the current UTC time as an ISO 8601 string.

    Returns
    -------
    str
        UTC timestamp in ISO 8601 format, e.g. ``'2024-01-15T10:30:00Z'``.
    """
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
