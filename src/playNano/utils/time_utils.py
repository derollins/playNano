"""Utility functions for time operations and timestamps in playNano."""

from __future__ import annotations

from datetime import datetime
from importlib.resources import files

# Allow compatability with Python 3.10
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
    scale: float,
    bar_length_nm: int = 100,
    font_scale: float = 1,
    draw_ts: bool = True,
    draw_scale: bool = True,
    color: tuple = (255, 255, 255),
) -> np.ndarray:
    """
    Draw a scale bar and/or timestamp onto an image (in-place via PIL).

    Parameters
    ----------
    image : np.ndarray (HxWx3, uint8)
    timestamp : float
    pixel_size_nm : float
    scale : float
    bar_length_nm : int
    draw_ts: whether to draw the 'Time: xx.xx s' in top-left
    draw_scale: whether to draw the scale bar + nm label in bottom-left
    font_scale: relative multiplier
    draw_ts : bool          # whether to draw timestamp
    draw_scale : bool       # whether to draw scale bar
    color : tuple           # RGB color for both
    """
    # Convert to PIL for drawing
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    W, H = pil.size

    # ==== Font setup ====
    steps_font_path = files("playNano.fonts").joinpath("Steps-Mono/Steps-Mono.otf")

    # compute a point size to match the GUI's QFont
    ptsize = int(15 * font_scale)
    try:
        font = ImageFont.truetype(steps_font_path, ptsize)
    except Exception:
        # fallback to default
        font = ImageFont.load_default()

    # Helper to measure text size
    def measure(text):
        try:
            return font.getsize(text)
        except AttributeError:
            bbox = draw.textbbox((0, 0), text, font=font)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])

    # ==== DRAW TIMESTAMP ====
    if draw_ts:
        ts_text = f"{timestamp:.2f} s"
        tw, th = measure(ts_text)
        y_offset = th + 2
        draw.text((10, y_offset), ts_text, font=font, fill=color)

    # ==== DRAW SCALE BAR ====
    if draw_scale and bar_length_nm > 0 and pixel_size_nm and pixel_size_nm > 0:
        bar_h = 5
        # compute pixel length: bar_length_nm / pixel_size_nm * scale_factor
        px_per_nm = 1.0 / pixel_size_nm
        raw_bar_px = bar_length_nm * px_per_nm
        # in GUI, bar drawn on the scaled pixmap → simulate via scale
        bar_w = int(raw_bar_px * scale)
        x0, y0 = 10, H - 22
        x1, y1 = x0 + bar_w, y0 + bar_h

        # draw filled rectangle
        draw.rectangle([x0, y0, x1, y1], fill=color)

        # label above bar
        label = f"{bar_length_nm} nm"
        lw, lh = measure(label)
        draw.text((x0, y0 - lh - 9), label, font=font, fill=color)

    # back to numpy
    return np.array(pil)


def utc_now_iso() -> str:
    """Return a ISO 8601 UTC timestamp."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
