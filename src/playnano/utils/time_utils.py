"""Helpers for frame times and for drawing the time and scale bar on frames."""

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
    Convert each frame's ``"timestamp"`` to seconds as a float, or None.

    - Python ``int`` and ``float``: converted to ``float`` (``bool`` counts
      as ``int``).
    - :class:`datetime.datetime`: seconds since 1970 (Unix time).
    - ISO 8601 strings: parsed, then seconds since 1970. Times without a time
      zone are read as local time.
    - Anything else, including NumPy numbers such as ``numpy.int64``, a
      missing key and strings that cannot be parsed: None.

    Parameters
    ----------
    metadata_list : list of dict
        Per-frame metadata. Not changed.

    Returns
    -------
    list of dict
        Shallow copies of the dicts, each with a ``"timestamp"`` key.

    Examples
    --------
    >>> from playnano.utils.time_utils import normalize_timestamps
    >>> normalize_timestamps([{"timestamp": 1}, {}, {"timestamp": "bad"}])
    [{'timestamp': 1.0}, {'timestamp': None}, {'timestamp': None}]
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
    Draw the time and a scale bar on an RGB frame.

    The time is drawn at the top left as ``"1.23 s"``, the scale bar and its
    label (``"100 nm"``) at the bottom left, in the bundled Steps-Mono font
    (or Pillow's default font if it cannot be loaded).

    Parameters
    ----------
    image : numpy.ndarray
        RGB frame, shape (height, width, 3), dtype uint8. Not changed.
    timestamp : float
        Time to show, in seconds.
    pixel_size_nm : float
        Pixel size of the frame before resizing, in nm.
    resize_scale : float
        How much the frame was enlarged (new height / old height). Sets the
        scale bar's length in pixels.
    annotation_scale : float
        Frame height divided by the 512-pixel reference height. Sets margins,
        bar thickness and text positions so they look the same at any size.
    bar_length_nm : int, optional
        Length of the scale bar, in nm. Default is 100.
    font_scale : float, optional
        Text height is ``15 * font_scale`` pixels. Default is 1.
    draw_ts : bool, optional
        Draw the time. Default is True.
    draw_scale : bool, optional
        Draw the scale bar, if ``bar_length_nm`` and ``pixel_size_nm`` are both
        positive. Default is True.
    color : tuple of int, optional
        RGB colour of the text and bar. Default is white, ``(255, 255, 255)``.

    Returns
    -------
    numpy.ndarray
        A new annotated frame, shape (height, width, 3), dtype uint8.

    Examples
    --------
    >>> import numpy as np
    >>> from playnano.utils.time_utils import draw_scale_and_timestamp
    >>> frame = np.zeros((100, 100, 3), dtype=np.uint8)
    >>> out = draw_scale_and_timestamp(frame, 1.5, 1.0, 1.0, 1.0, bar_length_nm=50)
    >>> out.shape, out.dtype
    ((100, 100, 3), dtype('uint8'))
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
        For example ``"2024-01-15T10:30:00.123456Z"``.
    """
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
