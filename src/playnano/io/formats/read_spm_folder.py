"""
Module to load a folder of NanoScope .spm files as an AFMImageStack.

Files contained within the same folder are read together.
Files read with the height data in nm.

Timing (per-frame): real ``Relative frame time`` values from each file's header
are used to compute both ``timestamp`` (seconds from frame 0) and
``frame_duration_s`` (gap to the next frame). Real values capture the actual
inter-frame jitter and dead time between acquisitions, which the synthesised
``lines / Scan Rate`` estimate would miss. If any file is missing that field
the reader falls back to the synth interval for the whole stack.

Frame direction (``scan_direction``) is stored per frame because a NanoScope time
series can capture alternating up/down frames. Absolute time (``start_epoch_ms``)
is derived from each file's ``Date`` field.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from AFMReader import spm

from playnano.afm_stack import AFMImageStack
from playnano.utils.io_utils import build_frame_metadata

logger = logging.getLogger(__name__)


def parse_spm_header(file_path, max_bytes=65536):
    """
    Extract ASCII header key-value pairs from a .spm file.

    NanoScope headers are backslash-prefixed 'key: value' lines at the start
    of the file, followed by the binary image data at ``Data offset``. This
    reads the first ``max_bytes`` and parses whatever key/value lines it finds.

    Parameters
    ----------
    file_path : str or Path
        Path to the .spm file.
    max_bytes : int, optional
        Bytes to read from the start of the file. The default (65 kB) covers
        the fields this reader needs on typical Dimension/BioScope files, but
        very large headers may need more.

    Returns
    -------
    dict[str, str]
        Mapping of header keys to values as strings (values keep their unit
        suffixes; use ``_first_float`` to strip them).
    """
    header_dict = {}
    with open(file_path, "rb") as f:
        raw = f.read(max_bytes)
        text = raw.decode("latin1", errors="ignore")

    for line in text.splitlines():
        if line.startswith("\\"):
            try:
                key, value = line[1:].split(":", 1)
                header_dict[key.strip()] = value.strip()
            except ValueError:
                continue  # skip malformed lines
    return header_dict


def _first_float(header, key):
    """Return leading numeric part of ``header[key]`` (strips unit suffix), or None."""
    # NanoScope values often carry units, e.g. 'Scan Rate: 3.38041' but also
    # 'Slow Axis Size: 1000 nm' or 'Scan Size: 503.906 503.906 nm'. Taking the
    # first whitespace-delimited token gets the number without needing per-key
    # unit knowledge.
    v = header.get(key)
    if v is None:
        return None
    try:
        return float(str(v).split()[0])
    except ValueError:
        return None


def _parse_spm_date(s):
    """
    Parse a NanoScope ``Date`` string to a naive local datetime, or None.

    NanoScope writes wall-clock time on the acquisition PC, e.g.
    ``'04:38:14 PM Fri Sep 11 2026'``: 12-hour clock with AM/PM, then a day
    abbreviation, month, day, year. No timezone, no sub-second component.
    """
    if s is None:
        return None
    try:
        return datetime.strptime(str(s).strip(), "%I:%M:%S %p %a %b %d %Y")
    except ValueError:
        return None


def _read_spm_header_fields(fpath):
    """
    Read the timing/direction fields this reader cares about from one .spm file.

    Returns
    -------
    tuple[float | None, datetime | None, str | None]
        ``(relative_frame_time_s, start_datetime, scan_direction)``. Any element may be
        ``None`` if the corresponding field is absent or unparseable — the
        caller decides how to fall back.

    Notes
    -----
    ``Frame direction`` is the *actual* direction of this frame (per-file), not
    the *setting* (``Capture direction``, which is what the operator asked for).
    Use ``Frame direction`` for motion: in default "capture both" mode it's
    what alternates across the sequence.
    """
    h = parse_spm_header(fpath)
    rel_t = _first_float(h, "Relative frame time")
    start = _parse_spm_date(h.get("Date"))
    fd = h.get("Frame direction")
    # Map NanoScope's 'Up'/'Down' to playNano's convention across readers.
    motion = (
        None
        if fd is None
        else {"Down": "topDown", "Up": "bottomUp"}.get(fd.strip(), fd.strip())
    )
    return rel_t, start, motion


def load_spm_folder(folder_path: Path | str, channel: str) -> AFMImageStack:
    """
    Load a NanoScope .spm time-series folder into an AFMImageStack.

    Real per-frame timing is used when ``Relative frame time`` is present in
    every file, giving actual inter-frame gaps rather than a uniform synthetic
    interval. Frame direction is stored per frame so a series with alternating
    up/down frames is faithfully represented.

    Parameters
    ----------
    folder_path : Path | str
        Folder containing the .spm frames. Files with numeric extensions
        (``.001``, ``.002`` …) are treated as .spm too.
    channel : str
        Channel name to extract.

    Returns
    -------
    AFMImageStack
        Loaded stack. ``pixel_size_nm`` is taken from the first frame;
        per-frame values live in ``frame_metadata`` (``timestamp``,
        ``frame_pixel_size_nm``, ``frame_duration_s``, ``start_epoch_ms``,
        ``scan_direction``, ``line_rate``).

    Notes
    -----
    The final frame's ``frame_duration_s`` is ``None`` — there's no frame N+1
    to measure against, and inventing a duration would silently claim
    information the file doesn't contain.

    ``start_epoch_ms`` is derived from ``Date``, which is naive local time
    with second precision. It is good enough for provenance but should not be
    used for sub-second alignment across machines.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a directory.")

    # NanoScope writes .spm or numeric extensions like .001/.002/…
    spm_files = sorted(
        f
        for f in folder.iterdir()
        if f.is_file()
        and (
            f.suffix.lower() == ".spm"
            or (f.suffix[1:].isdigit() and len(f.suffix) == 4)
        )
    )
    if not spm_files:
        raise FileNotFoundError(f"No .spm files found in {folder}.")
    logger.info(f"Found {len(spm_files)} .spm files.")

    # --- Shape, dtype, and the fallback line rate from the first file ---
    first_img, first_pixel_size_nm = spm.load_spm(spm_files[0], channel)
    height_px, width_px = first_img.shape
    dtype = first_img.dtype
    num_frames = len(spm_files)
    image_stack = np.empty((num_frames, height_px, width_px), dtype=dtype)

    first_header = parse_spm_header(spm_files[0])
    try:
        line_rate = float(first_header.get("Scan Rate"))
    except (TypeError, ValueError):
        line_rate = None
    lines_per_frame = height_px
    if line_rate is None or lines_per_frame is None:
        raise ValueError(
            f"Missing data: line_rate={line_rate}, lines_per_frame={lines_per_frame}"
        )
    # Synthetic per-frame interval, used only as the fallback below.
    frame_interval = lines_per_frame / line_rate

    # --- Per-file header fields for timing/direction, read ONCE ---
    rel_ts, starts, motions = zip(
        *[_read_spm_header_fields(fp) for fp in spm_files], strict=False
    )

    # Real per-frame timing when every file has 'Relative frame time'.
    # Otherwise fall back to the synth interval for the whole stack — mixing
    # real and synth values within one stack would silently produce a
    # misleading timeline.
    if all(t is not None for t in rel_ts):
        t0 = rel_ts[0]
        timestamps = np.array([t - t0 for t in rel_ts])
        if len(rel_ts) > 1:
            diffs = np.diff(rel_ts).astype(float)
            # No frame N+1 to measure against, so the last frame's duration is unknown.
            durations = np.concatenate([diffs, [np.nan]])
        else:
            durations = np.array([np.nan])
    else:
        logger.warning(
            "Some .spm files missing 'Relative frame time';"
            " falling back to synth timing."
        )
        timestamps = np.arange(num_frames) * frame_interval
        durations = np.full(num_frames, frame_interval)

    def _epoch_ms(dt):
        """
        Convert naive local datetime -> UTC epoch ms.

        This is approximate: assumes UTC, second precision.
        """
        return (
            None
            if dt is None
            else int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
        )

    # --- Per-frame loop: pixels + metadata ---
    frame_metadata = []
    for i, fpath in enumerate(spm_files):
        logger.debug(f"Loading {fpath.name}")
        img, px_size_nm = spm.load_spm(fpath, channel)
        if img.shape != (height_px, width_px):
            raise ValueError(f"Inconsistent image shape in {fpath}")
        image_stack[i] = img

        # NaN durations -> None so metadata stays clean (consumers tolerate None).
        dur = durations[i]
        frame_duration_s = None if not np.isfinite(dur) else float(dur)

        frame_metadata.append(
            build_frame_metadata(
                timestamp=float(timestamps[i]),
                frame_pixel_size_nm=px_size_nm,
                line_rate=line_rate,
                frame_duration_s=frame_duration_s,
                start_epoch_ms=_epoch_ms(starts[i]),
                scan_direction=motions[i],
            )
        )

    logger.debug(f"Loaded {num_frames} frames with shape {image_stack.shape}.")
    return AFMImageStack(
        data=image_stack,
        pixel_size_nm=first_pixel_size_nm,
        channel=channel,
        file_path=str(folder),
        frame_metadata=frame_metadata,
    )
