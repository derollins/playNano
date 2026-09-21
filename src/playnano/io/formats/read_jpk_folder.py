"""
Module to load .jpk AFM data files from a folder into Python NumPy arrays.

Files contained within the same folder are read together.
Files read with the height data in nm.
"""

import logging
from pathlib import Path
import re

from datetime import datetime
import numpy as np
import tifffile
from AFMReader.jpk import load_jpk

from playnano.afm_stack import AFMImageStack

logger = logging.getLogger(__name__)

# ratio 1.0 for standard scanning
# ratio 0.5 for bidirectional scanning
_BIDIRECTIONAL_RATIO_THRESHOLD = 0.7

TAG_START_DATE = "32771"
TAG_END_DATE = "32774"
TAG_SETTINGS = "32791"


def _extract_scan_rate(jpk_file: Path) -> float:
    """
    Extract the scan rate in lines per second from a .jpk image file.

    This is the 'Scanrate-Frequency' tag in the JPK TIFF metadata (tag 0x8049) and
    covers the whole trace and retrace including the overscan. For bidirectional
    scanning, the effective scan rate of the image is half this value.

    Parameters
    ----------
    jpk_file : Path
        Path to a .jpk file.

    Returns
    -------
    float
        The scan rate of the image in fast scan lines per second.
    """
    with tifffile.TiffFile(jpk_file) as tif:
        return (
            tif.pages[0].tags["32841"].value
        )  # Return the Scan Rate attribute from the tiff tag value.


def _parse_jpk_date(value: str) -> datetime:
    """
    Parse a JPK date tag ('yyyy-MM-dd HH:mm:ss[.SSS] zz') to a naive datetime.

    Keeps milliseconds when present, drops the trailing timezone abbreviation,
    and tolerates files without the fractional-seconds part.
    """
    body = " ".join(str(value).split()[:2])  # drop 'BST'/'GMT'
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(body, fmt)
        except ValueError:
            continue
    raise ValueError(f"unparseable JPK date: {value!r}")


def _frame_times(jpk_file: Path) -> tuple[datetime, datetime]:
    """
    Extract frame start and end times from a .jpk image file metadata.

    (start, end) datetimes of one image, from StartDate (0x8003 hex, 32771 dec) /
    EndDate (0x8006 hex, 32774 dec). And convert to naive datetimes (drop timezone).
    """
    with tifffile.TiffFile(jpk_file) as tif:
        tg = tif.pages[0].tags
        return _parse_jpk_date(tg[TAG_START_DATE].value), _parse_jpk_date(
            tg[TAG_END_DATE].value
        )


def _bidirectional_from_timing(
    duration_s, scanrate_hz, slow_lines, thresh=_BIDIRECTIONAL_RATIO_THRESHOLD
) -> tuple[bool, float]:
    """
    Infer bidirectional (interlaced) scanning from frame duration and scan rate.

    Infer interlace from documented tags only. Interlaced frames take ~half the naive
    slow_lines/scanrate time -> ratio ~0.5; normal ~1.0.
    """
    ratio = duration_s * scanrate_hz / slow_lines
    return ratio < thresh, ratio


def _bidirectional_from_blob(jpk_file: Path) -> bool:
    """
    Read bidirectional (interlaced) scanning from the JPK TIFF metadata settings blob.

    The 0x8017 (32791) tag contains a text blob with key-value pairs, including
    fast-imaging.feed-forward-parameters.yaxis.interlace = true/false which indicates
    bidirectional scanning.
    """
    with tifffile.TiffFile(jpk_file) as tif:
        tg = tif.pages[0].tags
        blob = tg[TAG_SETTINGS].value if TAG_SETTINGS in tg else ""

    def grab(k: str) -> str | None:
        match = re.search(re.escape(k) + r"\s*:\s*([^\n]+)", blob)
        return match.group(1) if match else None

    yaxis_interlace = grab("fast-imaging.feed-forward-parameters.yaxis.interlace")

    if yaxis_interlace is None:
        logger.warning(
            "Interlace setting not found in %s. Assuming False (likely not HS).",
            jpk_file,
        )
        return False

    return yaxis_interlace.strip().lower() == "true"


def _frame_timing(
    jpk_files, lines_per_frame
) -> tuple[np.ndarray, np.ndarray, list[datetime | None]]:
    """
    Per-frame timestamps (s from first frame) and durations (s).

    Primary: real StartDate/EndDate tags (ms precision, correct for interlaced and
    standard alike, and captures real inter-frame jitter / dropped frames).
    Fallback (if any date is unreadable): scan-rate timing, halved when interlaced.

    Notes
    -----
    If any frame contains unreadable or missing StartDate/EndDate metadata,
    timing falls back to scan-rate estimation for the entire stack to ensure
    a consistent timestamp source across all frames.
    """
    try:
        starts, ends = zip(*(_frame_times(fp) for fp in jpk_files))
        t0 = starts[0]
        timestamps = np.array([(s - t0).total_seconds() for s in starts])
        durations = np.array([(e - s).total_seconds() for s, e in zip(starts, ends)])
    except (KeyError, ValueError) as exc:
        rate = _extract_scan_rate(jpk_files[0])  # lines per second
        if not rate:
            raise ValueError(
                f"No StartDate/EndDate and no scan rate in {jpk_files[0].name}; cannot time frames."
            ) from exc
        logger.warning(
            "One or more JPK StartDate/EndDate tags were unreadable (%s); "
            "using scan-rate timing for the entire stack.",
            exc,
        )
        interval = lines_per_frame / rate
        if _bidirectional_from_blob(jpk_files[0]):
            interval /= 2.0  # interlaced images at ~2x
        timestamps = np.arange(len(jpk_files)) * interval
        durations = np.full(len(jpk_files), interval)
        starts = [None] * len(jpk_files)
    return timestamps, durations, list(starts)


def _scan_direction(jpk_file: Path) -> str:
    """
    Read scan direction from JPK TIFF Motion tag (0x804B / 32843).

    Returns
    -------
    str
        Typically 'bottomUp' or 'topDown'.
    """
    with tifffile.TiffFile(jpk_file) as tif:
        return str(tif.pages[0].tags["32843"].value)


def load_jpk_folder(
    folder_path: Path | str, channel: str, flip_image: bool = True
) -> AFMImageStack:
    """
    Load an AFM video from a folder of individual .jpk image files.

    AFMReader converts "height", "measuredHeight" and "amplitude" channels to nm.

    Parameters
    ----------
    folder_path : Path | str
        Path to folder containing .jpk files.
    channel : str
        Channel to extract.
    flip_image : bool, optional
        Flip each image vertically if True.

    Returns
    -------
    AFMImageStack
        Loaded AFM image stack with metadata and per-frame info.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a directory.")

    jpk_files = sorted(folder.glob("*.jpk"))

    if not jpk_files:
        raise FileNotFoundError(f"No .jpk files found in {folder}.")

    logger.info(f"Found {len(jpk_files)} .jpk files.")

    # Load first image to get shape, pixel size and scan rate
    first_img, first_pixel_size_nm = load_jpk(jpk_files[0], channel)

    height_px, width_px = first_img.shape
    dtype = first_img.dtype
    scan_rate = _extract_scan_rate(jpk_files[0])

    # Preallocate image stack
    num_frames = len(jpk_files)
    image_stack = np.empty((num_frames, height_px, width_px), dtype=dtype)

    # Real per-frame timing (ms precision) with interlace-aware fallback
    timestamps, frame_durations, starts = _frame_timing(jpk_files, height_px)

    if starts[0] is not None:  # dates present -> documented-tag inference
        bidirectional, _ = _bidirectional_from_timing(
            float(np.median(frame_durations)), scan_rate, height_px
        )
    else:  # dates missing -> we already fell back; blob
        bidirectional = _bidirectional_from_blob(jpk_files[0])

    frame_metadata = []

    # Load all images
    for i, fpath in enumerate(jpk_files):
        logger.debug(f"Loading {fpath.name}")
        img, px_size_nm = load_jpk(fpath, channel)
        if img.shape != (height_px, width_px):
            raise ValueError(f"Inconsistent image shape in {fpath}")
        if flip_image:
            img = np.flipud(img)
        image_stack[i] = img

        scan_direction = _scan_direction(fpath)

        # Compose per-frame metadata list
        frame_metadata.append(
            {
                "timestamp": float(timestamps[i]),
                "frame_duration_s": float(frame_durations[i]),
                "start_time": starts[i].isoformat() if starts[i] is not None else None,
                "frame_pixel_size_nm": px_size_nm,
                "bidirectional": bidirectional,
                "line_rate": scan_rate,  # lines per second
                "scan_direction": scan_direction,
            }
        )

    return AFMImageStack(
        data=image_stack,
        pixel_size_nm=first_pixel_size_nm,  # stack-level fallback/reference value
        channel=channel,
        file_path=str(folder),
        frame_metadata=frame_metadata,
    )
