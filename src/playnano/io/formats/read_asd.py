"""
Load ``.asd`` high-speed AFM files as an AFMImageStack.

Files are read with AFMReader. Topography (channel ``"TP"``) is converted to nm
from a unit guessed from the range of the data; other channels are left as
read.

Timing: ``.asd`` headers record one frame time for the whole file, so
``timestamp`` steps evenly and ``frame_duration_s`` is the same for every
frame. The header date is the end of the recording (when the file was
written), so ``start_epoch_ms`` is that date minus
``n_frames * frame_duration_s``.

Scan direction: the header does not record the slow-axis direction, so
``scan_direction`` is None. The header's ``scan_direction`` integer seems to
encode the acquisition mode and the fast-axis direction of channel 1, but it is
not documented. Its raw value, the file version and the channel names are kept
in ``stack.acquisition`` so they can be decoded later.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from AFMReader.asd import load_asd

from playnano.afm_stack import AFMImageStack
from playnano.io.formats._afmreader import unpack_afmreader_result
from playnano.utils.io_utils import (
    build_frame_metadata,
    convert_height_units_to_nm,
    guess_height_data_units,
)

logger = logging.getLogger(__name__)


def _standardize_units_to_nm(image_stack: np.ndarray, channel: str) -> np.ndarray:
    """
    Convert topography data to nanometres, guessing the source unit from data range.

    Only converts when ``channel`` is 'TP' (topography); other channels
    (error, phase) are returned unchanged. Guessing failure falls back to 'nm'.

    Parameters
    ----------
    image_stack : np.ndarray
        AFM height data array (2D or 3D).
    channel : str
        Channel name; must be 'TP' to trigger conversion.

    Returns
    -------
    np.ndarray
        The (possibly rescaled) image stack.
    """
    try:
        height_unit = guess_height_data_units(image_stack)
        logger.info(f"Guessed that the height unit is {height_unit}")
    except Exception as e:
        height_unit = "nm"
        logger.warning(f"Failed to guess height unit, defaulting to 'nm': {e}")

    if channel == "TP":
        image_stack[:] = convert_height_units_to_nm(image_stack, height_unit)
    return image_stack


def _sniff_file_version(asd_metadata: dict) -> int | None:
    """
    Infer the asd file version from AFMReader's header dict.

    AFMReader's ``load_asd`` doesn't return the version, so infer it from
    keys that differ between the version-0/1/2 parsers. Best-effort; returns
    ``None`` if the header shape doesn't match any known version.
    """
    if "number_of_frames" in asd_metadata:  # v2 adds this extra key at the end
        return 2
    if "x_rounding_degree" in asd_metadata:  # v1 splits rounding into x/y
        return 1
    if "rounding_degree" in asd_metadata:  # v0 has a single rounding value
        return 0
    return None


def _start_epoch_ms(asd_metadata: dict, total_duration_s: float) -> int | None:
    """
    Derive frame 0's start time as Unix epoch ms.

    The asd header's date fields record when the file was written — i.e. when
    the LAST frame finished — not when acquisition started. Backdate by
    ``total_duration_s`` to place frame 0. Naive local time assumed UTC, with
    second precision (see readme).
    """
    try:
        end_dt = datetime(
            asd_metadata["year"],
            asd_metadata["month"],
            asd_metadata["day"],
            asd_metadata["hour"],
            asd_metadata["minute"],
            asd_metadata["second"],
            tzinfo=timezone.utc,
        )
    except (KeyError, ValueError, TypeError):
        return None
    start_dt = end_dt - timedelta(seconds=total_duration_s)
    return int(start_dt.timestamp() * 1000)


def load_asd_file(file_path: Path | str, channel: str) -> AFMImageStack:
    """
    Load an ``.asd`` file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        The ``.asd`` file.
    channel : str
        Channel to load: ``"TP"`` (topography), ``"ER"`` (error) or ``"PH"``
        (phase).

    Returns
    -------
    AFMImageStack
        The stack, shape (n_frames, height, width). Topography is in nm.

        Each ``frame_metadata`` entry holds ``timestamp`` (s from the first
        frame), ``frame_pixel_size_nm``, ``line_rate`` (lines per second),
        ``frame_duration_s`` and ``start_epoch_ms``; ``scan_direction`` is None.

        ``stack.acquisition`` holds ``asd_file_version``,
        ``asd_scan_direction_raw`` (the undecoded header value) and
        ``asd_channels`` (the two channel slots of the header). ``bidirectional``
        is not set.
    """
    file_path = Path(file_path)

    # Read .asd data and header
    image_stack, pixel_size_nm, asd_metadata = unpack_afmreader_result(
        load_asd(file_path, channel)
    )
    image_stack = _standardize_units_to_nm(image_stack, channel)

    # --- Timing ---
    # Variable scan sizes / pixel sizes aren't allowed within a single .asd
    # file, and frame_time is a single header value, so all per-frame values
    # here are constants of the stack.
    frame_duration_s = float(asd_metadata["frame_time"]) / 1000.0  # ms -> s
    lines = int(asd_metadata["y_pixels"])
    num_frames = int(asd_metadata["num_frames"])
    line_rate = lines / frame_duration_s  # lines per second
    timestamps = np.arange(num_frames) * frame_duration_s
    base_epoch_ms = _start_epoch_ms(asd_metadata, num_frames * frame_duration_s)

    # --- Per-frame metadata ---
    frame_metadata = [
        build_frame_metadata(
            timestamp=float(timestamps[i]),
            frame_pixel_size_nm=pixel_size_nm,
            line_rate=line_rate,
            frame_duration_s=frame_duration_s,
            start_epoch_ms=(
                None
                if base_epoch_ms is None
                else base_epoch_ms + int(round(i * frame_duration_s * 1000))
            ),
            scan_direction=None,  # not encoded in asd
        )
        for i in range(num_frames)
    ]

    afm = AFMImageStack(
        data=image_stack,
        pixel_size_nm=pixel_size_nm,
        channel=channel,
        file_path=str(file_path),
        frame_metadata=frame_metadata,
    )

    # --- Stack-level acquisition provenance ---
    # AFMReader returns '' for an unused channel slot; normalise to None.
    ch1 = asd_metadata.get("channel1") or None
    ch2 = asd_metadata.get("channel2") or None
    afm.acquisition["asd_file_version"] = _sniff_file_version(asd_metadata)
    afm.acquisition["asd_scan_direction_raw"] = int(asd_metadata["scan_direction"])
    # AFMReader only allows the opening of one channel if they are both topography,
    # but the header has two slots; record both for provenance.
    afm.acquisition["asd_channels"] = (ch1, ch2)
    return afm
