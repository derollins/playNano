"""
Module to decode and load .h5-jpk high speed AFM data files into Python NumPy arrays.

Files containing multiple image frames are read together.
Converts the height data into nm from another metric unit (e.g. m).
"""

import logging
from pathlib import Path

import h5py
import numpy as np

from playnano.afm_stack import AFMImageStack
from playnano.utils.io_utils import (
    HEIGHT_UNITS,
    build_frame_metadata,
    convert_height_units_to_nm,
    decode_hdf5_attr,
    guess_height_data_units,
)

logger = logging.getLogger(__name__)


def _attr_to_bool(attr: bytes | str | bool | int | float) -> bool:
    """
    Convert an attribute to a boolean value.

    Parameters
    ----------
    attr : bytes, str, bool, int, or float
        The attribute to convert.

    Returns
    -------
    bool
        The boolean interpretation of the value.
    """
    if isinstance(attr, (bytes, str)):
        return decode_hdf5_attr(attr).strip().lower() == "true"
    return bool(attr)


def _discover_available_channels(f: h5py.File) -> dict[str, str]:
    """
    Discover all available scan channels in the HDF5 file.

    Parameters
    ----------
    f : h5py.File
        The open HDF5 file.

    Returns
    -------
    dict[str, str]
        Mapping of channel names (e.g. 'height_trace') to their full HDF5 path.

    Notes
    -----
    Assumes a single ``Measurement_*`` group per file (frames are the columns of each
    channel dataset, not separate measurements). If several measurements are present,
    only the first occurrence of each channel key is kept, so downstream only ever
    reads that one measurement — see the guard in :func:`load_h5jpk`.
    """
    channel_map = {}
    for m_key, m_group in f.items():
        if not m_key.startswith("Measurement_"):
            continue
        for c_key in m_group.keys():
            if not c_key.startswith("Channel_"):
                continue

            c_group = m_group[c_key]
            name = c_group.attrs.get("channel.name")
            if name is None:
                continue

            retrace = _attr_to_bool(c_group.attrs.get("retrace", False))
            tr_rt = "retrace" if retrace else "trace"
            full_key = f"{decode_hdf5_attr(name).strip().lower()}_{tr_rt}"
            full_path = f"{m_key}/{c_key}"
            # NOTE: first-wins. With multiple Measurement_* groups this silently
            # ignores all but the first; load_h5jpk asserts there's only one.
            if full_key not in channel_map:
                channel_map[full_key] = full_path
    return channel_map


def _get_channel_info(f: h5py.File, channel: str) -> tuple[h5py.Group, h5py.Group, str]:
    """
    Get the measurement group, channel group, and dataset name for a given channel.

    Parameters
    ----------
    f : h5py.File
        The open HDF5 file.
    channel : str
        The channel name to look for (e.g. 'height_trace').

    Returns
    -------
    tuple[h5py.Group, h5py.Group, str]
        A tuple containing the measurement group, channel group, and dataset name.
    """
    channel_map = _discover_available_channels(f)
    if channel not in channel_map:
        raise ValueError(
            f"Channel '{channel}' not found in file. "
            f"Available channels: {list(channel_map)}"
        )
    channel_path = channel_map[channel]
    channel_group = f[channel_path]
    measurement_key = channel_path.split("/")[0]
    measurement_group = f[measurement_key]

    # Read the actual dataset name from the group (robust to camelCase like
    # 'MeasuredHeight') rather than reconstructing it from the channel string.
    data_keys = [k for k in channel_group if k != "thumbnail"]
    if not data_keys:
        raise ValueError(f"No data dataset found in channel group '{channel_path}'.")

    if len(data_keys) != 1:
        logger.warning(
            "Channel '%s' has %d non-thumbnail datasets %s; using the first.",
            channel,
            len(data_keys),
            data_keys,
        )

    dataset_name = data_keys[0]

    return measurement_group, channel_group, dataset_name


def _get_z_scaling_h5(channel_group: h5py.Group) -> tuple[float, float]:
    """
    Extract the Z scaling multiplier and offset from an HDF5 channel group.

    Parameters
    ----------
    channel_group : h5py.Group
        The HDF5 group corresponding to a specific channel
        (e.g. /Measurement_000/Channel_001).

    Returns
    -------
    tuple[float, float]
        A tuple containing the scaling multiplier and offset.

    Notes
    -----
    Defaults to (1.0, 0.0) if attributes are not present.
    """
    try:
        multiplier = float(channel_group.attrs["net-encoder.scaling.multiplier"])
    except KeyError:
        multiplier = 1.0
        logger.warning(
            "Missing attribute 'net-encoder.scaling.multiplier'. "
            "Defaulting to multiplier = 1.0."
        )

    try:
        offset = float(channel_group.attrs["net-encoder.scaling.offset"])
    except KeyError:
        offset = 0.0
        logger.warning(
            "Missing attribute 'net-encoder.scaling.offset'. "
            "Defaulting to offset = 0.0."
        )

    logger.debug(f"Z value scaling: multiplier = {multiplier}, offset = {offset}")
    return multiplier, offset


def _get_z_unit_h5(channel_group: h5py.Group) -> str | None:
    """
    Extract the Z unit from an HDF5 channel group.

    Parameters
    ----------
    channel_group : h5py.Group
        The HDF5 group corresponding to a specific channel
        (e.g. /Measurement_000/Channel_001).

    Returns
    -------
    string
        The unit of the z data values (e.g. 'm', 'V', 'deg') or None if absent.

    Notes
    -----
    Defaults to None if attribute is not present.
    """
    try:
        raw = channel_group.attrs.get("net-encoder.scaling.unit.unit")
        if raw is None:
            logger.warning(
                "Missing attribute 'net-encoder.scaling.unit.unit'; unit unknown."
            )
            return None
        return decode_hdf5_attr(raw).strip()
    except Exception as e:
        logger.warning(f"Failed to read z unit, returning None: {e}")
        return None


def _get_image_shape(measurement_group: h5py.Group) -> tuple[int, int]:
    """
    Extract pixel width and height from an HDF5 JPK measurement group.

    The pixel dimensions are used to determine image shape.

    Parameters
    ----------
    measurement_group : h5py.Group
        HDF5 group corresponding to a Measurement (e.g. '/Measurement_000').

    Returns
    -------
    tuple[int, int]
        A tuple representing the image shape as (height_px, width_px).

    Raises
    ------
    KeyError
        If required attributes are missing in the measurement group.
    """
    try:
        width_px = measurement_group.attrs[
            "position-pattern.grid.ilength"
        ]  # number of pixels
        height_px = measurement_group.attrs[
            "position-pattern.grid.jlength"
        ]  # number of pixels

        return (height_px, width_px)

    except KeyError as e:
        missing = e.args[0]
        raise KeyError(
            f"Missing required attribute '{missing}' in HDF5 measurement group."
        ) from e


def _jpk_pixel_to_nm_scaling_h5(measurement_group: h5py.Group) -> float:
    """
    Extract pixel-to-nanometre scaling from an HDF5 JPK measurement group.

    This uses the fast scan axis (u/i) and converts the physical scan size to
    nanometres per pixel based on the scan length and pixel count.

    Parameters
    ----------
    measurement_group : h5py.Group
        HDF5 group corresponding to a Measurement (e.g. '/Measurement_000').

    Returns
    -------
    float
        Real-world size of a single pixel in nanometres.

    Raises
    ------
    KeyError
        If required attributes are missing in the measurement group.
    """
    try:
        ulength = measurement_group.attrs[
            "position-pattern.grid.ulength"
        ]  # physical length in meters
        ilength = measurement_group.attrs[
            "position-pattern.grid.ilength"
        ]  # number of pixels

        if ilength == 0:
            raise ValueError("Pixel count (ilength) is zero; cannot compute scaling.")

        return (ulength / ilength) * 1e9

    except KeyError as e:
        missing = e.args[0]
        raise KeyError(
            f"Missing required attribute '{missing}' in HDF5 measurement group."
        ) from e


def _is_bidirectional_h5(measurement_group: h5py.Group) -> bool:
    """Bidirectional (interlaced) scanning, read directly from measurement attrs."""
    return _attr_to_bool(
        measurement_group.attrs.get(
            "environment.fast-imaging.feed-forward-parameters.yaxis.interlace", False
        )
    )


def _get_motion_h5(measurement_group: h5py.Group) -> str | None:
    """Slow-axis frame direction ('topDown' / 'bottomUp')."""
    motion = measurement_group.attrs.get("motion")
    return decode_hdf5_attr(motion).strip() if motion is not None else None


def _frame_times_h5(measurement_group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-frame start/end times as epoch-millisecond int64 arrays.

    From the ``meta-data/start-times`` and ``meta-data/end-times`` datasets, which
    carry one entry per frame (aligned with the channel dataset's frame axis).
    """
    start_time = measurement_group["meta-data/start-times"][:].ravel().astype("int64")
    end_time = measurement_group["meta-data/end-times"][:].ravel().astype("int64")
    return start_time, end_time


def _get_line_rate(measurement_group: h5py.Group) -> float:
    """
    Extract image line rate from an HDF5 JPK measurement group.

    The line rate is the scan speed in terms of lines per second,
    i.e. the speed of imaging in fast scan lines / second.

    Parameters
    ----------
    measurement_group : h5py.Group
        HDF5 group corresponding to a Measurement (e.g. '/Measurement_000').

    Returns
    -------
    float
        The line rate of imaging in lines per second.

    Raises
    ------
    KeyError
        If required attributes are missing in the measurement group.
    """
    try:
        line_rate = measurement_group.attrs[
            "timing-settings.scanRate"
        ]  # scan lines per second

        return line_rate

    except KeyError as e:
        missing = e.args[0]
        raise KeyError(
            f"Missing required attribute '{missing}' in HDF5 measurement group."
        ) from e


def _frame_timing_h5(
    measurement_group: h5py.Group, num_frames: int, height_px: int
) -> tuple[np.ndarray, np.ndarray, list[int | None]]:
    """
    Per-frame timestamps (s from first frame), durations (s), and start epoch-ms.

    Primary: real ``meta-data/start-times``/``end-times`` (ms precision; captures
    real inter-frame jitter and uneven frame lengths).

    Fallback (only if those are missing or mis-sized): scan-rate timing, halved when
    interlaced.
    """
    try:
        start_time, end_time = _frame_times_h5(measurement_group)
        if len(start_time) != num_frames or len(end_time) != num_frames:
            raise ValueError(
                f"timing entries ({len(start_time)}) != frame count ({num_frames})"
            )
        timestamps = (start_time - start_time[0]) / 1000.0  # seconds from first frame
        durations = (end_time - start_time) / 1000.0  # duration in seconds
        start_ms = [int(s) for s in start_time]  # UNIX epoch ms for each frame
    except (KeyError, ValueError) as exc:
        logger.warning(
            "Per-frame start/end times unavailable (%s); "
            "using scan-rate timing for the entire stack.",
            exc,
        )
        try:
            rate = float(_get_line_rate(measurement_group))
        except KeyError:
            rate = None
        if not rate:
            raise ValueError(
                "No per-frame times and no scan rate; cannot time frames."
            ) from exc
        interval = height_px / rate  # slow lines / line rate
        if _is_bidirectional_h5(measurement_group):
            interval /= 2.0  # bidirectional images at ~2x
        timestamps = np.arange(num_frames) * interval
        durations = np.full(num_frames, float(interval))
        start_ms = [None] * num_frames
    return timestamps, durations, start_ms


def _guess_and_standardize_units_to_nm(image_stack: np.ndarray) -> np.ndarray:
    """
    Convert height data to nanometers for metric data.

    Attempts to guess the unit from data range; defaults to 'nm' on failure.

    Parameters
    ----------
    image_stack : np.ndarray
        AFM height data array (2D or 3D).

    Returns
    -------
    np.ndarray
        Height data converted to nanometers.
    """
    try:
        height_unit = guess_height_data_units(image_stack)
        logger.info(f"Guessed that the height unit is {height_unit}")
    except Exception as e:
        height_unit = "nm"
        logger.warning(f"Failed to guess height unit, defaulting to 'nm': {e}")
    image_stack[:] = convert_height_units_to_nm(image_stack, height_unit)
    return image_stack


def apply_z_unit_conversion(
    images: np.ndarray, channel_group: h5py.Group, channel: str = "height_trace"
) -> np.ndarray:
    """Apply z unit conversion to nanometers if needed, or guess if unknown."""
    try:
        z_unit = _get_z_unit_h5(channel_group)
    except Exception as e:
        logger.warning(f"Could not read unit for channel '{channel}': {e}")
        z_unit = None

    if z_unit is not None and z_unit in HEIGHT_UNITS:
        images = convert_height_units_to_nm(images, z_unit)
    elif z_unit is not None and z_unit in ["V", "v", "deg"]:
        pass  # No conversion needed
    else:
        images = _guess_and_standardize_units_to_nm(images)

    return images


def load_h5jpk(
    file_path: Path | str, channel: str, flip_image: bool = True
) -> AFMImageStack:
    """
    Load image stack from a JPK .h5-jpk file, scaled to nanometers.

    Frames are read from the channel's frame axis, z-scaled, converted to nm, and
    given real per-frame timestamps from the file's start/end-time datasets (with an
    interlace-aware scan-rate fallback). Interlace (bidirectional) and motion are read
    from the measurement attributes.

    Parameters
    ----------
    file_path : Path | str
        Path to the .h5-jpk file.
    channel : str
        Channel to extract.
    flip_image : bool, optional
        Flip each image vertically if True.

    Returns
    -------
    AFMImageStack
        Loaded AFM image stack with metadata and per-frame info.

    Notes
    -----
    In .h5-jpk files the pixel size, interlace mode and motion are defined at the
    measurement level and are constant across all frames, so those metadata values are
    identical for every frame. Timestamps and frame durations are per-frame.

    This differs from .jpk and .spm folder-based data, where pixel size may vary
    between frames and is stored per-frame.
    """
    file_path = Path(file_path)

    with h5py.File(file_path, "r") as f:
        # This reader assumes ONE measurement per file, with frames stored as the
        # columns of each channel dataset. Multi-measurement files (frames split
        # across Measurement_000/001/...) are not yet supported and so only their
        # first measurement is read.
        measurements = [k for k in f if k.startswith("Measurement_")]
        if len(measurements) > 1:
            logger.warning(
                "%s: %d measurements found; reading only '%s', "
                "frames in later measurements will be ignored.",
                file_path.name,
                len(measurements),
                measurements[0],
            )

        measurement_group, channel_group, dataset_name = _get_channel_info(f, channel)

        # Load raw image data: shape (pixels, frames)
        raw_images = channel_group[dataset_name][:]

        # Apply Z scaling and offset
        scaling, offset = _get_z_scaling_h5(channel_group)
        images = (raw_images * scaling) + offset

        # Convert to nm if necessary
        images = apply_z_unit_conversion(images, channel_group, channel)

        # Get image shape and number of frames
        height_px, width_px = _get_image_shape(measurement_group)
        num_frames = images.shape[1]

        # Reshape each column vector (height, width) to get (num_frames, height, width)
        image_stack = np.empty((num_frames, height_px, width_px), dtype=images.dtype)
        for i in range(num_frames):
            frame = images[:, i].reshape((height_px, width_px))
            if flip_image:
                frame = np.flipud(frame)
            image_stack[i] = frame

        # --- timing + scan mode (measurement-level, constant across frames) ---
        timestamps, frame_durations, start_ms = _frame_timing_h5(
            measurement_group, num_frames, height_px
        )
        bidirectional = _is_bidirectional_h5(measurement_group)
        scan_direction = _get_motion_h5(measurement_group)
        pixel_size_nm = _jpk_pixel_to_nm_scaling_h5(measurement_group)
        try:
            line_rate = _get_line_rate(measurement_group)
        except KeyError:
            line_rate = None

        # Compose per-frame metadata list
        frame_metadata = [
            build_frame_metadata(
                timestamp=timestamps[i],
                frame_pixel_size_nm=pixel_size_nm,
                line_rate=line_rate,
                frame_duration_s=frame_durations[i],
                start_epoch_ms=start_ms[i],
                scan_direction=scan_direction,
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
        afm.acquisition["bidirectional"] = (
            None if bidirectional is None else bool(bidirectional)
        )
    return afm
