"""
Module to decode and load .h5-jpk high speed AFM data files into Python NumPy arrays.

Files containing multiple image frames are read together.
Converts the height data into nm from another metric unit (e.g. m).
"""

import logging
from pathlib import Path

import h5py
import numpy as np

from playNano.afm_stack import AFMImageStack
from playNano.utils.io_utils import (
    convert_height_units_to_nm,
    guess_height_data_units,
    height_units,
)

logger = logging.getLogger(__name__)


def _decode_attr(attr: bytes | str) -> str:
    """
    Decode an attribute that may be bytes or a string.

    Parameters
    ----------
    attr : bytes or str
        The attribute to decode.

    Returns
    -------
    str
        The decoded string.
    """
    if isinstance(attr, bytes):
        return attr.decode("utf-8")
    return str(attr)


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
        return _decode_attr(attr).strip().lower() == "true"
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
            full_key = f"{_decode_attr(name).strip().lower()}_{tr_rt}"
            full_path = f"{m_key}/{c_key}"
            if full_key not in channel_map:
                channel_map[full_key] = full_path
    return channel_map


def _get_channel_info(f: h5py.File, channel: str):
    """
    Retrieve channel-related HDF5 groups and dataset name.

    Parameters
    ----------
    f : h5py.File
        The open HDF5 file object.
    channel : str
        The name of the channel to retrieve.

    Returns
    -------
    tuple[h5py.Group, h5py.Group, str]
        The measurement group, channel group, and dataset name.

    Raises
    ------
    ValueError
        If the channel is not found.
    """
    channel_map = _discover_available_channels(f)
    if channel not in channel_map:
        raise ValueError(
            f"Channel '{channel}' not found in file."
            f"Available channels: {list(channel_map)}"
        )
    channel_path = channel_map[channel]
    channel_group = f[channel_path]
    measurement_key = channel_path.split("/")[0]
    measurement_group = f[measurement_key]
    dataset_name = channel.split("_")[0].capitalize()
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
    multiplier = float(channel_group.attrs.get("net-encoder.scaling.multiplier", 1.0))
    offset = float(channel_group.attrs.get("net-encoder.scaling.offset", 0.0))

    return multiplier, offset


def _get_z_unit_h5(channel_group: h5py.Group) -> str:
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
        The unit of the z data values.

    Notes
    -----
    Defaults to None if attribute is not present.
    """
    try:
        z_unit = str(channel_group.attrs.get("net-encoder.scaling.unit.unit", 1.0))
    except Exception as e:
        z_unit = None
        logger.warning(f"Failed to read unit, returning None: {e}")
    return z_unit


def _get_image_shape(measurement_group: h5py.Group) -> float:
    """
    Extract pixel width and hight from an HDF5 JPK measurement group.

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
    None
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
    images: np.ndarray, channel_group: h5py.Group, channel: str = "TP"
) -> np.ndarray:
    """Apply z unit conversion to nanometers if needed, or guess if unknown."""
    try:
        z_unit = _get_z_unit_h5(channel_group)
    except Exception as e:
        logging.warning(f"Could not read unit for channel '{channel}': {e}")
        z_unit = None

    if z_unit is not None and z_unit in height_units:
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

    The images are loaded, reshaped into frames, and have timestamps generated.

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
    """
    file_path = Path(file_path)

    with h5py.File(file_path, "r") as f:
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

        # Generate timestamps per frame from line_rate
        line_rate = _get_line_rate(measurement_group)
        frame_interval = (
            height_px / line_rate
        )  # seconds per frame (height lines / lines per second)
        timestamps = np.arange(num_frames) * frame_interval

        # Compose per-frame metadata list
        frame_metadata = []
        for ts in timestamps:
            frame_metadata.append({"timestamp": ts, "line_rate": line_rate})

        return AFMImageStack(
            data=image_stack,
            pixel_size_nm=_jpk_pixel_to_nm_scaling_h5(measurement_group),
            channel=channel,
            file_path=str(file_path),
            frame_metadata=frame_metadata,
        )
