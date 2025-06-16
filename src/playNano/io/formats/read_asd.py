"""
Module to decode and load .asd high speed AFM data files into Python NumPy arrays.

Files containing multiple image frames are read together.
Converts the height data into nm from another metric unit (e.g. m).
"""

import logging
from pathlib import Path

import numpy as np
from AFMReader.asd import load_asd

from playNano.afm_stack import AFMImageStack
from playNano.utils.io_utils import convert_height_units_to_nm, guess_height_data_units

logger = logging.getLogger(__name__)


def _standardize_units_to_nm(image_stack: np.ndarray, channel: str) -> np.ndarray:
    """
    Convert height data to nanometers if the channel is topography ("TP").

    Attempts to guess the unit from data range; defaults to 'nm' on failure.

    Parameters
    ----------
    image_stack : np.ndarray
        AFM height data array (2D or 3D).
    channel : str
        Channel name; must be "TP" to trigger conversion.

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

    if channel == "TP":
        image_stack[:] = convert_height_units_to_nm(image_stack, height_unit)
    return image_stack


def load_asd_file(file_path: Path | str, channel: str) -> AFMImageStack:
    """
    Load image stack from an .asd file scaled to nanometers.

    Parameters
    ----------
    file_path : Path | str
        Path to the .asd file.
    channel : str
        Channel to extract.

    Returns
    -------
    AFMImageStack
        Loaded AFM image stack with metadata and per-frame info.
    """
    file_path = Path(file_path)

    # Read .asd data and header
    image_stack, pixel_size_nm, asd_metadata = load_asd(file_path, channel)

    image_stack = _standardize_units_to_nm(image_stack, channel)

    frame_time = asd_metadata["frame_time"]
    lines = asd_metadata["y_pixels"]
    num_frames = asd_metadata["num_frames"]

    line_rate = lines / (frame_time / 1000)  # lines per second
    frame_interval = lines / line_rate  # seconds per frame (lines / lines per second)
    timestamps = np.arange(num_frames) * frame_interval

    # Compose per-frame metadata list
    frame_metadata = []
    for ts in timestamps:
        frame_metadata.append({"timestamp": ts, "line_rate": line_rate})

    return AFMImageStack(
        data=image_stack,
        pixel_size_nm=pixel_size_nm,
        channel=channel,
        file_path=str(file_path),
        frame_metadata=frame_metadata,
    )
