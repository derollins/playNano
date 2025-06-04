"""
Module to decode and load .asd high speed AFM data files into Python NumPy arrays.

Files containing multiple image frames are read together.
"""

import logging
from pathlib import Path

import numpy as np
from AFMReader.asd import load_asd

from playNano.stack.afm_stack import AFMImageStack

logger = logging.getLogger(__name__)


def load_asd_file(file_path: Path | str, channel: str) -> AFMImageStack:
    """
    Load image stack from an .asd file.

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
