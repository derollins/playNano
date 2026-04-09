import re
from datetime import datetime
from pathlib import Path
import logging

import numpy as np
from nanosurf.lib.util import nhf_reader

from playnano.afm_stack import AFMImageStack
from playnano.utils.io_utils import convert_height_units_to_nm

logger = logging.getLogger(__name__)


def _discover_available_channels(
    measurement: nhf_reader.NHFMeasurement,
) -> list[str]:
    """
    Discover all available scan channels in the NHF measurement.

    Returns channel names with _trace / _retrace suffixes.
    """
    channels = []

    forward = measurement.segment.get("Forward")
    backward = measurement.segment.get("Backward")

    forward_ch = set(forward.channel.keys()) if forward else set()
    backward_ch = set(backward.channel.keys()) if backward else set()

    all_channels = sorted(forward_ch | backward_ch)

    for ch in all_channels:
        if ch in forward_ch:
            channels.append(f"{ch}_trace")
        if ch in backward_ch:
            channels.append(f"{ch}_retrace")

    return channels


def _find_pixel_size(measurement):
    """Extract pixel size in nanometers from the NHF measurement metadata."""
    pixel_width = measurement.dataset_size_x
    meter_width = measurement.dataset_range_x
    return (meter_width / pixel_width) * 1e9


def get_nhf_time(file_path: Path) -> datetime:
    """Extract the timestamp from the "created" attribute of a .nhf file."""
    with nhf_reader.NHFFileReader(file_path) as nhf_file:
        key = next(k for k in nhf_file.measurement if k.startswith("Image"))
        measurement = nhf_file.measurement[key]

        t = measurement.attribute["created"]

        # unwrap numpy scalars
        if isinstance(t, np.ndarray):
            t = t.item()

        # decode bytes
        if isinstance(t, (bytes, np.bytes_)):
            t = t.decode("utf-8")

        # handle ISO 8601 with Z (UTC)
        if isinstance(t, str) and t.endswith("Z"):
            return datetime.fromisoformat(t.replace("Z", "+00:00"))

        # fallback
        return datetime.fromisoformat(t)


def _timestamps_to_elapsed_seconds(times: list[datetime]) -> np.ndarray:
    """
    Convert absolute datetimes to elapsed seconds starting at 0.
    """
    t0 = times[0]
    return np.array([(t - t0).total_seconds() for t in times], dtype=float)


def get_image_number(file_path: Path) -> int:
    """Extract the image number from the NHF measurement_name attribute."""
    with nhf_reader.NHFFileReader(file_path) as nhf_file:
        if len(nhf_file.measurement) != 1:
            raise ValueError(
                f"{file_path.name} contains {len(nhf_file.measurement)} measurements"
            )

        measurement = next(iter(nhf_file.measurement.values()))
        name = measurement.attribute.get("measurement_name")

        if not isinstance(name, str):
            raise ValueError(f"{file_path.name} missing measurement_name")

        match = re.search(r"\d+", name)
        if match is None:
            raise ValueError(f"{file_path.name} has invalid measurement_name: {name!r}")

        return int(match.group())


def load_nhf(file_path: Path, channel: str):
    """
    Load a single .nhf file and extract the specified channel as a 2D numpy array.

    Parameters
    ----------
    file_path : Path
        Path to the .nhf file to load.
    channel : str
        Name of the channel to extract (e.g. 'height_trace').

    Returns
    -------
    tuple[np.ndarray, float, float]
        A tuple containing the image data as a 2D numpy array, the pixel size in nm,
        and the scan line rate.

    Raises
    ------
    ValueError
        If the specified channel is not found in the file.

    Notes
    -----
    This function uses the `nanosurf` library to read .nhf files. It looks for the
    specified channel in the 'Forward' and 'Backward' segments of the measurement
    depending if the channel name is suffixed with '_trace' or '_retrace' (same
    behaviour as the JPK reader). It also extracts the pixel size from the measurement
    metadata.
    """
    with nhf_reader.NHFFileReader(file_path) as nhf_file:
        key = next(k for k in nhf_file.measurement.keys() if k.startswith("Image"))

        measurement = nhf_file.measurement[key]

        channel_names = _discover_available_channels(measurement)

        if channel not in channel_names:
            raise ValueError(
                f"Channel '{channel}' not found in {file_path.name}. Available channels: {list(channel_names)}"  # noqa E501
            )

        if measurement.measurement_type == nhf_reader.NHFMeasurementType.Image:
            if channel.endswith("_retrace"):
                chosen_segment = measurement.segment["Backward"]
            else:
                chosen_segment = measurement.segment["Forward"]

            channel_base = channel.rsplit("_", 1)[0]  # Remove _trace/_retrace suffix

            read_channel = chosen_segment.read_channel(channel_base, as_matrix=True)
            img = read_channel.dataset
            unit = read_channel.unit

            img = convert_height_units_to_nm(img, unit)

        px_size_nm = _find_pixel_size(measurement)
        line_rate = measurement.attribute["scan_line_rate"]
        return img, px_size_nm, line_rate


def load_nhf_folder(
    folder_path: Path | str,
    channel: str,
) -> AFMImageStack:
    """
    Load an AFM video from a folder of individual .nhf image files.

    Parameters
    ----------
    folder_path : Path | str
        Path to folder containing .nhf files.
    channel : str
        Channel to extract.

    Returns
    -------
    AFMImageStack
        Loaded AFM image stack with metadata and per-frame info.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a directory")

    nhf_files = sorted(folder.glob("*.nhf"), key=get_image_number)

    if not nhf_files:
        raise FileNotFoundError(f"No .nhf files in {folder}")

    image_stack = []
    frame_metadata = []
    times = []
    for fpath in nhf_files:
        img, px_size_nm, line_rate = load_nhf(fpath, channel)

        if line_rate is None:
            raise ValueError("Missing data: line_rate=None")

        t = get_nhf_time(fpath)

        image_stack.append(img)
        times.append(t)
        frame_metadata.append(
            {
                "frame_pixel_size_nm": px_size_nm,
                "line_rate": line_rate,
            }
        )

    image_stack = np.stack(image_stack)

    time_seconds = _timestamps_to_elapsed_seconds(times)

    dt = np.diff(time_seconds)
    if not np.all(dt >= 0):
        raise ValueError("NHF timestamps are not monotonic (time goes backwards).")
    if np.any(dt == 0):
        logger.warning(
            "NHF timestamps contain repeated values (zero time difference between frames)."  # noqa E501
        )

    for i, frame in enumerate(frame_metadata):
        frame["timestamp"] = time_seconds[i]

    return AFMImageStack(
        data=image_stack,
        pixel_size_nm=frame_metadata[0]["frame_pixel_size_nm"],
        channel=channel,
        file_path=str(folder),
        frame_metadata=frame_metadata,
    )
