"""
Module to decode and load .aris high speed AFM data files into a AFMImageStack object.

Files containing multiple image frames are read together. Since the frames in .aris
files can be different sizes the pixel scaling value for the first frame is held as
an AFMImageStack object (pixel_size_nm) while the values for indevidual frames are
stored per frame in the frame_metadat list of dictionaries under the key,
``frame_pixel_size_nm``.
"""

import logging
from pathlib import Path

import h5py
import numpy as np

from playnano.afm_stack import AFMImageStack
from playnano.utils.io_utils import decode_hdf5_attr

logger = logging.getLogger(__name__)


def _get_channel_names(info: h5py.Group) -> list[str]:
    """
    Extract and decode channel names from an HDF5 group's attributes.

    Parameters
    ----------
    info : h5py.Group
        The HDF5 group containing the 'ChannelNames' attribute.

    Returns
    -------
    list[str]
        A list of decoded channel name strings.
    """
    return [decode_hdf5_attr(name) for name in info.attrs["ChannelNames"]]


def _aris_initial_pixel_to_nm_scaling_h5(info: h5py.Group) -> float:
    """
    Extract pixel-to-nanometre scaling from an ARIS HDF5 DataSetInfo Global group.

    This uses the fast scan axis (FastScanSize) and converts the physical scan size to
    nanometres per pixel based on the numper of pixels per line (ScanPoints).

    Parameters
    ----------
    info : h5py.Group
        HDF5 group containing the metadata associated with a DataSet (DataSetInfo).

    Returns
    -------
    float
        Real-world size of a single pixel in nanometres.

    Raises
    ------
    KeyError
        If required attributes are missing in the DataSetInfo group.
    """
    try:
        scan_width = info["Global/Parameters/Scan"].attrs[
            "FastScanSize"
        ]  # physical length in meters
        pixel_scan_width = info["Global/Parameters/Scan"].attrs[
            "ScanPoints"
        ]  # number of pixels

        if pixel_scan_width == 0:
            raise ValueError(
                "Pixel count (ScanPoints) is zero; cannot compute scaling."
            )

        return (scan_width / pixel_scan_width) * 1e9

    except KeyError as e:
        missing = e.args[0]
        raise KeyError(
            f"Missing required attribute '{missing}' in HDF5 measurement group."
        ) from e


def load_aris(
    file_path: Path | str,
    channel: str,
) -> AFMImageStack:
    """
    Load image stack from a Asylum Research .aris file, scaled to nanometers.

    The images are loaded, reshaped into frames, and have timestamps generated.

    Parameters
    ----------
    file_path : Path | str
        Path to the .h5-jpk file.
    channel : str
        Channel to extract.

    Returns
    -------
    AFMImageStack
        Loaded AFM image stack with metadata and per-frame info.
    """
    file_path = Path(file_path)
    with h5py.File(file_path, "r") as file:
        data = file[
            "/DataSet/Resolution 0"
        ]  # where the image data is stored per frame per channel
        info = file["/DataSetInfo"]  # where image metadata, channel names etc.

        file_channels = _get_channel_names(info)

        if channel in file_channels:
            selected_ch = channel
        else:
            raise ValueError(f"Channel '{channel}' is not available.")

        data_keys = list(data.keys())
        sorted_keys = sorted(data_keys, key=lambda k: int(k.split()[1]))
        frames = []
        pixel_sizes_nm = []

        initial_pixel_size_nm = _aris_initial_pixel_to_nm_scaling_h5(info)

        initial_scan_pixel_width = info["Global/Parameters/Scan"].attrs["ScanPoints"]

        for frame_key in sorted_keys:
            frames.append(data[frame_key][selected_ch]["Image"][:])
            # If the scan size changes it is recorded in the per frame record in the
            # metadata group
            try:
                new_scan_width = info["Frames"][frame_key]["Parameters"]["Scan"].attrs[
                    "FastScanSize"
                ]
            except (KeyError, AttributeError):
                new_scan_width = None

            if new_scan_width is None:
                pixel_sizes_nm.append(initial_pixel_size_nm)
            else:
                try:
                    new_scan_points = info["Frames"][frame_key]["Parameters"][
                        "Scan"
                    ].attrs["ScanPoints"]
                except (KeyError, AttributeError):
                    new_scan_points = None
                if new_scan_points is None:
                    new_pixel_size = (new_scan_width / initial_scan_pixel_width) * 1e9
                else:
                    new_pixel_size = (new_scan_width / new_scan_points) * 1e9

                pixel_sizes_nm.append(new_pixel_size)

        image_stack = np.stack(frames)

        # Read timestamps and line_rate
        timestamps = info["Series"]["Time"][:]
        if len(timestamps) != len(sorted_keys):
            raise ValueError(
                f"Timestamp count ({len(timestamps)}) does not match frame count ({len(sorted_keys)})."  # noqa
            )
        line_rate = info["Global/Parameters/Scan"].attrs["ScanRate"]

        # Compose per-frame metadata list
        frame_metadata = []
        for frame in range(len(sorted_keys)):
            frame_metadata.append(
                {
                    "timestamp": float(timestamps[frame]),
                    "frame_pixel_size_nm": float(pixel_sizes_nm[frame]),
                    "line_rate": float(line_rate),
                }
            )

        return AFMImageStack(
            data=image_stack,
            pixel_size_nm=initial_pixel_size_nm,
            channel=channel,
            file_path=str(file_path),
            frame_metadata=frame_metadata,
        )
