"""
Load Asylum Research ``.aris`` high-speed AFM files as an AFMImageStack.

ARIS files are HDF5 files. Frames can differ in scan size, so each frame's
pixel size is stored in its ``frame_metadata`` entry as
``frame_pixel_size_nm``; the stack's ``pixel_size_nm`` is that of the first
frame.

.. warning::
   Heights are returned in the units stored in the file, which is metres for
   height channels, not nm as the other readers return.
"""

import logging
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from playnano.afm_stack import AFMImageStack
from playnano.utils.io_utils import build_frame_metadata, decode_hdf5_attr

logger = logging.getLogger(__name__)


def _parse_aris_start(value: str) -> datetime | None:
    """
    Parse ARIS StartTime ('YYYY-MM-DD HH:MM:SS.ffffff... +HHMM') to aware datetime.

    ARIS stamps go to sub-microseconds; Python's %f is 6 digits, so trim any excess
    (e.g. '.638777400' -> '.638777').
    """
    if value is None:
        return None
    s = str(value).strip()
    # trim fractional seconds to 6 digits so %f parses
    if "." in s:
        head, tail = s.split(".", 1)
        frac, _, rest = tail.partition(" ")
        s = f"{head}.{frac[:6]} {rest}" if rest else f"{head}.{frac[:6]}"
    for fmt in ("%Y-%m-%d %H:%M:%S.%f %z", "%Y-%m-%d %H:%M:%S %z"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _get_channel_names(info: h5py.Group) -> list[str]:
    """
    Extract and decode channel names from the ARIS HDF5 DataSetInfo group.

    Parameters
    ----------
    info : h5py.Group
        The '/DataSetInfo' group of the ARIS file.

    Returns
    -------
    list of str
        Decoded channel names.
    """
    return [decode_hdf5_attr(name) for name in info.attrs["ChannelNames"]]


def _aris_global_pixel_to_nm_scaling_h5(info: h5py.Group) -> float:
    """
    Extract pixel-to-nanometre scaling from an ARIS HDF5 DataSetInfo Global group.

    This uses the fast scan axis (FastScanSize) in meters and converts the physical
    scan size to nanometres per pixel based on the number of pixels per line
    (ScanPoints).

    FastScanSize (in meters) / ScanPoints (number of pixels) * 1e9

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
        If required attributes are missing.
    ValueError
        If ScanPoints is zero.
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


def _get_sorted_frame_keys(data: h5py.Group) -> list[str]:
    """
    Return frame keys sorted numerically.

    ARIS frame datasets use keys like 'Frame 0', 'Frame 1', 'Frame 2'... .

    Parameters
    ----------
    data : h5py.Group
        The '/DataSet/Resolution 0' group containing frame datasets.

    Returns
    -------
    list of str
        Sorted frame keys.
    """
    data_keys = list(data.keys())
    return sorted(data_keys, key=lambda k: int(k.split()[1]))


def load_frames_and_scaling(
    data: h5py.Group,
    info: h5py.Group,
    selected_ch: str,
) -> tuple[np.ndarray, list[float]]:
    """
    Read every frame of one channel and work out each frame's pixel size.

    Parameters
    ----------
    data : h5py.Group
        The ``/DataSet/Resolution 0`` group, holding one group per frame.
    info : h5py.Group
        The ``/DataSetInfo`` group, holding the scan parameters and any changes
        made to them during the recording.
    selected_ch : str
        Channel to read.

    Returns
    -------
    image_stack : numpy.ndarray
        Frames in time order, shape (n_frames, height, width), in the units
        stored in the file.
    pixel_sizes_nm : list of float
        Pixel size of each frame, in nm. Frames without their own scan size use
        the global one.
    """
    sorted_keys = _get_sorted_frame_keys(data)

    initial_pixel_size_nm = _aris_global_pixel_to_nm_scaling_h5(info)
    initial_scan_pixel_width = info["Global/Parameters/Scan"].attrs["ScanPoints"]

    frames = []
    pixel_sizes_nm = []
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
                new_scan_points = info["Frames"][frame_key]["Parameters"]["Scan"].attrs[
                    "ScanPoints"
                ]
            except (KeyError, AttributeError):
                new_scan_points = None
            if new_scan_points is None:
                new_pixel_size = (new_scan_width / initial_scan_pixel_width) * 1e9
            else:
                new_pixel_size = (new_scan_width / new_scan_points) * 1e9

            pixel_sizes_nm.append(new_pixel_size)

    image_stack = np.stack(frames)

    return image_stack, pixel_sizes_nm


def load_aris(
    file_path: Path | str,
    channel: str,
) -> AFMImageStack:
    """
    Load an Asylum Research ``.aris`` file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        The ``.aris`` file.
    channel : str
        Channel to load, for example ``"HeightTrace"``, ``"HeightRetrace"``,
        ``"AmplitudeRetrace"`` or ``"PhaseRetrace"``; the names depend on the
        file.

    Returns
    -------
    AFMImageStack
        The stack. Heights are in the units stored in the file (metres for
        height channels), not nm. Each ``frame_metadata`` entry holds
        ``timestamp`` (s from the first frame), ``frame_pixel_size_nm``,
        ``line_rate`` (the global scan rate, in lines per second) and
        ``start_epoch_ms`` (if the file records its start time).

    Raises
    ------
    ValueError
        If the channel is not in the file, or the number of frame times does
        not match the number of frames.
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

        image_stack, pixel_sizes_nm = load_frames_and_scaling(data, info, selected_ch)

        # Read timestamps and line_rate
        timestamps = info["Series"]["Time"][:]
        if len(timestamps) != image_stack.shape[0]:
            raise ValueError(
                f"Timestamp count ({len(timestamps)}) does not match frame count ({image_stack.shape[0]})."  # noqa
            )
        line_rate = info["Global/Parameters/Scan"].attrs[
            "ScanRate"
        ]  # TODO: honour per-frame ScanRate overrides (it can change between frames)

        # Absolute time (aware, sub-ms): base epoch + per-frame Series/Time offset
        start_dt = _parse_aris_start(
            decode_hdf5_attr(info["Global"].attrs.get("StartTime"))
        )
        base_ms = int(start_dt.timestamp() * 1000) if start_dt is not None else None

        def _epoch_ms(i):
            return (
                None
                if base_ms is None
                else base_ms + int(round(float(timestamps[i]) * 1000))
            )

        # Compose per-frame metadata list
        frame_metadata = []
        for frame in range(image_stack.shape[0]):
            frame_metadata.append(
                build_frame_metadata(
                    timestamp=float(timestamps[frame])
                    - float(timestamps[0]),  # seconds from frame 0
                    frame_pixel_size_nm=pixel_sizes_nm[frame],
                    line_rate=float(line_rate),
                    start_epoch_ms=_epoch_ms(frame),
                )
            )

        return AFMImageStack(
            data=image_stack,
            pixel_size_nm=pixel_sizes_nm[0],
            channel=channel,
            file_path=str(file_path),
            frame_metadata=frame_metadata,
        )
