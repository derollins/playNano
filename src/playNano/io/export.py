"""
Tools for exporting AFM image stacks in multiple formats.

Provides functions to export AFM stacks with metadata as OME-TIFF, NPZ, or HDF5
bundles. Handles path validation, metadata embedding, and file structure creation.

Dependencies
------------
- numpy
- tifffile
- h5py
- json
- pathlib
"""

import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import tifffile

from playNano.afm_stack import AFMImageStack
from playNano.utils.io_utils import prepare_output_directory, sanitize_output_name

logger = logging.getLogger(__name__)


def check_path_is_path(path):
    """
    Ensure the input is returned as a ``pathlib.Path``.

    Converts strings to ``Path`` objects. Raises ``TypeError`` for unsupported types.

    Parameters
    ----------
    path : str or Path
        The input path to validate or convert.

    Returns
    -------
    Path
        A ``pathlib.Path`` object representing the input path.

    Raises
    ------
    TypeError
        If the input is not a ``str`` or ``Path``.
    """
    if isinstance(path, str):
        logger.debug(f"Converting {path} to Path object.")
        path = Path(path)
    elif isinstance(path, Path):
        pass
    else:
        raise TypeError(f"{path} is not a string or a Path.")
    return path


def save_ome_tiff_stack(
    path: Path,
    stack: np.ndarray,
    pixel_size_nm: float,
    timestamps: list[float],
    channel: str = "height_trace",
) -> None:
    """
    Save a 3D AFM image stack as an OME-TIFF file with metadata.

    Parameters
    ----------
    path : Path
        Path to the output ".ome.tif" file.
    stack : np.ndarray
        AFM image stack of shape (n_frames, H, W) and dtype float or uint.
    pixel_size_nm : float
        Physical pixel size in nanometers.
    timestamps : list of float
        List of timestamps corresponding to each frame.
    channel : str, optional
        Channel name to embed in OME metadata. Default is "height_trace".

    Returns
    -------
    None

    Notes
    -----
    - Pixel size is stored in micrometers.
    - Time increment and time points are embedded in OME metadata.
    - Stack is reshaped to 5D TCZYX format as required by OME-TIFF.
    """
    path = check_path_is_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # tifffile’s OME writer expects a 5D array in TCZYX or TZYX format.
    # We have a purely 2D grayscale stack over time (no channels or Z),
    # so reshape to (T, C=1, Z=1, Y, X)
    # i.e. data_5d[t, c, z, y, x]
    data_5d = stack.astype(np.float32)[
        ..., np.newaxis, np.newaxis
    ]  # becomes (T, H, W, 1, 1)
    data_5d = np.moveaxis(data_5d, (1, 2), (3, 4))  # now (T, 1, 1, H, W)

    # Build a minimal OME metadata dictionary
    # PhysicalSizeX/Y are in micrometers (µm), so divide nm by 1000
    ome_metadata = {
        "axes": "TCZYX",
        "PhysicalSizeX": float(pixel_size_nm) * 1e-3,
        "PhysicalSizeY": float(pixel_size_nm) * 1e-3,
        "PhysicalSizeZ": 1.0,  # we’re not truly volumetric, so set Z spacing to 1 µm
        "TimeIncrement": timestamps[1],  # assume uniform time increments
        "TimePoint": [float(t) if t is not None else 0.0 for t in timestamps],
        "ChannelName": [channel],  # just one channel here
    }

    dpi = 25_400_000.0 / float(pixel_size_nm)

    # Write the OME-TIFF
    # - data_5d is shape (T, C, Z, Y, X)
    # - photometric='minisblack' is appropriate for grayscale
    # - ome=True instructs tifffile to embed OME-XML
    tifffile.imwrite(
        str(path),
        data_5d,
        photometric="minisblack",
        metadata=ome_metadata,
        ome=True,
        resolution=(dpi, dpi),
        resolutionunit="INCH",
    )


def save_npz_bundle(
    path: Path,
    stack: np.ndarray,
    pixel_size_nm: float,
    timestamps: list[float],
    channel: str = "height_trace",
) -> None:
    """
    Save an AFM stack and metadata in a compressed ``.npz`` bundle.

    Parameters
    ----------
    path : Path
        Destination file path. ``.npz`` extension is added if missing.
    stack : np.ndarray
        Image stack of shape (N, H, W).
    pixel_size_nm : float
        Physical pixel size in nanometers.
    timestamps : list of float
        Frame timestamps in seconds.
    channel : str, default="height_trace"
        Channel name saved as part of the metadata.

    Returns
    -------
    None

    Notes
    -----
    - Data and metadata are saved as compressed NumPy arrays.
    - Missing timestamps are stored as ``NaN``.
    """

    path = check_path_is_path(path)
    path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    # We can store pixel_size_nm as a 0‐D array, timestamps as 1‐D array
    np.savez_compressed(
        str(path),
        data=stack.astype(np.float32),
        pixel_size_nm=np.array(pixel_size_nm, dtype=np.float32),
        timestamps=np.array(
            [float(t) if t is not None else np.nan for t in timestamps],
            dtype=np.float64,
        ),
        channel=np.array(channel, dtype=object),
    )


def save_h5_bundle(
    path: Path,
    stack: np.ndarray,
    pixel_size_nm: float,
    timestamps: list[float],
    frame_metadata: list[dict],
    channel: str = "height_trace",
) -> None:
    """
    Save an AFM stack and all metadata in a single HDF5 file.

    Parameters
    ----------
    path : Path
        Destination file path. ``.h5`` extension is enforced.
    stack : np.ndarray
        Image stack of shape (N, H, W).
    pixel_size_nm : float
        Physical pixel size in nanometers.
    timestamps : list of float
        Timestamps for each frame in seconds.
    frame_metadata : list of dict
        Full list of per-frame metadata dictionaries.
    channel : str, default="height_trace"
        Channel name stored in file attributes.

    Returns
    -------
    None

    Notes
    -----
    - Image data is stored with gzip compression.
    - Frame metadata is serialized as JSON and saved as an attribute.
    """
    path = check_path_is_path(path)
    path = path.with_suffix(".h5")
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(path), "w") as f:
        f.create_dataset("data", data=stack.astype(np.float32), compression="gzip")
        f.create_dataset("pixel_size_nm", data=np.float32(pixel_size_nm))
        f.create_dataset(
            "timestamps",
            data=np.array(
                [float(t) if t is not None else np.nan for t in timestamps],
                dtype=np.float64,
            ),
        )
        # If you want to keep full per‐frame metadata, embed as JSON in an attribute:
        f.attrs["channel"] = channel
        f.attrs["frame_metadata"] = json.dumps(frame_metadata)

    # after closing, user can reopen in Python and
    # reparse 'frame_metadata' via json.loads(...)


def export_bundles(
    afm_stack: AFMImageStack,
    output_folder: Path,
    base_name: str,
    formats: list[str],
    raw: bool = False,
) -> None:
    """
    Export AFM stacks with metadata to selected formats.

    Parameters
    ----------
    afm_stack : AFMImageStack
        AFM stack object containing image data and metadata.
    output_folder : Path
        Directory where output files will be saved.
    base_name : str
        Base name for output files (without extensions).
    formats : list of {"tif", "npz", "h5"}
        Formats to export. Multiple formats can be specified.
    raw : bool, default=False
        If ``True``, export the raw (unprocessed) data. Otherwise, export
        the processed data.


    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If any requested format is not one of ``{"tif", "npz", "h5"}``.

    Notes
    -----
    - Automatically creates ``output_folder`` if it does not exist.
    - Processed exports append ``"_filtered"`` to ``base_name``.
    - Pixel size, timestamps, and channel metadata are included in all formats.
    """

    # Determine whether to use raw or processed data
    # (allows saving of unfiltered from play mode)
    if raw is False:
        stack_data = afm_stack.data
    elif raw is True and "raw" in afm_stack.processed:
        stack_data = afm_stack.processed["raw"]

    timestamps = [md.get("timestamp") for md in afm_stack.frame_metadata]

    base_name = sanitize_output_name(base_name, Path(afm_stack.file_path).stem)

    raw_exists = "raw" in afm_stack.processed
    filtered_exists = raw_exists and any(
        key != "raw" for key in afm_stack.processed.keys()
    )
    if filtered_exists and raw is False:
        base_name = f"{base_name}_filtered"

    output_folder = prepare_output_directory(output_folder, default="output")
    output_folder.mkdir(parents=True, exist_ok=True)

    valid = {"tif", "npz", "h5"}
    for fmt in formats:
        if fmt not in valid:
            logger.error(f"Unsupported export format '{fmt}'. Choose from {valid}.")
            sys.exit(1)

    if "tif" in formats:
        tif_path = output_folder / f"{base_name}.ome.tif"
        logger.info(f"Writing OME-TIFF → {tif_path}")
        save_ome_tiff_stack(
            path=tif_path,
            stack=stack_data,
            pixel_size_nm=afm_stack.pixel_size_nm,
            timestamps=timestamps,
            channel=afm_stack.channel,
        )

    if "npz" in formats:
        npz_path = output_folder / f"{base_name}"
        logger.info(f"Writing NPZ bundle → {npz_path}.npz")
        save_npz_bundle(
            path=npz_path,
            stack=stack_data,
            pixel_size_nm=afm_stack.pixel_size_nm,
            timestamps=timestamps,
            channel=afm_stack.channel,
        )

    if "h5" in formats:
        h5_path = output_folder / f"{base_name}"
        logger.info(f"Writing HDF5 bundle → {h5_path}.h5")
        save_h5_bundle(
            path=h5_path,
            stack=stack_data,
            pixel_size_nm=afm_stack.pixel_size_nm,
            timestamps=timestamps,
            frame_metadata=afm_stack.frame_metadata,
            channel=afm_stack.channel,
        )

    logger.debug(f"[export] Bundles ({formats}) written to {output_folder}")
