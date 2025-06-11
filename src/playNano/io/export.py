"""Tools for exporting data in various formats."""

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


def save_ome_tiff_stack(
    path: Path,
    stack: np.ndarray,
    pixel_size_nm: float,
    timestamps: list[float],
    channel: str = "height_trace",
) -> None:
    """
    Save a 3D AFM stack as an OME-TIFF, embedding physical sizes and timepoints.

    - path: Path to “.ome.tif” file (you can name it “.tif” but
        ome=True writes OME XML internally)
    - stack: shape = (n_frames, H, W), dtype float or uint
    - pixel_size_nm: physical pixel size in nm
    - timestamps: list of length n_frames
    - channel: string channel name (stored in OME metadata)
    """
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
    Save a 3D AFM stack plus metadata into a compressed .npz file.

    - path: Path to “.npz” (no suffix needed; do path.with_suffix(".npz"))
    """
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
    Save a 3D AFM stack plus all metadata into a single HDF5 file.

    - path: Path to “.h5” (we'll force .h5 suffix).
    - frame_metadata: full list of dicts (one dict per frame).
    """
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
    Write out requested bundles from an AFM stack (.data must be final version).

    Parameters
    ----------
    afm_stack : AFMImageStack
        The AFM stack containing final .data, .pixel_size_nm, .frame_metadata, .channel
    out_folder : Path
        Directory to write export files (will be created if needed)
    base_name : str
        Base file name (no extension) for each export, e.g. "sample_01"
    formats : list of str
        Which formats to write; valid set = {"tif", "npz", "h5"}.
    raw : bool, optional
        If True, use the raw data from `afm_stack.processed["raw"]`.
        If False, use the final processed data in `afm_stack.data`.
        Default is False (use processed data).

    Raises
    ------
    SystemExit
        If any element of `formats` is not in {"tif","npz","h5"}.
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
