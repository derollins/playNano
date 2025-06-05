"""Tools for exporting data in various formats."""

import json
from pathlib import Path

import h5py
import numpy as np
import tifffile


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
