"""
Tools for exporting AFM image stacks in multiple formats.

This module provides utilities for serializing
:class:`~playNano.afm_stack.AFMImageStack` objects
into round-trip-safe formats: OME-TIFF, NPZ, and HDF5.
Each export preserves pixel data, metadata, masks, provenance, and processing history.

Supported formats
-----------------
- **OME-TIFF**: For image analysis interoperability (Bio-Formats compatible).
- **NPZ**: Compact NumPy archive with full AFMImageStack data and metadata.
- **HDF5**: Hierarchical bundle ideal for provenance-rich workflows.

Each export supports both *filtered* and *raw* modes.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import tifffile

from playNano.afm_stack import AFMImageStack
from playNano.utils.io_utils import (
    make_json_safe,
    prepare_output_directory,
    sanitize_output_name,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Path utilities
# -------------------------------------------------------------------------
def check_path_is_path(path: str | Path) -> Path:
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
        return Path(path)
    if isinstance(path, Path):
        return path
    raise TypeError(f"{path!r} is not a string or Path.")


# -------------------------------------------------------------------------
# OME-TIFF Export
# -------------------------------------------------------------------------
def save_ome_tiff_stack(
    path: Path, afm_stack: AFMImageStack, raw: bool = False
) -> None:
    """
    Save an :class:`~playNano.afm_stack.AFMImageStack` as an OME-TIFF file.

    The OME-TIFF export embeds image data, pixel calibration, timestamps,
    and provenance metadata into a single, standards-compliant file suitable
    for downstream analysis in microscopy or image-processing software.

    File structure
    --------------
    - Image data is stored in 5D OME-TIFF format with axes ``(T, C, Z, Y, X)``.
      Only a single channel (C=1) and a single Z-slice (Z=1) are used.
    - Physical calibration is stored in micrometres (µm) under
      ``PhysicalSizeX`` and ``PhysicalSizeY``.
    - Provenance and processed-layer keys are stored as binary JSON
      under private TIFF tags ``65000`` and ``65001``.

    Parameters
    ----------
    path : Path
        Output path for the `.ome.tif` file.
    afm_stack : AFMImageStack
        The AFM image stack to export.
    raw : bool, optional
        If True, export the unprocessed raw snapshot
        (``processed['raw']`` if present). Otherwise, export the current
        data in ``.data`` with all processing applied.

    Notes
    -----
    - Each frame's timestamp is stored in the OME metadata as ``DeltaT``.
    - The pixel size (in nm) is converted to micrometres for OME compliance.
    - The exported file includes additional TIFF tags:

      * **65000:** JSON-encoded provenance dictionary
      * **65001:** JSON list of processed layer names

    - The file can be reloaded with standard OME-TIFF readers such as
      :mod:`tifffile` or :mod:`aicsimageio`.

    Examples
    --------
    >>> from pathlib import Path
    >>> from playNano.io.export_data import save_ome_tiff_stack
    >>> save_ome_tiff_stack(Path("output/stack.ome.tif"), stack)
    >>> save_ome_tiff_stack(Path("output/raw_stack.ome.tif"), stack, raw=True)
    """

    # --- Select data and timestamps ---
    if raw and "raw" in afm_stack.processed:
        data = afm_stack.processed["raw"]
        meta_src = afm_stack.state_backups.get(
            "frame_metadata_before_edit", afm_stack.frame_metadata
        )
    else:
        data = afm_stack.data
        meta_src = afm_stack.frame_metadata

    timestamps = [md["timestamp"] for md in meta_src]

    # --- Reshape to 5D (T,C,Z,Y,X) ---
    data_5d = data.astype(np.float32)[..., None, None]
    data_5d = np.moveaxis(data_5d, (1, 2), (3, 4))

    # --- Metadata ---
    provenance_clean = {k: copy.deepcopy(v) for k, v in afm_stack.provenance.items()}
    provenance_json = json.dumps(provenance_clean, default=str).encode("utf-8")
    processed_json = json.dumps(list(afm_stack.processed.keys())).encode("utf-8")

    channel = afm_stack.channel
    planes = [{"DeltaT": float(t)} for t in timestamps]

    ome_metadata = {
        "axes": "TCZYX",
        "PhysicalSizeX": afm_stack.pixel_size_nm * 1e-3,
        "PhysicalSizeY": afm_stack.pixel_size_nm * 1e-3,
        "PhysicalSizeZ": 1.0,
        "TimeIncrement": (
            (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        ),
        "Plane": planes,
        "Channel": [{"Name": channel}],
    }

    extratags = [
        (65000, 7, len(provenance_json), provenance_json, True),
        (65001, 7, len(processed_json), processed_json, True),
    ]

    dpi = 25_400_000.0 / float(afm_stack.pixel_size_nm)

    tifffile.imwrite(
        str(path),
        data_5d,
        photometric="minisblack",
        metadata=ome_metadata,
        ome=True,
        resolution=(dpi, dpi),
        resolutionunit="INCH",
        extratags=extratags,
    )

    logger.info(f"Wrote OME-TIFF → {path}")


# -------------------------------------------------------------------------
# NPZ Export
# -------------------------------------------------------------------------


def save_npz_bundle(path: Path, stack: AFMImageStack, raw: bool = False) -> None:
    """
    Save :class:`~playNano.afm_stack.AFMImageStack` with metadata as a `.npz` bundle.

    The NPZ archive consolidates the AFM stack data, metadata, provenance,
    masks, and processed layers for full round-trip reconstruction.

    File structure
    --------------
    data                    : float32 array of shape (n_frames, H, W)
    processed__<step>       : float32 arrays for each processing step
    masks__<mask>           : bool arrays for each mask
    timestamps              : float64 array of length n_frames
    frame_metadata_json     : UTF-8 encoded JSON string
    provenance_json         : UTF-8 encoded JSON string
    state_backups_json      : UTF-8 encoded JSON string (only if present)
    pixel_size_nm           : float32 scalar
    channel                 : object array (string)

    Parameters
    ----------
    path : Path
        Destination path for the `.npz` file. The extension will be added
        automatically.
    stack : AFMImageStack
        The stack to export.
    raw : bool, optional
        If True, only the unprocessed raw snapshot and essential metadata are saved.
        Otherwise, the full data (`.data`), masks, and processed layers are included.

    Notes
    -----
    - ``state_backups_json`` is only included if the stack defines a ``state_backups``
      attribute.
    - When ``raw=True``, timestamps are taken from
      ``state_backups['frame_metadata_before_edit']`` (if available),
      otherwise from ``.frame_metadata``.
    - Provenance is serialized using
      :func:`~playNano.utils.io_utils.make_json_safe`
      to ensure JSON compatibility.
    - The saved file can be reloaded via
      :func:`~playNano.io.import_data.load_npz_bundle`.

    Examples
    --------
    >>> from pathlib import Path
    >>> from playNano.io.export_data import save_npz_bundle
    >>> save_npz_bundle(Path("output/stack_export"), stack)
    >>> save_npz_bundle(Path("output/raw_only"), stack, raw=True)
    """
    path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    # --- Core metadata ---
    channel = np.array(stack.channel, dtype=object)
    provenance_clean = make_json_safe(stack.provenance)
    provenance_json = json.dumps(provenance_clean).encode("utf-8")

    arrays = {
        "pixel_size_nm": np.array(stack.pixel_size_nm, dtype=np.float32),
        "channel": channel,
        "frame_metadata_json": np.array(json.dumps(stack.frame_metadata), dtype=object),
        "provenance_json": np.array(provenance_json, dtype=object),
    }

    # only save state_backups if it exists
    if hasattr(stack, "state_backups"):
        arrays["state_backups_json"] = np.array(
            json.dumps(stack.state_backups), dtype=object
        )

    # --- Data and optional layers ---
    if raw and "raw" in stack.processed:
        data = stack.processed["raw"]
        meta_src = getattr(stack, "state_backups", {}).get(
            "frame_metadata_before_edit", stack.frame_metadata
        )
        timestamps = [m["timestamp"] for m in meta_src]
        arrays["data"] = data.astype(np.float32)
        arrays["timestamps"] = np.array(timestamps, dtype=np.float64)
    else:
        arrays["data"] = stack.data.astype(np.float32)
        arrays["timestamps"] = np.array(stack.get_frame_times(), dtype=np.float64)
        for name, arr in stack.processed.items():
            arrays[f"processed__{name}"] = arr.astype(np.float32)
        for name, m in stack.masks.items():
            arrays[f"masks__{name}"] = m.astype(bool)

    np.savez_compressed(str(path), **arrays)
    logger.info(f"Wrote NPZ bundle → {path}")


# -------------------------------------------------------------------------
# HDF5 Export
# -------------------------------------------------------------------------
def save_h5_bundle(path: Path, stack: AFMImageStack, raw: bool = False) -> None:
    """
    Save an :class:`~playNano.afm_stack.AFMImageStack` and metadata as an HDF5 bundle.

    The hierarchical layout preserves round-trip reconstruction fidelity and stores
    all relevant AFM stack data, processed layers, masks, timestamps, and provenance.

    File structure
    --------------
    /data                    : float32 array of shape (n_frames, H, W)
    /processed/<step>        : float32 datasets for each processing step
    /masks/<mask>            : bool datasets for each mask
    /timestamps              : float64 array of length n_frames
    /frame_metadata_json     : UTF-8 encoded JSON string
    /provenance_json         : UTF-8 encoded JSON string
    /state_backups_json      : UTF-8 encoded JSON string (only if present)

    Root attributes
    ---------------
    pixel_size_nm : float
        Physical pixel size in nanometers.
    channel       : str
        Name of the imaging channel.

    Parameters
    ----------
    path : Path
        Destination path for the HDF5 file. The '.h5' suffix will be added
        automatically.
    stack : AFMImageStack
        The stack to export.
    raw : bool, optional
        If True, only the unprocessed raw snapshot (`processed['raw']`) is exported.
        Otherwise, the full `.data`, masks, and processed layers are included.

    Notes
    -----
    - `state_backups_json` is only created if the stack has a non-empty
    `state_backups` attribute.
    - Timestamps are taken from `state_backups['frame_metadata_before_edit']`
    if exporting raw data, otherwise from `.frame_metadata`.
    - Provenance is sanitized via :func:`~playNano.utils.io_utils.make_json_safe`.
    """
    path = check_path_is_path(path).with_suffix(".h5")
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(path), "w") as f:
        # --- Core data ---
        if raw and "raw" in stack.processed:
            f.create_dataset(
                "data",
                data=stack.processed["raw"].astype(np.float32),
                compression="gzip",
            )
            meta_src = stack.state_backups.get(
                "frame_metadata_before_edit", stack.frame_metadata
            )
        else:
            f.create_dataset(
                "data", data=stack.data.astype(np.float32), compression="gzip"
            )

            proc_grp = f.create_group("processed")
            for name, arr in stack.processed.items():
                proc_grp.create_dataset(
                    name, data=arr.astype(np.float32), compression="gzip"
                )

            mask_grp = f.create_group("masks")
            for name, m in stack.masks.items():
                mask_grp.create_dataset(name, data=m.astype(bool), compression="gzip")

            meta_src = stack.frame_metadata

        timestamps = np.array([md["timestamp"] for md in meta_src], dtype=np.float64)
        f.create_dataset("timestamps", data=timestamps)

        # --- JSON metadata ---
        provenance_clean = make_json_safe(stack.provenance)
        provenance_json = json.dumps(provenance_clean).encode("utf-8")

        f.create_dataset("frame_metadata_json", data=np.string_(json.dumps(meta_src)))
        f.create_dataset("provenance_json", data=np.string_(provenance_json))
        if getattr(stack, "state_backups", None):
            f.create_dataset(
                "state_backups_json", data=np.string_(json.dumps(stack.state_backups))
            )

        # --- Root attributes ---
        f.attrs["pixel_size_nm"] = stack.pixel_size_nm
        f.attrs["channel"] = stack.channel

    logger.info(f"Wrote {'raw ' if raw else ''}HDF5 bundle → {path}")


# -------------------------------------------------------------------------
# Unified export entry point
# -------------------------------------------------------------------------
def export_bundles(
    afm_stack: AFMImageStack,
    output_folder: Path,
    base_name: str,
    formats: list[str],
    raw: bool = False,
) -> None:
    """
    Export an :class:`AFMImageStack` in one or more serialization formats.

    Parameters
    ----------
    afm_stack : AFMImageStack
        The stack to export.
    output_folder : Path
        Target directory for output files.
    base_name : str
        Base filename (no extension).
    formats : list of {"tif", "npz", "h5"}
        Which formats to produce.
    raw : bool, optional
        If True, exports only the unprocessed raw snapshot, (``.processed['raw']``).
        Default is False.

    Raises
    ------
    SystemExit
        If an unsupported format string is provided.
    """
    valid = {"tif", "npz", "h5"}
    bad = set(formats) - valid
    if bad:
        logger.error(f"Unsupported format(s): {bad}. Choose from {valid}.")
        sys.exit(1)

    output_folder = prepare_output_directory(output_folder, default="output")
    output_folder.mkdir(parents=True, exist_ok=True)

    stem = sanitize_output_name(
        base_name,
        Path(afm_stack.file_path).stem if afm_stack.file_path else "playnano_export",
    )

    if not raw and any(k != "raw" for k in afm_stack.processed):
        stem += "_filtered"

    if "tif" in formats:
        save_ome_tiff_stack(output_folder / f"{stem}.ome.tif", afm_stack, raw)
    if "npz" in formats:
        save_npz_bundle(output_folder / f"{stem}.npz", afm_stack, raw)
    if "h5" in formats:
        save_h5_bundle(output_folder / f"{stem}.h5", afm_stack, raw)
