"""
Tools for exporting AFM image stacks in multiple formats.

Provides functions to export AFM stacks with metadata as OME-TIFF, NPZ, or HDF5
bundles. Handles path validation, metadata embedding, and file structure creation.
"""

import copy
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
        path = Path(path)
    elif isinstance(path, Path):
        pass
    else:
        raise TypeError(f"{path} is not a string or a Path.")
    return path


def save_ome_tiff_stack(
    path: Path,
    afm_stack: AFMImageStack,
    raw: bool = False,
) -> None:
    """
    Save an AFMImageStack as an OME-TIFF.

    Embedds both pixel data and key metadata (timestamps, provenance,
    processed-step names).

    Parameters
    ----------
    path : Path
        Output path for the OME-TIFF file (will be overwritten).
    afm_stack : AFMImageStack
        The image stack to export (its `.data` or `.processed['raw']`).
    raw : bool
        If True, use the raw snapshot (`.processed['raw']`), otherwise
        use the current `.data` array (post-filtered). Default is False.

    Returns
    -------
    None
    """
    # Select data array
    if raw and "raw" in afm_stack.processed:
        data = afm_stack.processed["raw"]
    else:
        data = afm_stack.data

    # Reshape to 5D (T,C,Z,Y,X)
    data_5d = data.astype(np.float32)[..., None, None]
    data_5d = np.moveaxis(data_5d, (1, 2), (3, 4))

    # Prepare provenance dict excluding 'analysis'
    provenance_clean = {
        k: copy.deepcopy(v) for k, v in afm_stack.provenance.items() if k != "analysis"
    }
    provenance_json = json.dumps(provenance_clean, default=str)
    provenance_bytes = provenance_json.encode("utf-8")

    # Prepare processed keys list as JSON bytes
    processed_json = json.dumps(list(afm_stack.processed.keys()))
    processed_bytes = processed_json.encode("utf-8")

    timestamps = afm_stack.get_frame_times()
    channel = afm_stack.channel

    # Add Plane elements for timestamps
    planes = [{"DeltaT": float(t)} for t in timestamps]

    # Build OME metadata dictionary (standard fields only)
    ome_metadata = {
        "axes": "TCZYX",
        "PhysicalSizeX": afm_stack.pixel_size_nm * 1e-3,
        "PhysicalSizeY": afm_stack.pixel_size_nm * 1e-3,
        "PhysicalSizeZ": 1.0,
        "TimeIncrement": timestamps[1] - timestamps[0] if len(timestamps) > 1 else 0.0,
        "Plane": planes,
        "Channel": [{"Name": channel}],
    }

    # Create custom TIFF tags for provenance and processed keys
    # Tag 65000 and 65001 are arbitrary but in the private/custom range
    extratags = [
        (65000, 7, len(provenance_bytes), provenance_bytes, True),  # Provenance JSON
        (65001, 7, len(processed_bytes), processed_bytes, True),  # Processed keys JSON
    ]

    dpi = 25_400_000.0 / float(afm_stack.pixel_size_nm)

    tifffile.imwrite(
        str(path),
        data_5d,
        photometric="minisblack",
        metadata=ome_metadata,  # standard OME metadata here
        ome=True,
        resolution=(dpi, dpi),
        resolutionunit="INCH",
        extratags=extratags,
    )


def save_npz_bundle(path: Path, stack: AFMImageStack, raw: bool = False) -> None:
    """
    Save an AFMImageStack (data + metadata + processed + masks) in a single .npz file.

    Top-level arrays / metadata keys:
      - data               : float32 array, (n_frames, H, W)
      - pixel_size_nm      : float32 scalar array
      - timestamps         : float64 array, (n_frames,)
      - channel            : object scalar array
      - frame_metadata_json: object scalar array (JSON dump of list of dicts)
      - provenance_json    : object scalar array (JSON dump of provenance dict)

    Then one array per processed-step under key `processed__<step_name>`, and one per
    mask under `masks__<mask_name>`.

    Parameters
    ----------
    path : Path
        Base filepath for the .npz (the “.npz” extension will be added).
    stack : AFMImageStack
        The stack to serialize.
    raw : bool
        If True, save only the raw snapshot (`stack.processed['raw']`), and
        exclude `.processed` and `.masks`. Otherwise, save the entire AFMImageStack.
        Default is False.

    Returns
    -------
    None
    """
    # ensure .npz extension
    path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = np.array(stack.get_frame_times(), dtype=np.float64)
    channel = np.array(stack.channel, dtype=object)
    frame_metadata_json = np.array(json.dumps(stack.frame_metadata), dtype=object)
    provenance_clean = {k: v for k, v in stack.provenance.items() if k != "analysis"}
    provenance_json = np.array(json.dumps(provenance_clean), dtype=object)

    if raw and "raw" in stack.processed:
        data_to_save = stack.processed["raw"]
        arrays = {
            "data": data_to_save.astype(np.float32),
            "pixel_size_nm": np.array(stack.pixel_size_nm, dtype=np.float32),
            "timestamps": timestamps,
            "channel": channel,
            "frame_metadata_json": frame_metadata_json,
            "provenance_json": provenance_json,
        }
    else:
        arrays = {
            "data": stack.data.astype(np.float32),
            "pixel_size_nm": np.array(stack.pixel_size_nm, dtype=np.float32),
            "timestamps": timestamps,
            "channel": channel,
            "frame_metadata_json": frame_metadata_json,
            "provenance_json": provenance_json,
        }
        for name, arr in stack.processed.items():
            arrays[f"processed__{name}"] = arr.astype(np.float32)
        for name, m in stack.masks.items():
            arrays[f"masks__{name}"] = m.astype(bool)

    np.savez_compressed(str(path), **arrays)
    logger.info(f"Wrote NPZ bundle → {path}")


def save_h5_bundle(path: Path, stack: AFMImageStack, raw: bool = False) -> None:
    """
    Save an AFMImageStack (data, metadata, processed, masks) into a single HDF5 file.

    File structure
    --------------
    - ``/data`` : float32 dataset, shape (n_frames, H, W)

    - ``/processed/<step>`` : float32 datasets, per-step snapshots

    - ``/masks/<mask>`` : boolean datasets, per-mask

    - ``/timestamps`` : float64 dataset, shape (n_frames,)

    - ``/frame_metadata_json`` : variable-length UTF-8 string, JSON dump of list of
      dicts

    - ``/provenance_json`` : variable-length UTF-8 string, JSON dump of provenance dict

    Root attributes
    ---------------
    - ``pixel_size_nm`` : float

    - ``channel`` : UTF-8 string

    Parameters
    ----------
    path : Path
        Base filepath for the HDF5 file (the ``.h5`` extension will be added if
        missing).
    stack : AFMImageStack
        The stack to serialize.
    raw : bool
        If True, save only the raw snapshot (``stack.processed['raw']``), excluding
        ``.processed`` and ``.masks``. Otherwise, save the entire AFMImageStack.
        Default is False.

    Returns
    -------
    None
    """
    path = check_path_is_path(path).with_suffix(".h5")
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(path), "w") as f:
        if raw and "raw" in stack.processed:
            f.create_dataset(
                "data",
                data=stack.processed["raw"].astype(np.float32),
                compression="gzip",
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

        timestamps = np.array(stack.get_frame_times(), dtype=np.float64)
        f.create_dataset("timestamps", data=timestamps)

        meta_json = json.dumps(stack.frame_metadata).encode("utf-8")
        provenance_clean = {
            k: v for k, v in stack.provenance.items() if k != "analysis"
        }
        provenance_json = np.array(json.dumps(provenance_clean), dtype=object)

        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("frame_metadata_json", data=meta_json, dtype=dt)
        f.create_dataset("provenance_json", data=provenance_json, dtype=dt)

        f.attrs["pixel_size_nm"] = float(stack.pixel_size_nm)
        f.attrs["channel"] = stack.channel

    logger.info(f"Wrote HDF5 bundle → {path}")


def export_bundles(
    afm_stack: AFMImageStack,
    output_folder: Path,
    base_name: str,
    formats: list[str],
    raw: bool = False,
) -> None:
    """
    Export a playNano AFMImageStack in OME-TIFF, NPZ, and/or HDF5 formats.

    OME-TIFF
    --------
    - Writes a single 5D TCZYX TIFF containing *only* pixel data.
    - Embeds these metadata in OME-XML “UserData…” tags:
      - `PhysicalSizeX/Y/Z`, `TimeIncrement`, `TimePoint`, `ChannelName`
      - `UserDataProcessed`: JSON list of processing step names
      - `UserDataProvenance`: JSON dump of the stack's provenance dict
    - Use ``raw=True`` to export the unfiltered snapshot
      (``.processed['raw']``). Otherwise, exports the current
      ``.data`` (post-filtered).

    NPZ & HDF5 Bundles
    ------------------
    - Round-trip safe serialization of the *entire* AFMImageStack:
      ``.data``, ``.processed``, ``.masks``, ``.frame_metadata``,
      ``.pixel_size_nm``, ``.channel``, and ``.provenance``.
    - Reloading yields the identical Python object.
    - Use ``raw=True`` to export just the unfiltered snapshot
      (``.processed['raw']``). Otherwise, exports all data.

    Notes
    -----
    - If ``raw=True``, all formats (TIFF, NPZ, HDF5) will contain only the raw
      snapshot (`.processed['raw']`) and ignore `.processed` and `.masks`.
    - If ``raw=False``, OME-TIFF exports `afm_stack.data`, while NPZ and HDF5
      export the entire AFMImageStack.

    Parameters
    ----------
    afm_stack : AFMImageStack
        The stack to export.
    output_folder : Path
        Directory in which to write the outputs.
    base_name : str
        Base filename (no extension) for each export.
    formats : list of {"tif", "npz", "h5"}
        Formats to produce.
    raw : bool
        If True, the exports will contain the raw snapshot
        (``.processed['raw']``). Default is False.

    Raises
    ------
    SystemExit
        If any entry in ``formats`` is not one of {"tif","npz","h5"}.
    """
    valid = {"tif", "npz", "h5"}
    bad = set(formats) - valid
    if bad:
        logger.error(f"Unsupported format(s): {bad}. Choose from {valid}.")
        sys.exit(1)

    # ensure output folder exists
    output_folder = prepare_output_directory(output_folder, default="output")
    output_folder.mkdir(parents=True, exist_ok=True)

    # build the common stem only ONCE
    stem = sanitize_output_name(base_name, Path(afm_stack.file_path).stem)
    has_filtered = any(k != "raw" for k in afm_stack.processed)
    if not raw and has_filtered:
        stem += "_filtered"

    # 1) OME‑TIFF
    if "tif" in formats:
        tif_path = output_folder / f"{stem}.ome.tif"
        logger.info(f"Writing OME-TIFF → {tif_path}")
        save_ome_tiff_stack(path=tif_path, afm_stack=afm_stack, raw=raw)

    # 2) NPZ
    if "npz" in formats:
        npz_path = output_folder / f"{stem}.npz"
        logger.info(f"Writing NPZ bundle → {npz_path}")
        save_npz_bundle(path=npz_path, stack=afm_stack, raw=raw)

    # 3) HDF5
    if "h5" in formats:
        h5_path = output_folder / f"{stem}.h5"
        logger.info(f"Writing HDF5 bundle → {h5_path}")
        save_h5_bundle(path=h5_path, stack=afm_stack, raw=raw)
