"""
Save AFM image stacks as OME-TIFF, NPZ or HDF5 files.

- **OME-TIFF** (:func:`save_ome_tiff_stack`): one image stack with pixel size,
  frame times and provenance, readable by other microscopy software (for
  example Fiji/Bio-Formats).
- **NPZ** (:func:`save_npz_bundle`) and **HDF5** (:func:`save_h5_bundle`):
  bundles with the data, every processed layer and mask, the frame metadata and
  the provenance, which playNano can load again with
  :mod:`playnano.io.data_loaders`.

:func:`export_bundles` writes any of the three at once, as ``playnano process
--export`` does. All formats store heights as float32.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import tifffile

from playnano.afm_stack import AFMImageStack
from playnano.utils.io_utils import (
    ensure_not_input_path,
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
    Return ``path`` as a :class:`pathlib.Path`.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to convert.

    Returns
    -------
    pathlib.Path
        ``path`` as a Path.

    Raises
    ------
    TypeError
        If ``path`` is neither a str nor a Path.
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
    Save a stack as an OME-TIFF file.

    The data is written as float32 with axes (T, C, Z, Y, X) and one channel and
    Z plane. The OME metadata records the pixel size (in µm), each frame's time
    (``DeltaT``, in s), the interval between the first two frames
    (``TimeIncrement``) and the channel name. Two private TIFF tags hold JSON:

    - 65000: the provenance (``stack.provenance``).
    - 65001: the names of the processed layers. The layers themselves and the
      masks are not saved; use NPZ or HDF5 for those.

    Parameters
    ----------
    path : pathlib.Path
        Output file, for example ``Path("out/stack.ome.tif")``. Used as given;
        its folder must exist.
    afm_stack : AFMImageStack
        The stack to save.
    raw : bool, optional
        If True, save the unprocessed snapshot ``stack.processed["raw"]`` instead
        of ``stack.data``, without masks or processed layers. If the stack has
        no ``"raw"`` snapshot, ``stack.data`` is saved. Default is False.

    Raises
    ------
    ValueError
        If the frames do not all have the same pixel size.

    See Also
    --------
    playnano.io.data_loaders.load_ome_tiff_stack : Load the file again.

    Examples
    --------
    >>> from pathlib import Path
    >>> from playnano.io.export_data import save_ome_tiff_stack
    >>> save_ome_tiff_stack(Path("output/stack.ome.tif"), stack)  # doctest: +SKIP
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

    pixel_sizes = [md["frame_pixel_size_nm"] for md in meta_src]
    if not np.allclose(afm_stack.pixel_size_nm, pixel_sizes):
        raise ValueError(
            f"pixel_size_nm is not consistent across frames "
            f"(min={min(pixel_sizes):.4f}, max={max(pixel_sizes):.4f} nm)."
        )

    ome_metadata = {
        "axes": "TCZYX",
        "PhysicalSizeX": afm_stack.pixel_size_nm * 1e-3,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": afm_stack.pixel_size_nm * 1e-3,
        "PhysicalSizeYUnit": "µm",
        "TimeIncrement": (
            (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        ),
        "TimeIncrementUnit": "s",
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
    Save a stack, with its layers, masks and metadata, as a NumPy ``.npz`` bundle.

    The archive is compressed and contains these arrays:

    - ``data``: float32, shape (n_frames, height, width).
    - ``processed__<name>``: float32, one per entry in ``stack.processed``,
      including ``processed__raw``.
    - ``masks__<name>``: bool, one per entry in ``stack.masks``.
    - ``timestamps``: float64, frame times in seconds.
    - ``frame_metadata_json``, ``provenance_json``, ``state_backups_json``:
      JSON text.
    - ``pixel_size_nm``: float32 scalar.
    - ``channel``: the channel name.

    Parameters
    ----------
    path : pathlib.Path
        Output file. Its extension is replaced by ``.npz`` (``out/stack``
        becomes ``out/stack.npz``), and missing folders are created.
    stack : AFMImageStack
        The stack to save.
    raw : bool, optional
        If True, save the unprocessed snapshot ``stack.processed["raw"]`` instead
        of ``stack.data``, without masks or processed layers. If the stack has
        no ``"raw"`` snapshot, ``stack.data`` is saved. Default is False. The
        timestamps are then those from before any frames were removed.

    See Also
    --------
    playnano.io.data_loaders.load_npz_bundle : Load the bundle again.

    Examples
    --------
    >>> from pathlib import Path
    >>> from playnano.io.export_data import save_npz_bundle
    >>> save_npz_bundle(Path("output/stack"), stack)  # doctest: +SKIP
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
    Save a stack, with its layers, masks and metadata, as an HDF5 bundle.

    The file contains these datasets, compressed with gzip where they are
    arrays:

    - ``/data``: float32, shape (n_frames, height, width).
    - ``/processed/<name>``: float32, one per entry in ``stack.processed``,
      including ``/processed/raw``.
    - ``/masks/<name>``: bool, one per entry in ``stack.masks``.
    - ``/timestamps``: float64, frame times in seconds.
    - ``/frame_metadata_json``, ``/provenance_json``: JSON text.
    - ``/state_backups_json``: JSON text, only if ``stack.state_backups`` is
      not empty.

    and the file attributes ``pixel_size_nm`` and ``channel``.

    Parameters
    ----------
    path : str or pathlib.Path
        Output file. Its extension is replaced by ``.h5``, and missing folders
        are created.
    stack : AFMImageStack
        The stack to save.
    raw : bool, optional
        If True, save the unprocessed snapshot ``stack.processed["raw"]`` instead
        of ``stack.data``, without masks or processed layers. If the stack has
        no ``"raw"`` snapshot, ``stack.data`` is saved. Default is False. The
        timestamps and frame metadata are then those from before any frames were
        removed.

    See Also
    --------
    playnano.io.data_loaders.load_h5_bundle : Load the bundle again.
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

        f.create_dataset("frame_metadata_json", data=np.bytes_(json.dumps(meta_src)))
        f.create_dataset("provenance_json", data=np.bytes_(provenance_json))
        if getattr(stack, "state_backups", None):
            f.create_dataset(
                "state_backups_json", data=np.bytes_(json.dumps(stack.state_backups))
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
    Save a stack in one or more of the formats OME-TIFF, NPZ and HDF5.

    Files are named ``<name>.ome.tif``, ``<name>.npz`` and ``<name>.h5``.
    ``_filtered`` is added to the name when processed data is saved, that is
    when ``raw`` is False and the stack has processed layers other than
    ``"raw"``.

    Parameters
    ----------
    afm_stack : AFMImageStack
        The stack to save.
    output_folder : str or pathlib.Path or None
        Folder for the files; created if missing. If None, ``./output``.
    base_name : str or None
        File name without extension. Any folder part and known extension are
        removed. If empty or None, the name of the file the stack was loaded
        from is used.
    formats : list of str
        Any of ``"tif"``, ``"npz"`` and ``"h5"``.
    raw : bool, optional
        If True, save the unprocessed snapshot ``stack.processed["raw"]``. See
        :func:`save_npz_bundle`. Default is False.

    Raises
    ------
    ValueError
        If a format is not recognised, or an output file would overwrite the
        file the stack was loaded from. Nothing is written in either case.
    """
    valid = {"tif", "npz", "h5"}
    bad = set(formats) - valid
    if bad:
        raise ValueError(
            f"Unsupported export format(s): {', '.join(sorted(bad))}. "
            f"Choose from {', '.join(sorted(valid))}."
        )

    output_folder = prepare_output_directory(output_folder, default="output")
    output_folder.mkdir(parents=True, exist_ok=True)

    stem = sanitize_output_name(
        base_name,
        Path(afm_stack.file_path).stem if afm_stack.file_path else "playnano_export",
    )

    if not raw and any(k != "raw" for k in afm_stack.processed):
        stem += "_filtered"

    out_paths = {
        "tif": output_folder / f"{stem}.ome.tif",
        "npz": output_folder / f"{stem}.npz",
        "h5": output_folder / f"{stem}.h5",
    }
    for fmt in formats:
        ensure_not_input_path(out_paths[fmt], afm_stack.file_path)

    if "tif" in formats:
        save_ome_tiff_stack(out_paths["tif"], afm_stack, raw)
    if "npz" in formats:
        save_npz_bundle(out_paths["npz"], afm_stack, raw)
    if "h5" in formats:
        save_h5_bundle(out_paths["h5"], afm_stack, raw)
