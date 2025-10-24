"""
Data loaders for AFM image stacks exported by **playNano**.

This module provides readers for serialized AFMImageStack bundles created
by the export routines (``.npz``, ``.h5``, and OME-TIFF). Each loader
reconstructs a :class:`~playNano.afm_stack.AFMImageStack` with correct
data, pixel size, channel name, and per-frame metadata (timestamps).
All loaders restore provenance and any stored processing or mask data.

Functions
---------
load_npz_bundle
    Load a `.npz` bundle into an :class:`~playNano.afm_stack.AFMImageStack`.
load_h5_bundle
    Load a `.h5` bundle into an :class:`~playNano.afm_stack.AFMImageStack`.
load_ome_tiff_stack
    Load an OME-TIFF bundle into an :class:`~playNano.afm_stack.AFMImageStack`.
"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import tifffile

from playNano.afm_stack import AFMImageStack

logger = logging.getLogger(__name__)


def load_npz_bundle(path: Path, channel: str = "height_trace") -> AFMImageStack:
    """
    Load an :class:`~playNano.afm_stack.AFMImageStack` from a `.npz` bundle.

    The `.npz` file must contain the following keys:

    - ``data`` : ``float32`` array of shape ``(n_frames, H, W)``
    - ``pixel_size_nm`` : scalar ``float``
    - ``channel`` : ``str`` scalar
    - ``frame_metadata_json`` : JSON-encoded list of dicts
    - ``provenance_json`` : JSON-encoded dict
    - ``processed__<step>`` : optional processed frame arrays
    - ``masks__<mask>`` : optional boolean mask arrays
    - ``state_backups_json`` : optional JSON-encoded dict of saved states

    This is the structure produced by :func:`playNano.io.export_data.save_npz_bundle`.

    Parameters
    ----------
    path : pathlib.Path
        Path to the `.npz` file.
    channel : str, default="height_trace"
        Provided for API compatibility with :func:`~playNano.io.loader.load_afm_stack`
        but ignored when reading the bundle.

    Returns
    -------
    playNano.afm_stack.AFMImageStack
        Reconstructed AFM image stack with attributes populated:
        ``.processed``, ``.masks``, and ``.provenance``.

    Raises
    ------
    ValueError
        If required keys are missing or JSON blobs cannot be decoded.
    """
    arrs = np.load(str(path), allow_pickle=True)

    # Core data
    data = arrs["data"]
    pixel_size_nm = float(arrs["pixel_size_nm"].item())
    channel = str(arrs["channel"].item())

    # Metadata
    try:
        frame_metadata = json.loads(arrs["frame_metadata_json"].item())
    except KeyError:
        raise ValueError(f"{path} missing 'frame_metadata_json'") from None
    except Exception as e:
        raise ValueError(
            f"{path}: invalid JSON in 'frame_metadata_json': {e}"
        ) from None

    try:
        provenance = json.loads(arrs["provenance_json"].item())
    except KeyError:
        raise ValueError(f"{path} missing 'provenance_json'") from None
    except Exception as e:
        raise ValueError(f"{path}: invalid JSON in 'provenance_json': {e}") from None

    state_backups = None
    if "state_backups_json" in arrs:
        try:
            state_backups = json.loads(arrs["state_backups_json"].item())
        except Exception as e:
            raise ValueError(
                f"{path}: invalid JSON in 'state_backups_json': {e}"
            ) from None

    # Build stack
    stack = AFMImageStack(
        data=data,
        pixel_size_nm=pixel_size_nm,
        channel=channel,
        file_path=path,
        frame_metadata=frame_metadata,
    )
    if state_backups is not None:
        stack.state_backups = state_backups

    # Provenance
    saved_prov = provenance.copy()
    # annotate bundle info
    saved_prov.setdefault("bundle", {}).update(bundle_file=str(path), bundle_type="npz")
    # then replace stack.provenance wholesale
    stack.provenance = saved_prov

    # Processed and mask layers
    for key in arrs.files:
        if key.startswith("processed__"):
            step = key.split("__", 1)[1]
            stack.processed[step] = arrs[key].astype(np.float32)
        elif key.startswith("masks__"):
            mask = key.split("__", 1)[1]
            stack.masks[mask] = arrs[key].astype(bool)

    return stack


def load_h5_bundle(path: Path, channel: str = "height_trace") -> AFMImageStack:
    """
    Load an :class:`~playNano.afm_stack.AFMImageStack` from an HDF5 bundle.

    Expected HDF5 structure
    -----------------------
    Datasets
        - ``/data`` : ``float32`` array of shape ``(n_frames, H, W)``
        - ``/processed/<step>`` : optional processed datasets
        - ``/masks/<mask>`` : optional boolean mask datasets
        - ``/frame_metadata_json`` : UTF-8 encoded JSON (list of dicts)
        - ``/provenance_json`` : UTF-8 encoded JSON (dict)
        - ``/state_backups_json`` : optional UTF-8 JSON (dict)
    Attributes
        - ``pixel_size_nm`` : scalar float
        - ``channel`` : string

    Files with the structure are produced by
    :func:`playNano.io.export_data.save_h5_bundle`.

    Parameters
    ----------
    path : pathlib.Path
        Path to the `.h5` file.
    channel : str, default="height_trace"
        Provided for API compatibility with :func:`~playNano.io.loader.load_afm_stack`
        but ignored when reading the bundle.

    Returns
    -------
    playNano.afm_stack.AFMImageStack
        Fully reconstructed AFM image stack with provenance, processed steps,
        and masks restored.

    Raises
    ------
    ValueError
        If required datasets are missing or JSON decoding fails.
        If required datasets (``frame_metadata_json`` or ``provenance_json``)
        are missing or contain invalid JSON.
    """
    with h5py.File(str(path), "r") as f:
        data = f["data"][()].astype(np.float32)
        pixel_size_nm = float(f.attrs["pixel_size_nm"])
        channel = str(f.attrs["channel"])

        processed = {
            n: ds[()].astype(np.float32) for n, ds in f.get("processed", {}).items()
        }
        masks = {n: ds[()].astype(bool) for n, ds in f.get("masks", {}).items()}

        if "frame_metadata_json" not in f:
            raise ValueError(f"{path} missing 'frame_metadata_json'")
        if "provenance_json" not in f:
            raise ValueError(f"{path} missing 'provenance_json'")

        try:
            frame_metadata = json.loads(f["frame_metadata_json"][()].decode("utf-8"))
            provenance = json.loads(f["provenance_json"][()].decode("utf-8"))
        except Exception as e:
            raise ValueError(f"{path}: invalid JSON metadata: {e}") from None

        state_backups = None
        if "state_backups_json" in f:
            try:
                state_backups = json.loads(f["state_backups_json"][()].decode("utf-8"))
            except Exception as e:
                raise ValueError(f"{path}: invalid 'state_backups_json': {e}") from None

        stack = AFMImageStack(
            data=data,
            pixel_size_nm=pixel_size_nm,
            channel=channel,
            file_path=path,
            frame_metadata=frame_metadata,
        )

        stack.processed = processed
        stack.masks = masks
        if state_backups is not None:
            stack.state_backups = state_backups

        # Attach provenance and mark as bundle
        saved_prov = provenance.copy()
        saved_prov.setdefault("bundle", {}).update(
            bundle_file=str(path), bundle_type="h5"
        )
        # then replace stack.provenance wholesale
        stack.provenance = saved_prov

    return stack


def load_ome_tiff_stack(path: Path, channel: str = "height_trace") -> AFMImageStack:
    """
    Load an OME-TIFF bundle into an :class:`~playNano.afm_stack.AFMImageStack`.

    Attempts to parse OME-XML and custom metadata tags to reconstruct
    pixel size, timestamps, and provenance. Falls back gracefully if
    certain metadata are unavailable.

    Parameters
    ----------
    path : pathlib.Path
        Path to the `.ome.tif` file created by
        :func:`~playNano.io.export_data.save_ome_tiff_stack`.
    channel : str, optional
        Fallback channel name if none is found in OME metadata.

    Returns
    -------
    playNano.afm_stack.AFMImageStack
        Reconstructed AFMImageStack with:

        - ``data`` : 3D ``float32`` array ``(T, H, W)``
        - ``pixel_size_nm`` : float, derived from OME physical size
        - ``channel`` : str, from OME Channel or fallback
        - ``frame_metadata`` : list of dicts containing timestamps
        - ``provenance`` : dict reconstructed from custom or embedded tags

    Raises
    ------
    ValueError
        If the image array shape is unsupported or essential metadata
        cannot be parsed.
    """
    import xml.etree.ElementTree as ET

    # read image + ome metadata
    with tifffile.TiffFile(path) as tif:
        img = tif.asarray()
        ome_xml = tif.ome_metadata
        description_tag = tif.pages[0].tags.get("ImageDescription")
        metadata_dict = {}
        if description_tag is not None:
            try:
                metadata_dict = json.loads(description_tag.value)
            except Exception:
                metadata_dict = {}

        # Try to read the custom tag (example tag 65000)
        custom_tag_id = 65000
        custom_tag_data = None
        if custom_tag_id in tif.pages[0].tags:
            try:
                custom_tag_data = json.loads(
                    tif.pages[0].tags[custom_tag_id].value.decode("utf-8")
                )
            except Exception as e:
                logger.warning(f"Could not decode custom tag {custom_tag_id}: {e}")

        # Normalize dimensions
        if img.ndim == 5:
            data = img[:, 0, 0, :, :].astype(np.float32)
        elif img.ndim == 3:
            data = img.astype(np.float32)
        else:
            raise ValueError(f"Unexpected OME-TIFF shape: {img.shape}")

        # Defaults
        ps_nm = 1.0
        timestamps = list(range(data.shape[0]))
        channel_name = channel

        # Parse OME-XML for physical sizes, timestamps, and channel name
        try:
            root = ET.fromstring(ome_xml)
            ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

            pixels = root.find(".//ome:Pixels", namespaces=ns)
            if pixels is not None and pixels.attrib.get("PhysicalSizeX"):
                ps_nm = float(pixels.attrib["PhysicalSizeX"]) * 1e3  # µm → nm

            planes = root.findall(".//ome:Plane", namespaces=ns)
            time_points = [
                float(p.attrib.get("DeltaT", i)) for i, p in enumerate(planes)
            ]
            if time_points:
                timestamps = time_points

            channel_elem = root.find(".//ome:Channel", namespaces=ns)
            if channel_elem is not None and "Name" in channel_elem.attrib:
                channel_name = channel_elem.attrib["Name"]
        except Exception as e:
            logger.warning(f"Failed to parse OME-XML metadata for {path}: {e}")

        frame_metadata = [{"timestamp": t} for t in timestamps]

        stack = AFMImageStack(
            data=data,
            pixel_size_nm=ps_nm,
            channel=channel_name,
            file_path=path,
            frame_metadata=frame_metadata,
        )

        provenance_clean = {}
        if custom_tag_data is not None:
            provenance_clean = custom_tag_data
        elif "UserDataProvenance" in metadata_dict:
            try:
                provenance_clean = json.loads(metadata_dict["UserDataProvenance"])
            except Exception as e:
                logger.warning(f"Could not decode provenance from {path}: {e}")

        # Always add bundle info
        provenance_clean.setdefault("bundle", {}).update(
            bundle_file=str(path), bundle_type="ome-tiff"
        )
        stack.provenance = provenance_clean

        return stack
