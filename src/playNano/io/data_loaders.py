"""
Data loaders for stacks exported by playNano.

This module provides readers for AFMImageStack bundles created
by our export routines (`.npz`, `.h5`, and OME-TIFF). Each loader
will reconstruct an AFMImageStack with correct data, pixel size,
channel, and per-frame metadata (timestamps).

Functions
---------
load_npz_bundle
    Load a `.npz` bundle into an AFMImageStack.
load_h5_bundle
    Load an HDF5 bundle into an AFMImageStack.
load_ome_tiff_stack
    Load an OME-TIFF bundle into an AFMImageStack.
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
    Load an AFMImageStack from a `.npz` bundle produced by `save_npz_bundle`.

    Expects keys:
      - data               : float32 (n_frames, H, W)
      - pixel_size_nm      : float32 scalar
      - channel            : object scalar
      - frame_metadata_json: object scalar (JSON list of dict)
      - provenance_json    : object scalar (JSON dict)
      - processed__<step>  : float32 arrays
      - masks__<mask>      : boolean arrays

    Parameters
    ----------
    path : Path
        Path to the `.npz` file.
    channel: str
        For being called by load_afm_stack but ignored. Default is "height_trace".

    Returns
    -------
    AFMImageStack
        Reconstructed stack, with `.processed`, `.masks`, and `.provenance` filled.

    Raises
    ------
    ValueError
        If required arrays (`frame_metadata_json` or `provenance_json`) are missing
        or contain invalid JSON.

    """
    arrs = np.load(str(path), allow_pickle=True)

    # core
    data = arrs["data"]
    pixel_size_nm = float(arrs["pixel_size_nm"].item())
    # timestamps = list(arrs["timestamps"]) # read from frame metadata
    channel = str(arrs["channel"].item())

    # metadata blobs
    try:
        raw_meta = arrs["frame_metadata_json"].item()
    except KeyError:
        raise ValueError(
            f"{path} is not a playNano NPZ bundle (missing frame_metadata_json)"
        ) from None

    try:
        frame_metadata = json.loads(raw_meta)
    except Exception as e:
        raise ValueError(
            f"{path}: invalid JSON in 'frame_metadata_json': {e}"
        ) from None

    try:
        raw_prov = arrs["provenance_json"].item()
    except KeyError:
        raise ValueError(
            f"{path} is not a playNano NPZ bundle (missing provenance_json)"
        ) from None

    try:
        provenance = json.loads(raw_prov)
    except Exception as e:
        raise ValueError(f"{path}: invalid JSON in 'provenance_json': {e}") from None

    # build stack
    stack = AFMImageStack(
        data=data,
        pixel_size_nm=pixel_size_nm,
        channel=channel,
        file_path=path,
        frame_metadata=frame_metadata,
    )

    # Mark that this came from an export bundle
    # first, load the saved provenance
    saved_prov = provenance.copy()
    # annotate bundle info
    saved_prov.setdefault("bundle", {}).update(bundle_file=str(path), bundle_type="npz")
    # then replace stack.provenance wholesale
    stack.provenance = saved_prov

    # extract processed & masks
    for key in arrs.files:
        if key.startswith("processed__"):
            step = key.split("__", 1)[1]
            stack.processed[step] = arrs[key].astype(np.float32)
        elif key.startswith("masks__"):
            mask = key.split("__", 1)[1]
            stack.masks[mask] = arrs[key].astype(bool)

    # overwrite data if last processed step should become current?
    # (optional; typically leave data==raw)
    return stack


def load_h5_bundle(path: Path, channel: str = "height_trace") -> AFMImageStack:
    """
    Load an AFMImageStack from a `.h5` bundle produced by ``save_h5_bundle``.

    Expected groups
    ---------------
    - ``/data`` : float32, shape (n_frames, H, W)
    - ``/processed/<step>`` : float32 subgroups
    - ``/masks/<mask>`` : boolean subgroups
    - ``/timestamps`` : float64, shape (n_frames,)
    - ``/frame_metadata_json`` : UTF-8 string (JSON list of dict)
    - ``/provenance_json`` : UTF-8 string (JSON dict)

    File attributes
    ---------------
    - ``pixel_size_nm`` : float
    - ``channel`` : UTF-8 string

    Parameters
    ----------
    path : Path
        Path to the `.h5` file.

    channel : str, default="height_trace"
        Required for compatibility with ``load_afm_stack`` but ignored.

    Returns
    -------
    AFMImageStack
        Reconstructed stack, with ``.processed``, ``.masks``,
        and ``.provenance`` filled.

    Raises
    ------
    ValueError
        If required datasets (``frame_metadata_json`` or ``provenance_json``)
        are missing or contain invalid JSON.
    """
    with h5py.File(str(path), "r") as f:
        data = f["data"][()].astype(np.float32)
        pixel_size_nm = float(f.attrs["pixel_size_nm"])
        channel = str(f.attrs["channel"])

        # Load all processed snapshots
        processed = {}
        if "processed" in f:
            for name, ds in f["processed"].items():
                processed[name] = ds[()].astype(np.float32)

        # Load masks
        masks = {}
        if "masks" in f:
            for name, ds in f["masks"].items():
                masks[name] = ds[()].astype(bool)

        # timestamps = list(f["timestamps"][()])    # Read from frame metadata

        # Metadata
        if "frame_metadata_json" not in f:
            raise ValueError(
                f"{path} is not a playNano HDF5 bundle (missing 'frame_metadata_json')"
            )
        raw_meta = f["frame_metadata_json"][()]
        try:
            frame_metadata = json.loads(raw_meta.decode("utf-8"))
        except Exception as e:
            raise ValueError(
                f"{path}: invalid JSON in 'frame_metadata_json': {e}"
            ) from None

        if "provenance_json" not in f:
            raise ValueError(
                f"{path} is not a playNano HDF5 bundle (missing 'provenance_json')"
            )
        raw_prov = f["provenance_json"][()]
        try:
            provenance = json.loads(raw_prov.decode("utf-8"))
        except Exception as e:
            raise ValueError(
                f"{path}: invalid JSON in 'provenance_json': {e}"
            ) from None

        # Construct AFMImageStack
        stack = AFMImageStack(
            data=data,
            pixel_size_nm=pixel_size_nm,
            channel=channel,
            file_path=path,
            frame_metadata=frame_metadata,
        )

        stack.processed = processed
        stack.masks = masks

        # Attach provenance and mark as bundle
        saved_prov = provenance.copy()
        # annotate bundle info
        saved_prov.setdefault("bundle", {}).update(
            bundle_file=str(path), bundle_type="h5"
        )
        # then replace stack.provenance wholesale
        stack.provenance = saved_prov

    return stack


def load_ome_tiff_stack(path: Path, channel: str = "height_trace") -> AFMImageStack:
    """
    Load an OME-TIFF bundle into an AFMImageStack.

    Parameters
    ----------
    path : Path
        Path to the `.ome.tif` file produced by `save_ome_tiff_stack`.
    channel : str, optional
        Fallback channel name if none is found in OME metadata.

    Returns
    -------
    AFMImageStack
        Reconstructed AFMImageStack with:
        - `data`: 3D float32 array `(T, H, W)` or 5D float32 array `(T, 1, 1, H, W)`
        - `pixel_size_nm`: float (converted from µm metadata if available)
        - `channel`: first entry of `ChannelName` OME tag or the `channel` parameter
        - `frame_metadata`: list of dicts with `"timestamp"` if available

    Raises
    ------
    ValueError
        If required datasets (`frame_metadata_json` or `provenance_json`) are missing
        or contain invalid JSON.
    """
    import json
    import xml.etree.ElementTree as ET

    # read image + ome metadata
    with tifffile.TiffFile(path) as tif:
        img = tif.asarray()
        ome_xml = tif.ome_metadata

        # Try to read ImageDescription tag as JSON metadata fallback
        description_tag = tif.pages[0].tags.get("ImageDescription", None)
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
            tag = tif.pages[0].tags[custom_tag_id]
            try:
                # The tag value is bytes, decode and parse JSON
                custom_tag_data = json.loads(tag.value.decode("utf-8"))
            except Exception as e:
                logger.warning(
                    f"Could not decode custom tag {custom_tag_id} from {path}: {e}"
                )

        # Handle array dimensions
        if img.ndim == 5:
            data = img[:, 0, 0, :, :].astype(np.float32)
        elif img.ndim == 3:
            data = img.astype(np.float32)
        else:
            raise ValueError(f"Unexpected OME-TIFF array shape: {img.shape}")

        # Defaults
        ps_nm = 1.0
        timestamps = list(range(data.shape[0]))
        channel_name = channel

        # Parse OME-XML for physical sizes, timestamps, and channel name
        try:
            root = ET.fromstring(ome_xml)
            ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
            pixels = root.find(".//ome:Pixels", namespaces=ns)
            if pixels is not None:
                ps_x = pixels.attrib.get("PhysicalSizeX")
                if ps_x is not None:
                    ps_nm = float(ps_x) * 1e3  # µm → nm

            time_points = [
                float(t.attrib.get("DeltaT", i))
                for i, t in enumerate(root.findall(".//ome:Plane", namespaces=ns))
            ]
            if time_points:
                timestamps = time_points
            else:
                logger.info("Could not read timestamps, defaulting to frame indices.")
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

        # Recover provenance JSON from custom tag or fallback to
        # ImageDescription metadata
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
