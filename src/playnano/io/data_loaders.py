"""
Load stacks saved by playNano as NPZ, HDF5 or OME-TIFF files.

- :func:`load_npz_bundle` and :func:`load_h5_bundle` restore everything
  :mod:`playnano.io.export_data` saved: data, processed layers, masks, frame
  metadata, provenance and the saved metadata from before frame removal.
- :func:`load_ome_tiff_stack` restores the data, pixel size, frame times,
  channel name and provenance. OME-TIFF files do not hold processed layers or
  masks.

Usually these are called through :func:`playnano.io.loader.load_afm_stack` or
:meth:`playnano.afm_stack.AFMImageStack.load_data`, which choose the loader from
the file extension.
"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import tifffile

from playnano.afm_stack import AFMImageStack

logger = logging.getLogger(__name__)


def load_npz_bundle(path: Path, channel: str = "height_trace") -> AFMImageStack:
    """
    Load a stack from an NPZ bundle.

    Reads files written by :func:`~playnano.io.export_data.save_npz_bundle`.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.npz`` file.
    channel : str, optional
        Ignored; the channel saved in the bundle is used. Accepted so all
        loaders can be called the same way by
        :func:`~playnano.io.loader.load_afm_stack`.

    Returns
    -------
    AFMImageStack
        The stack, with ``processed``, ``masks``, ``frame_metadata``,
        ``state_backups`` and ``provenance`` restored.
        ``stack.provenance["bundle"]`` records the file it was read from
        (``bundle_file``) and its type (``bundle_type``).

    Raises
    ------
    KeyError
        If ``data``, ``pixel_size_nm`` or ``channel`` is missing.
    ValueError
        If ``frame_metadata_json`` or ``provenance_json`` is missing, or a JSON
        entry cannot be decoded.
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
    Load a stack from an HDF5 bundle.

    Reads files written by :func:`~playnano.io.export_data.save_h5_bundle`.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.h5`` file.
    channel : str, optional
        Ignored; the channel saved in the bundle is used. Accepted so all
        loaders can be called the same way by
        :func:`~playnano.io.loader.load_afm_stack`.

    Returns
    -------
    AFMImageStack
        The stack, with ``processed``, ``masks``, ``frame_metadata``,
        ``state_backups`` (if saved) and ``provenance`` restored.
        ``stack.provenance["bundle"]`` records the file it was read from
        (``bundle_file``) and its type (``bundle_type``).

    Raises
    ------
    KeyError
        If the ``/data`` dataset or the ``pixel_size_nm`` or ``channel``
        attribute is missing.
    ValueError
        If ``/frame_metadata_json`` or ``/provenance_json`` is missing, or a
        JSON dataset cannot be decoded.
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
    Load a stack from an OME-TIFF file.

    Made for files saved by :func:`~playnano.io.export_data.save_ome_tiff_stack`,
    but other OME-TIFF stacks with axes (T, C, Z, Y, X) or (T, Y, X) also load;
    only the first channel and Z plane are read.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.ome.tif`` file.
    channel : str, optional
        Channel name to use if the file does not name one. Default is
        ``"height_trace"``.

    Returns
    -------
    AFMImageStack
        The stack, as float32. Values are taken from the OME metadata where
        present, with these fallbacks:

        - pixel size from ``PhysicalSizeX`` (µm, converted to nm), else 1.0 nm;
        - frame times from each plane's ``DeltaT``, else the frame index;
        - provenance from TIFF tag 65000, else from a ``UserDataProvenance``
          entry in the image description, else empty.

        ``frame_metadata`` holds only ``timestamp`` and ``frame_pixel_size_nm``.
        ``stack.provenance["bundle"]`` records the file it was read from
        (``bundle_file``) and its type (``bundle_type``).

    Raises
    ------
    ValueError
        If the image does not have 3 or 5 dimensions.
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
        for frame in frame_metadata:
            frame["frame_pixel_size_nm"] = ps_nm

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
