"""
Common loader for various high speed AFM video formats *and* playNano export bundles.

Supported extensions:
  - .jpk        (folder)
  - .spm        (folder)
  - .h5-jpk     (single-file JPK)
  - .asd        (single-file ASD)
  - .ome.tif / .tif  (OME-TIFF bundles)
  - .npz        (playNano NPZ bundles)
  - .h5         (playNano HDF5 bundles)
"""

import logging
from pathlib import Path

from playNano.afm_stack import AFMImageStack
from playNano.io.data_loaders import (
    load_h5_bundle,
    load_npz_bundle,
    load_ome_tiff_stack,
)
from playNano.io.formats.read_asd import load_asd_file
from playNano.io.formats.read_h5jpk import load_h5jpk
from playNano.io.formats.read_jpk_folder import load_jpk_folder
from playNano.io.formats.read_spm_folder import load_spm_folder

logger = logging.getLogger(__name__)


def get_loader_for_folder(
    folder_path: Path, folder_loaders: dict
) -> tuple[str, callable]:
    """
    Determine the appropriate loader for a folder containing AFM data.

    Parameters
    ----------
    folder_path : Path
        Path to the folder.
    folder_loaders : dict
        Mapping from file extension to loader function.

    Returns
    -------
    (str, callable)
        The chosen extension and loader function.

    Raises
    ------
    FileNotFoundError
        If no known file types are found.
    """
    logger.debug(f"Determining loader for folder {folder_path}")
    suffix_counts = {}
    for f in folder_path.iterdir():
        if f.is_file():
            ext = f.suffix.lower()

            # Handle 'old' nanoscope numeric extensions like .001, .002 as ".spm"
            if ext[1:].isdigit() and len(ext) == 4:
                ext = ".spm"

            if ext in folder_loaders:
                suffix_counts[ext] = suffix_counts.get(ext, 0) + 1

    if not suffix_counts:
        raise FileNotFoundError("No supported AFM files found in the folder.")

    # Prefer .jpk for now
    chosen_ext = (
        ".jpk" if ".jpk" in suffix_counts else max(suffix_counts, key=suffix_counts.get)
    )
    return chosen_ext, folder_loaders[chosen_ext]


def get_loader_for_file(
    file_path: Path, file_loaders: dict, folder_loaders: dict
) -> callable:
    """
    Determine the appropriate loader for a single multi-frame AFM file.

    Parameters
    ----------
    file_path : Path
        Path to the file.
    file_loaders : dict
        Mapping from file extensions to file loader functions.
    folder_loaders : dict
        Mapping from extensions for folder loaders (for error handling).

    Returns
    -------
    (str, callable)
        The file extention string and the loader function for the file.

    Raises
    ------
    ValueError
        If the file type is unsupported or better handled as a folder.
    """
    logger.debug(f"Determining loader for file {file_path}")
    ext = file_path.suffix.lower()
    if not ext:
        raise ValueError(f"{file_path} has no extension and cannot be identified.")

    if ext in file_loaders:
        return ext, file_loaders[ext]
    elif ext in folder_loaders:
        raise ValueError(
            f"The {ext} file type is typically a single-frame export."
            "To load HS-AFM video, pass the full folder instead."
        )
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_afm_stack(file_path: Path, channel: str = "height_trace") -> AFMImageStack:
    """
    Unified interface to load AFM stacks from various file formats.

    High speed AFM videos can be saved as either individual frames
    within a folder or as multiple frames within a single file.
    This loader splits these two approaches and loads both into
    the common AFMImageStack object for processing.

    As well as the file formats exported from AFM instruments, this
    function also read raw and processed exports from playNano (NPZ,
    OME_TIF and HDF5).

    All data values with length units (i.e. m) are converted to nm.

    Parameters
    ----------
    file_path : Path | str
        Path to the AFM data file or folder of files.
    channel : str
        Scan channel name.

    Returns
    -------
    AFMImageStack
        Loaded image stack with metadata.
    """
    logger.debug(f"Raw input path: {file_path}")
    file_path = Path(file_path).resolve()
    logger.debug(f"Resolved path: {file_path}")
    logger.debug(f"Loading AFM stack from {file_path} for channel '{channel}'")

    folder_loaders = {
        ".jpk": load_jpk_folder,
        ".spm": load_spm_folder,
        # Add others as needed
    }

    file_loaders = {
        ".h5-jpk": load_h5jpk,
        ".asd": load_asd_file,
        ".npz": load_npz_bundle,
        ".h5": load_h5_bundle,
        ".ome.tif": load_ome_tiff_stack,
        ".tif": load_ome_tiff_stack,
        # Add others as needed
    }

    # Load folder
    if file_path.is_dir():
        logger.debug(f"Loading folder {file_path}")
        ext, loader = get_loader_for_folder(file_path, folder_loaders)
        logger.debug(
            f"Loading folder {file_path} with loader {loader.__name__} for extension {ext}"  # noqa: E501
        )
        return loader(file_path, channel=channel)

    # Load file
    elif file_path.is_file():
        logger.debug(f"Loading file {file_path}")
        ext, loader = get_loader_for_file(file_path, file_loaders, folder_loaders)
        logger.debug(
            f"Loading file {file_path} with loader {loader.__name__} for extension {ext}"  # noqa: E501
        )

        return loader(file_path, channel=channel)

    else:
        raise FileNotFoundError(f"{file_path} is neither a file nor a directory.")
