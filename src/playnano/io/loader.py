"""
Load AFM videos from instrument files and from playNano's own exports.

:func:`load_afm_stack` picks a reader from the file extension:

- Folders of frames: ``.jpk`` (:mod:`~playnano.io.formats.read_jpk_folder`) and
  ``.spm`` or numbered ``.001``, ``.002`` ... files
  (:mod:`~playnano.io.formats.read_spm_folder`).
- Single files: ``.h5-jpk`` (:mod:`~playnano.io.formats.read_h5jpk`), ``.asd``
  (:mod:`~playnano.io.formats.read_asd`) and ``.aris``
  (:mod:`~playnano.io.formats.read_aris`).
- playNano exports: ``.npz``, ``.h5``, and ``.ome.tif``, ``.ome.tiff``,
  ``.tif`` or ``.tiff`` (:mod:`~playnano.io.data_loaders`).

Extensions are matched without regard to case.
"""

import logging
from pathlib import Path

from playnano.afm_stack import AFMImageStack
from playnano.io.data_loaders import (
    load_h5_bundle,
    load_npz_bundle,
    load_ome_tiff_stack,
)
from playnano.io.formats.read_aris import load_aris
from playnano.io.formats.read_asd import load_asd_file
from playnano.io.formats.read_h5jpk import load_h5jpk
from playnano.io.formats.read_jpk_folder import load_jpk_folder
from playnano.io.formats.read_spm_folder import load_spm_folder

logger = logging.getLogger(__name__)


def get_loader_for_folder(
    folder_path: Path, folder_loaders: dict
) -> tuple[str, callable]:
    """
    Choose the folder reader for a folder of frame files.

    Parameters
    ----------
    folder_path : pathlib.Path
        The folder.
    folder_loaders : dict
        Maps each extension (lower case, with the dot) to its reader.

    Returns
    -------
    tuple of (str, callable)
        The extension and its reader. If the folder holds several known types,
        ``.jpk`` is chosen if present, otherwise the most common. Numbered
        extensions such as ``.001`` count as ``.spm``.

    Raises
    ------
    FileNotFoundError
        If the folder contains no file of a known type.
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
    Choose the reader for a single file.

    Parameters
    ----------
    file_path : pathlib.Path
        The file.
    file_loaders : dict
        Maps each extension (lower case, with the dot) to its reader. Combined
        extensions such as ``.ome.tif`` are checked first.
    folder_loaders : dict
        Extensions of folder formats, used to give a helpful error when a single
        frame of a folder format is passed.

    Returns
    -------
    tuple of (str, callable)
        The extension and its reader.

    Raises
    ------
    ValueError
        If the file has no extension, an unknown one, or one that belongs to a
        folder format (for example a single ``.jpk`` frame).
    """
    logger.debug(f"Determining loader for file {file_path}")

    suffixes = [s.lower() for s in file_path.suffixes]

    # --- Case 1: no suffix at all ---
    if not suffixes:
        raise ValueError(f"{file_path} has no extension and cannot be identified.")

    # --- Case 2: multi-suffix formats like .ome.tif ---
    multi = "".join(suffixes)  # e.g. [".ome", ".tif"] -> ".ome.tif"
    if multi in file_loaders:
        return multi, file_loaders[multi]

    # --- Case 3: single suffix fallback ---
    ext = suffixes[-1]  # always safe now
    if ext in file_loaders:
        return ext, file_loaders[ext]

    # --- Case 4: folder-only formats ---
    if ext in folder_loaders:
        raise ValueError(
            f"The {ext} file type is typically a single-frame export."
            "To load HS-AFM video, pass the full folder instead."
        )

    # --- Case 5: unsupported extension ---
    raise ValueError(f"Unsupported file type: {ext}")


def load_afm_stack(file_path: Path, channel: str = "height_trace") -> AFMImageStack:
    """
    Load an AFM video from a file or a folder of frame files.

    The reader is chosen from the extension (see the module description).
    Instrument readers return heights in nm for length channels; see each
    reader for details.

    Parameters
    ----------
    file_path : str or pathlib.Path
        The file, or the folder of frame files.
    channel : str, optional
        Channel to load, for example ``"height_trace"`` for ``.h5-jpk`` files.
        Channel names depend on the format. Ignored for NPZ and HDF5 bundles.
        Default is ``"height_trace"``.

    Returns
    -------
    AFMImageStack
        The loaded stack.

    Raises
    ------
    FileNotFoundError
        If the path does not exist, or a folder contains no known file type.
    ValueError
        If the file type is not supported.

    See Also
    --------
    playnano.afm_stack.AFMImageStack.load_data : The usual way to call this.
    """
    logger.debug(f"Raw input path: {file_path}")
    file_path = Path(file_path).resolve()
    logger.debug(f"Resolved path: {file_path}")
    logger.debug(f"Loading AFM stack from {file_path} for channel '{channel}'")

    # All file formats in lowercase
    folder_loaders = {
        ".jpk": load_jpk_folder,
        ".spm": load_spm_folder,
        # Add others as needed
    }

    file_loaders = {
        ".h5-jpk": load_h5jpk,
        ".asd": load_asd_file,
        ".aris": load_aris,
        ".npz": load_npz_bundle,
        ".h5": load_h5_bundle,
        ".ome.tif": load_ome_tiff_stack,
        ".tif": load_ome_tiff_stack,
        ".ome.tiff": load_ome_tiff_stack,
        ".tiff": load_ome_tiff_stack,
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
