from pathlib import Path
import os

from playNano.loaders.read_h5jpk import load_h5jpk
from playNano.stack.image_stack import AFMImageStack

# Loaders:
from playNano.loaders.read_jpk_folder import load_jpk_folder
from playNano.loaders.read_asd_folder import load_asd_folder
from playNano.loaders.read_spm_folder import load_spm_folder

# from playNano.loaders.read_asd import load_asd  # future reader


def get_loader_for_folder(
    folder_path: Path, folder_loaders: dict
) -> tuple[str, callable]:
    """
    Determines the appropriate loader for a folder containing AFM data.

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
    suffix_counts = {}
    for f in folder_path.iterdir():
        if f.is_file():
            ext = f.suffix.lower()
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
    Determines the appropriate loader for a single multi-frame AFM file.

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
    callable
        The loader function for the file.

    Raises
    ------
    ValueError
        If the file type is unsupported or better handled as a folder.
    """
    ext = file_path.suffix.lower()
    if ext in file_loaders:
        return file_loaders[ext]
    elif ext in folder_loaders:
        raise ValueError(
            f"The {ext} file type is typically a single-frame export. To load HS-AFM video, pass the full folder instead."
        )
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_afm_stack(
    file_path: Path | str, channel: str = "height_trace"
) -> AFMImageStack:
    """
    Unified interface to load AFM stacks from various file formats.

    High speed AFM videos can be saved as either individual frames within a folder or as multiple frames within a single file.
    This loader splits these two approaches and loads both into the common AFMImageStack object for processing.

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
    file_path = Path(file_path)

    folder_loaders = {
        ".jpk": load_jpk_folder,
        ".asd": load_asd_folder,
        ".spm": load_spm_folder,
    }

    file_loaders = {
        ".h5-jpk": load_h5jpk,
        # Add others as needed
    }

    # Load folder
    if file_path.is_dir():
        ext, loader = get_loader_for_folder(file_path, folder_loaders)
        return loader(file_path, channel=channel)

    # Load file
    elif file_path.is_file():
        loader = get_loader_for_file(file_path, file_loaders, folder_loaders)
        return loader(file_path, channel=channel)

    else:
        raise FileNotFoundError(f"{file_path} is neither a file nor a directory.")
