import logging
from pathlib import Path

import numpy as np
import tifffile
from AFMReader.jpk import load_jpk

from playNano.stack.image_stack import AFMImageStack

logger = logging.getLogger(__name__)


def _extract_scan_rate(jpk_file: Path) -> float:
    """
    Extract the scan rate in lines per second from a .jpk image file.

    Parameters
    ----------
    jpk_file : Path
        Path to a .jpk file.

    Returns
    -------
    float
        The scan rate of the image in fast scan lines per second.
    """
    with tifffile.TiffFile(jpk_file) as tif:
        return (
            tif.pages[0].tags["32841"].value
        )  # Return the Scan Rate attribute from the tiff tag value.


def load_jpk_folder(
    folder_path: Path | str, channel: str, flip_image: bool = True
) -> AFMImageStack:
    """
    Load an AFM video from a folder of individual .jpk image files.

    Parameters
    ----------
    folder_path : Path | str
        Path to folder containing .jpk files.
    channel : str
        Channel to extract.
    flip_image : bool, optional
        Flip each image vertically if True.

    Returns
    -------
    AFMImageStack
        Loaded AFM image stack with metadata and per-frame info.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a directory.")

    jpk_files = sorted(folder.glob("*.jpk"))

    if not jpk_files:
        raise FileNotFoundError(f"No .jpk files found in {folder}.")

    logger.info(f"Found {len(jpk_files)} .jpk files.")

    # Load first image to get shape and pixel size
    first_img, first_pixel_size_nm = load_jpk(jpk_files[0], channel)
    if flip_image:
        first_img = np.flipud(first_img)
    height_px, width_px = first_img.shape
    dtype = first_img.dtype

    # Preallocate image stack
    num_frames = len(jpk_files)
    image_stack = np.empty((num_frames, height_px, width_px), dtype=dtype)

    # Extract metadata from first image

    # Line rate and timestamps
    line_rate = _extract_scan_rate(jpk_files[0])  # lines per second
    lines_per_frame = height_px  # number of fast scan lines in an image
    frame_rate = line_rate / lines_per_frame  # frames per second
    frame_interval = 1.0 / frame_rate  # time taken per frame
    timestamps = np.arange(num_frames) * frame_interval

    # Load all images
    for i, fpath in enumerate(jpk_files):
        logger.debug(f"Loading {fpath.name}")
        img, px_size_nm = load_jpk(fpath, channel)
        if img.shape != (height_px, width_px):
            raise ValueError(f"Inconsistent image shape in {fpath}")
        if not np.isclose(px_size_nm, first_pixel_size_nm):
            raise ValueError(f"Inconsistent pixel size in {fpath}")
        if flip_image:
            img = np.flipud(img)
        image_stack[i] = img

    frame_metadata = [{"timestamp": ts} for ts in timestamps]

    return AFMImageStack(
        image_stack=image_stack,
        pixel_size_nm=first_pixel_size_nm,
        img_shape=(height_px, width_px),
        line_rate=line_rate,  # in lines per second
        channel=channel,
        file_path=str(folder),
        frame_metadata=frame_metadata,
    )
