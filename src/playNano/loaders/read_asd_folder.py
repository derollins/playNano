"""
Placeholder for a functional script to load a folder containing .asd frames
of a high speed AFM video.
"""

import logging
from pathlib import Path

from playNano.stack.image_stack import AFMImageStack

logger = logging.getLogger(__name__)


def _extract_scan_rate(asd_file: Path) -> float:
    """
    Extract the scan rate in lines per second from a .asd image file.

    Parameters
    ----------
    asd_file : Path
        Path to a .asd file.

    Returns
    -------
    float
        The scan rate of the image in fast scan lines per second.
    """
    asd_scan_rate = None
    return asd_scan_rate


def load_asd_folder(folder_path: Path | str, channel: str) -> AFMImageStack:
    """
        Load an AFM video from a folder of individual .asd image files.

        Parameters
        ----------
        folder_path : Path | str
            Path to folder containing .asd files.
        channel : str
            Channel to extract.

        Returns
        -------
        AFMImageStack
            Loaded AFM image stack with metadata and per-frame info.

        folder = Path(folder_path)
        if not folder.is_dir():
            raise ValueError(f"{folder} is not a directory.")

        asd_files = sorted(folder.glob("*.asd"))

        if not asd_files:
            raise FileNotFoundError(f"No .asd files found in {folder}.")

        logger.info(f"Found {len(asd_files)} .asd files.")

        # Load first image to get shape and pixel size
        first_img, first_pixel_size_nm = load_asd(asd_files[0], channel)
        height_px, width_px = first_img.shape
        dtype = first_img.dtype

        # Preallocate image stack
        num_frames = len(asd_files)
        image_stack = np.empty((num_frames, height_px, width_px), dtype=dtype)

        # Extract metadata from first image

        #Line rate and timestamps
        line_rate = _extract_scan_rate(asd_files[0]) # lines per second
        lines_per_frame = height_px # number of fast scan lines in an image
        frame_rate = line_rate / lines_per_frame # frames per second
        frame_interval = 1.0 / frame_rate # time taken per frame
        timestamps = np.arange(num_frames) * frame_interval

        # Load all images
        for i, fpath in enumerate(asd_files):
            logger.debug(f"Loading {fpath.name}")
            img, px_size_nm= load_asd(fpath, channel)
            if img.shape != (height_px, width_px):
                raise ValueError(f"Inconsistent image shape in {fpath}")
            if not np.isclose(px_size_nm, first_pixel_size_nm):
                raise ValueError(f"Inconsistent pixel size in {fpath}")
            image_stack[i] = img

        frame_metadata = [{'timestamp': ts} for ts in timestamps]

        return AFMImageStack(
            image_stack=image_stack,
            pixel_size_nm=first_pixel_size_nm,
            img_shape=(height_px, width_px),
            line_rate=line_rate, # in lines per second
            channel= channel,
            file_path=str(folder),
            frame_metadata=frame_metadata,
    )
    """
    raise ValueError(".asd not yet supported. Raise an issue on GitHub.")
