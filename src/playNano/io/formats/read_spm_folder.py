"""Placeholder for a functional script to load a folder containing .spm frames."""

import logging
from pathlib import Path

from playNano.afm_stack import AFMImageStack

logger = logging.getLogger(__name__)


def _extract_scan_rate(spm_file: Path) -> float:
    """
    Extract the scan rate in lines per second from a .spm image file.

    Parameters
    ----------
    spm_file : Path
        Path to a .spm file.

    Returns
    -------
    float
        The scan rate of the image in fast scan lines per second.
    """
    spm_scan_rate = None
    return spm_scan_rate


def load_spm_folder(folder_path: Path | str, channel: str) -> AFMImageStack:
    """
        Load an AFM video from a folder of individual .spm image files.

        Parameters
        ----------
        folder_path : Path | str
            Path to folder containing .spm files.
        channel : str
            Channel to extract.

        Returns
        -------
        AFMImageStack
            Loaded AFM image stack with metadata and per-frame info.

        folder = Path(folder_path)
        if not folder.is_dir():
            raise ValueError(f"{folder} is not a directory.")

        spm_files = sorted(folder.glob("*.spm"))

        if not spm_files:
            raise FileNotFoundError(f"No .spm files found in {folder}.")

        logger.info(f"Found {len(spm_files)} .spm files.")

        # Load first image to get shape and pixel size
        first_img, first_pixel_size_nm = load_spm(spm_files[0], channel)
        height_px, width_px = first_img.shape
        dtype = first_img.dtype

        # Preallocate image stack
        num_frames = len(spm_files)
        image_stack = np.empty((num_frames, height_px, width_px), dtype=dtype)

        # Extract metadata from first image

        #Line rate and timestamps
        line_rate = _extract_scan_rate(spm_files[0]) # lines per second
        lines_per_frame = height_px # number of fast scan lines in an image
        frame_rate = line_rate / lines_per_frame # frames per second
        frame_interval = 1.0 / frame_rate # time taken per frame
        timestamps = np.arange(num_frames) * frame_interval

        # Load all images
        for i, fpath in enumerate(spm_files):
            logger.debug(f"Loading {fpath.name}")
            img, px_size_nm= load_spm(fpath, channel)
            if img.shape != (height_px, width_px):
                raise ValueError(f"Inconsistent image shape in {fpath}")
            if not np.isclose(px_size_nm, first_pixel_size_nm):
                raise ValueError(f"Inconsistent pixel size in {fpath}")
            image_stack[i] = img

        # Compose per-frame metadata list
        frame_metadata = []
        for ts in timestamps:
            frame_metadata.append({
                "timestamp": ts,
                "line_rate": line_rate
            })

        return AFMImageStack(
            data=image_stack,
            pixel_size_nm=first_pixel_size_nm,
            channel= channel,
            file_path=str(folder),
            frame_metadata=frame_metadata,
    )
    """
    raise ValueError(".spm not yet supported. Raise an issue on GitHub.")
