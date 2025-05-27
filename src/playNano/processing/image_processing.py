"""Module for applying flattening to AFM images in Numpy arrays"""
import logging

import numpy as np
from topostats.filters import Filters

logger = logging.getLogger(__name__)


def flatten_afm_frame(
    frame: np.ndarray,
    filename: str = "frame",
    pixel_to_nm: float = 1.0,
    filter_config: dict = None,
) -> np.ndarray:
    """
    Apply a filtering pipeline to flatten an AFM image frame using TopoStats filters.

    Parameters
    ----------

frame : np.ndarray
        2D NumPy array representing the AFM image frame.
    filename : str, optional
        Name of the frame, used for logging and identification. Default is "frame".
    pixel_to_nm : float, optional
        Scaling factor to convert pixels to nanometers. Default is 1.0.
    filter_config : dict, optional
        Dictionary of filter configuration parameters.
        If None, a default configuration is used.

    Returns
    -------
    np.ndarray or None
        The flattened image after Gaussian filtering. Returns None if the
        input is invalid or if filtering fails.

    Notes
    -----
    - Uses the `Filters` class from `topostats.filters` to apply a series
    of image processing steps.
    - Handles invalid input (e.g., empty arrays or NaNs) gracefully.
    - Logs detailed information about the filtering process.
    """
    if filter_config is None:
        filter_config = {
            "row_alignment_quantile": 0.05,
            "threshold_method": "std_dev",
            "otsu_threshold_multiplier": 1.0,
            "threshold_std_dev": {"above": 0.5, "below": 10},
            "threshold_absolute": None,
            "gaussian_size": 2,
            "gaussian_mode": "nearest",
            "remove_scars": {"run": False},
        }

    try:
        logger.info(
            f"Flattening frame '{filename}': shape={frame.shape}, dtype={frame.dtype}"
        )

        if frame.size == 0 or np.isnan(frame).any():
            logger.warning(
                f"Input frame invalid for {filename}: empty or contains NaNs"
            )
            return None

        filters = Filters(
            image=frame,
            filename=filename,
            pixel_to_nm_scaling=pixel_to_nm,
            **filter_config,
        )
        filters.filter_image()  # run filtering pipeline

        gaussian_filtered = filters.images.get("gaussian_filtered", None)
        if gaussian_filtered is None:
            logger.warning(
                f"Warning: 'gaussian_filtered' image not"
                f"found in filters.images for {filename}"
            )
            return None

        return gaussian_filtered

    except Exception as e:
        logger.warning(f"Exception flattening frame {filename}: {e}")
        return None


def flatten_stack(image_stack: np.ndarray, pixel_to_nm: float = 1.0) -> np.ndarray:
    """
    Apply AFM frame flattening to a stack of images.

    Parameters
    ----------
    image_stack : np.ndarray
        3D NumPy array of shape (N, H, W), where N is the number of frames.
    pixel_to_nm : float, optional
        Scaling factor to convert pixels to nanometers. Default is 1.0.

    Returns
    -------
    np.ndarray
        3D NumPy array of flattened frames with the same shape as the input stack.

    Notes
    -----
    - Internally calls `flatten_afm_frame` on each frame in the stack.
    - Frames that fail to flatten (e.g., due to invalid input) will
    result in None and may raise an error during stacking.
    """
    return np.stack(
        [
            flatten_afm_frame(frame, filename=f"frame_{i}", pixel_to_nm=pixel_to_nm)
            for i, frame in enumerate(image_stack)
        ]
    )
