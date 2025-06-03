"""Module for applying flattening and filtering to AFM images in Numpy arrays."""

import logging

import numpy as np
from scipy.ndimage import gaussian_filter as scipy_gaussian
from scipy.ndimage import median_filter as scipy_median
from topostats.filters import Filters

logger = logging.getLogger(__name__)


def topostats_flatten(
    frame: np.ndarray,
    filename: str = "frame",
    pixel_to_nm: float = 1.0,
    filter_config: dict = None,
) -> np.ndarray:
    """
    Apply a filtering pipeline to flatten an AFM image frame.

    Uses TopoStats filters to flatten the AFM image frames held with the Numpy array.

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


def flatten_poly(frame: np.ndarray, degree: int = 1) -> np.ndarray:
    """
    Remove background by polynomial surface fitting.

    Parameters
    ----------
    frame : np.ndarray
        2D AFM image frame.
    degree : int, optional
        Degree of polynomial fit (default is 1 for linear).

    Returns
    -------
    np.ndarray
        Flattened frame with background subtracted.
    """

    x, y = np.meshgrid(np.arange(frame.shape[1]), np.arange(frame.shape[0]))
    xx = x.ravel()
    yy = y.ravel()
    zz = frame.ravel()

    if degree == 1:
        A = np.c_[np.ones(xx.shape), xx, yy]
    elif degree == 2:
        A = np.c_[np.ones(xx.shape), xx, yy, xx**2, xx * yy, yy**2]
    else:
        raise ValueError("Only degree 1 or 2 supported.")

    coeffs, *_ = np.linalg.lstsq(A, zz, rcond=None)
    fitted = A @ coeffs
    return frame - fitted.reshape(frame.shape)


def median_filter(frame: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Reduce noise using median filtering.

    Parameters
    ----------
    frame : np.ndarray
        2D AFM image frame.
    size : int, optional
        Size of the filter window (default is 3).

    Returns
    -------
    np.ndarray
        Denoised frame.
    """
    return scipy_median(frame, size=size)


def gaussian_filter(frame: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Smooth image using Gaussian filtering.

    Parameters
    ----------
    frame : np.ndarray
        2D AFM image frame.
    sigma : float, optional
        Standard deviation for Gaussian kernel (default is 1.0).

    Returns
    -------
    np.ndarray
        Smoothed frame.
    """
    return scipy_gaussian(frame, sigma=sigma)


def register_filters():
    return {
        "topostats_flatten": topostats_flatten,
        "median_filter": median_filter,
        "gaussian_filter": gaussian_filter,
        "flatten_poly": flatten_poly,
    }
