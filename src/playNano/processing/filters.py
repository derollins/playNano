"""Module for applying flattening and filtering to AFM images in Numpy arrays."""

import logging

import numpy as np
from scipy import ndimage
from sklearn.linear_model import LinearRegression
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


def row_median_align(data: np.ndarray) -> np.ndarray:
    """
    Subtract the median of each row from that row to remove horizontal banding.

    Parameters
    ----------
    data : np.ndarray
        2D AFM image data.

    Returns
    -------
    np.ndarray
        Row-aligned image.
    """
    aligned = data.astype(np.float64).copy()
    # Compute median for each row
    medians = np.median(aligned, axis=1)
    # Subtract median from each row
    aligned = aligned - medians[:, np.newaxis]
    return aligned


def remove_plane(data: np.ndarray) -> np.ndarray:
    """
    Fit a 2D plane to the image using linear regression and subtract it.

    Uses a 2D plane (z = ax + by + c) to remove to remove overall tilt.

    Parameters
    ----------
    data : np.ndarray
        2D AFM image data.

    Returns
    -------
    np.ndarray
        Plane-removed image.
    """
    h, w = data.shape
    # Create coordinate grids
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = data.astype(np.float64)
    # Flatten arrays for regression
    Xf = X.ravel()
    Yf = Y.ravel()
    Zf = Z.ravel()
    # Stack X and Y as features
    features = np.vstack((Xf, Yf)).T
    # Fit linear regression model
    model = LinearRegression()
    model.fit(features, Zf)
    # Predict plane values
    plane = model.predict(features).reshape(h, w)
    return data - plane


def polynomial_flatten(data: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Fit and subtract a 2D polynomial of given order to remove slow surface trends.

    Parameters
    ----------
    data : np.ndarray
        2D AFM image data (should be already plane-removed for best results).
    order : int
        Order of the polynomial (currently supports order=2 for quadratic).
        Default is 2.

    Returns
    -------
    np.ndarray
        Polynomial-flattened image.
    """
    if order != 2:
        raise ValueError("Currently only quadratic (order=2) flattening is supported.")
    h, w = data.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = data.astype(np.float64)
    # Flatten
    Xf = X.ravel()
    Yf = Y.ravel()
    Zf = Z.ravel()
    # Design matrix for quadratic: [1, x, y, x^2, x*y, y^2]
    A = np.stack([np.ones_like(Xf), Xf, Yf, Xf**2, Xf * Yf, Yf**2], axis=1)
    # Solve least squares for coefficients
    coeff, _, _, _ = np.linalg.lstsq(A, Zf, rcond=None)
    # Compute fitted surface
    Z_fit = (
        coeff[0]
        + coeff[1] * X
        + coeff[2] * Y
        + coeff[3] * X**2
        + coeff[4] * X * Y
        + coeff[5] * Y**2
    )
    return data - Z_fit


def zero_mean(data: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Subtract the overall mean height to center the background around zero.
    If a mask is provided, mean is computed only over background (mask == False).

    Parameters
    ----------
    data : np.ndarray
        2D AFM image data.
    mask : np.ndarray, optional
        Boolean mask of same shape as data; True indicates region to exclude from mean.

    Returns
    -------
    np.ndarray
        Zero-mean image.
    """
    img = data.astype(np.float64).copy()
    if mask is None:
        mean_val = np.mean(img)
    else:
        if mask.shape != img.shape:
            raise ValueError("Mask must have same shape as data.")
        # Compute mean over background (where mask is False)
        mean_val = np.mean(img[~mask])
    return img - mean_val


def gaussian_filter(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply a Gaussian low-pass filter to smooth high-frequency noise.

    Parameters
    ----------
    data : np.ndarray
        2D AFM image data.
    sigma : float
        Standard deviation for Gaussian kernel, in pixels.

    Returns
    -------
    np.ndarray
        Smoothed image.
    """
    return ndimage.gaussian_filter(data, sigma=sigma)


def register_filters():
    """Return list of filter options."""
    return {
        "remove_plane": remove_plane,
        "row_median_align": row_median_align,
        "zero_mean": zero_mean,
        "topostats_flatten": topostats_flatten,
        "polynomial_flatten": polynomial_flatten,
        "gaussian_filter": gaussian_filter,
    }
