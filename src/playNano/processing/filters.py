"""Module for applying flattening and filtering to AFM images in Numpy arrays."""

import logging

import numpy as np
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


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
    Subtract a 2D polynomial surface of given order to flatten AFM image data.

    Parameters
    ----------
    data : np.ndarray
        2D AFM image data.
    order : int
        Polynomial order for surface fitting (e.g., 1 for linear, 2 for quadratic).

    Returns
    -------
    np.ndarray
        Flattened image with polynomial background removed.

    Raises
    ------
    ValueError
        If data is not a 2D array or if order is not a positive integer.
    """
    # Validate input shape and type
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input data must be a 2D NumPy array.")
    if not isinstance(order, int) or order < 1:
        raise ValueError("Polynomial order must be a positive integer.")

    h, w = data.shape

    # Generate coordinate grid for surface fitting
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = data.astype(np.float64)

    # Prepare design matrix with all polynomial terms up to the given order
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    try:
        poly = PolynomialFeatures(order)
        A = poly.fit_transform(coords)
    except Exception as e:
        raise RuntimeError(f"Failed to generate polynomial features: {e}")

    # Solve for least-squares polynomial surface
    Zf = Z.ravel()
    try:
        coeff, _, _, _ = np.linalg.lstsq(A, Zf, rcond=None)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Least squares fitting failed: {e}")

    # Reconstruct the fitted surface and subtract it
    Z_fit = A @ coeff
    flattened = Z - Z_fit.reshape(h, w)

    return flattened


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
        unmasked = img[~mask]
        if unmasked.size == 0:
            mean_val = np.mean(img)
            raise ValueError(
                "Mask excludes all pixels — cannot compute mean. "
                "zero_mean applied without mask."
            )
        mean_val = np.mean(unmasked)
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
        "polynomial_flatten": polynomial_flatten,
        "gaussian_filter": gaussian_filter,
    }
