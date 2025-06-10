"""Module for filtering AFM data in NumPy arrays with a boolean mask."""

import logging

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import playNano.processing.filters as filters

logger = logging.getLogger(__name__)


def remove_plane_masked(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fit a 2D plane on background only and subtract it from the full image.

    Parameters
    ----------
    data : np.ndarray
        2D AFM image.
    mask : np.ndarray
        Boolean mask of same shape; True=foreground (excluded),
        False=background (used to fit).

    Returns
    -------
    np.ndarray
        Plane-removed image.

    Raises
    ------
    ValueError
        If mask.shape != data.shape.
    """
    if mask.shape != data.shape:
        raise ValueError("Mask must have same shape as data.")
    bg_idx = ~mask.ravel()
    if np.sum(bg_idx) < 3:
        raise ValueError("Not enough background pixels to fit a plane.")

    h, w = data.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = data.astype(np.float64)
    features = np.vstack((X.ravel(), Y.ravel())).T
    Zf = Z.ravel()

    model = LinearRegression()
    model.fit(features[bg_idx], Zf[bg_idx])
    plane = model.predict(features).reshape(h, w)
    return data - plane


def polynomial_flatten_masked(
    data: np.ndarray,
    mask: np.ndarray,
    order: int = 2,
) -> np.ndarray:
    """
    Fit a 2D polynomial using background (mask==False) and subtract it.

    Parameters
    ----------
    data : np.ndarray
        2D AFM image.
    order : int
        Polynomial order. Default order=2.
    mask : np.ndarray
        Boolean mask of same shape; True=foreground, False=background.

    Returns
    -------
    np.ndarray
        Polynomial-flattened image.

    Raises
    ------
    ValueError
        If mask.shape != data.shape or order is not a positive integer.
    """
    if mask.shape != data.shape:
        raise ValueError("Mask must have same shape as data.")
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
        raise RuntimeError(f"Failed to generate polynomial features: {e}") from e

    bg_idx = ~mask.ravel()

    if np.count_nonzero(bg_idx) < A.shape[1]:
        raise ValueError("Not enough background pixels to perform polynomial fit.")

    # Solve for least-squares polynomial surface
    Zf = Z.ravel()
    try:
        coeff, _, _, _ = np.linalg.lstsq(A[bg_idx], Zf[bg_idx], rcond=None)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Least squares fitting failed: {e}") from e

    # Reconstruct the fitted surface and subtract it
    Z_fit = A @ coeff
    flattened = data.astype(np.float64) - Z_fit.reshape(h, w)

    return flattened


def row_median_align_masked(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute each row's median using background pixels and subtract from each full row.

    Parameters
    ----------
    data : np.ndarray
        2D AFM image.
    mask : np.ndarray
        Boolean mask of same shape; True=foreground, False=background.

    Returns
    -------
    np.ndarray
        Row-masked-alignment image.

    Raises
    ------
    ValueError
        If mask.shape != data.shape.
    """
    if mask.shape != data.shape:
        raise ValueError("Mask must have same shape as data.")
    aligned = data.astype(np.float64).copy()
    h, w = data.shape

    for i in range(h):
        row = data[i, :]
        mask_row = mask[i, :]
        if np.all(mask_row):
            med = 0.0
        else:
            med = np.median(row[~mask_row])
        aligned[i, :] -= med
    return aligned


def register_mask_filters():
    """Return list of masking options."""
    return {
        "remove_plane": remove_plane_masked,
        "polynomial_flatten": polynomial_flatten_masked,
        "row_median_align": row_median_align_masked,
        "zero_mean": lambda data, mask: filters.zero_mean(data, mask=mask),
    }
