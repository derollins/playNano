"""Module for filtering AFM data in NumPy arrays with a boolean mask."""

import logging

import numpy as np
from sklearn.linear_model import LinearRegression

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
    h, w = data.shape

    # Create coordinate grid
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = data.astype(np.float64)

    Xf = X.ravel()
    Yf = Y.ravel()
    Zf = Z.ravel()
    features = np.vstack((Xf, Yf)).T

    bg_idx = ~mask.ravel()
    model = LinearRegression()
    model.fit(features[bg_idx], Zf[bg_idx])

    plane = model.predict(features).reshape(h, w)
    return data - plane


def polynomial_flatten_masked(
    data: np.ndarray, order: int, mask: np.ndarray
) -> np.ndarray:
    """
    Fit a 2D polynomial (quadratic only) using background (mask==False) and subtract it.

    Parameters
    ----------
    data : np.ndarray
        2D AFM image.
    order : int
        Polynomial order. Only order=2 is supported here.
    mask : np.ndarray
        Boolean mask of same shape; True=foreground, False=background.

    Returns
    -------
    np.ndarray
        Polynomial-flattened image.

    Raises
    ------
    ValueError
        If order != 2 or mask.shape != data.shape.
    """
    if order != 2:
        raise ValueError("Only quadratic flattening supported.")
    if mask.shape != data.shape:
        raise ValueError("Mask must have same shape as data.")

    h, w = data.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = data.astype(np.float64)

    Xf = X.ravel()
    Yf = Y.ravel()
    Zf = Z.ravel()
    A = np.stack([np.ones_like(Xf), Xf, Yf, Xf**2, Xf * Yf, Yf**2], axis=1)

    bg_idx = ~mask.ravel()
    coeff, _, _, _ = np.linalg.lstsq(A[bg_idx], Zf[bg_idx], rcond=None)

    Z_fit = (
        coeff[0]
        + coeff[1] * X
        + coeff[2] * Y
        + coeff[3] * X**2
        + coeff[4] * X * Y
        + coeff[5] * Y**2
    )
    return data - Z_fit


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
