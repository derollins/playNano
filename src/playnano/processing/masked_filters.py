"""Module for filtering AFM data in NumPy arrays with a boolean mask."""

import logging

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from playnano.utils.versioning import versioned_filter

logger = logging.getLogger(__name__)


@versioned_filter("0.1.0")
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


@versioned_filter("0.1.0")
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


@versioned_filter("0.1.0")
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


@versioned_filter("0.2.0")
def zero_mean_masked(data: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
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
        logger.warning(
            "Masked zero_mean filter selected but no mask found. Applying unmasked."
        )
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
    logger.debug(f"Mean value calculated: {mean_val}. Subtracting.")
    return img - mean_val


@versioned_filter("0.1.0")
def zero_median_masked(data: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Subtract the overall median height to center the background around zero.

    If a mask is provided, median is computed only over background (mask == False).

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
        mean_val = np.median(img)
    else:
        if mask.shape != img.shape:
            raise ValueError("Mask must have same shape as data.")
        # Compute mean over background (where mask is False)
        unmasked = img[~mask]
        if unmasked.size == 0:
            mean_val = np.median(img)
            raise ValueError(
                "Mask excludes all pixels — cannot compute mean. "
                "zero_mean applied without mask."
            )
        mean_val = np.median(unmasked)
    return img - mean_val


def register_mask_filters():
    """Return list of masking options."""
    return {
        "remove_plane": remove_plane_masked,
        "polynomial_flatten": polynomial_flatten_masked,
        "row_median_align": row_median_align_masked,
        "zero_mean": zero_mean_masked,
        "zero_median": zero_median_masked,
    }
