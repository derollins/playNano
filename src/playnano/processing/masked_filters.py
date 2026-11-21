"""
2D filters that fit or measure the background only, using a boolean mask.

Each function takes a frame and a mask of the same shape, in which True marks
foreground (features) and False marks background. The statistic or fit is
computed from background pixels and then applied to the whole frame. The
:class:`~playnano.processing.pipeline.ProcessingPipeline` uses these in place of
the filters in :mod:`playnano.processing.filters` when a mask is active.
"""

import logging

import numpy as np

from playnano.processing.filters import _fit_polynomial_surface
from playnano.utils.versioning import versioned_filter

logger = logging.getLogger(__name__)


@versioned_filter("0.1.1")
def remove_plane_masked(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fit a plane to the background pixels and subtract it from the whole image.

    Parameters
    ----------
    data : numpy.ndarray
        2D AFM image.
    mask : numpy.ndarray
        Boolean array with the same shape as ``data``. True marks foreground
        pixels, which are left out of the fit; False marks background.

    Returns
    -------
    numpy.ndarray
        Image with the plane removed, as float64.

    Raises
    ------
    ValueError
        If ``mask`` and ``data`` have different shapes, or fewer than 3
        background pixels are finite.

    Notes
    -----
    NaN and infinite pixels are left out of the fit and stay NaN or infinite
    in the output. Version 0.1.1 fits with NumPy instead of scikit-learn
    (same result) and handles NaN pixels instead of raising.
    """
    if mask.shape != data.shape:
        raise ValueError("Mask must have same shape as data.")
    z = np.asarray(data, dtype=np.float64)
    return z - _fit_polynomial_surface(z, order=1, fit_mask=~mask)


@versioned_filter("0.2.0")
def polynomial_flatten_masked(
    data: np.ndarray,
    mask: np.ndarray,
    order: int = 2,
) -> np.ndarray:
    """
    Fit a 2D polynomial to the background pixels and subtract it from the whole image.

    Parameters
    ----------
    data : numpy.ndarray
        2D AFM image.
    mask : numpy.ndarray
        Boolean array with the same shape as ``data``. True marks foreground
        pixels, which are left out of the fit; False marks background.
    order : int, optional
        Total order of the polynomial: 1 is a plane, 2 is quadratic. Default is 2.

    Returns
    -------
    numpy.ndarray
        Image with the polynomial background removed, as float64.

    Raises
    ------
    ValueError
        If ``mask`` and ``data`` have different shapes, ``order`` is not a
        positive integer, or fewer background pixels are finite than the
        polynomial has terms.
    RuntimeError
        If the least-squares fit fails.

    Notes
    -----
    NaN and infinite pixels are left out of the fit and stay NaN or infinite
    in the output. Version 0.2.0 scales x and y to [-1, 1] before fitting, so
    orders of 3 and above fit correctly on large frames.
    """
    if mask.shape != data.shape:
        raise ValueError("Mask must have same shape as data.")
    if not isinstance(order, int) or order < 1:
        raise ValueError("Polynomial order must be a positive integer.")

    z = data.astype(np.float64)
    return z - _fit_polynomial_surface(z, order, fit_mask=~mask)


@versioned_filter("0.1.0")
def row_median_align_masked(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Subtract from each row the median of its background pixels.

    Parameters
    ----------
    data : numpy.ndarray
        2D AFM image.
    mask : numpy.ndarray
        Boolean array with the same shape as ``data``. True marks foreground
        pixels, which are left out of the fit; False marks background.

    Returns
    -------
    numpy.ndarray
        Row-aligned image, as float64. Rows with no background pixels are left
        unchanged.

    Raises
    ------
    ValueError
        If ``mask`` and ``data`` have different shapes.
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
    Subtract the mean height of the background so it is centred on zero.

    Parameters
    ----------
    data : numpy.ndarray
        2D AFM image.
    mask : numpy.ndarray, optional
        Boolean array with the same shape as ``data``. True marks foreground
        pixels, which are left out of the mean. If None, the mean of the whole
        image is used and a warning is logged.

    Returns
    -------
    numpy.ndarray
        Zero-mean image, as float64.

    Raises
    ------
    ValueError
        If ``mask`` and ``data`` have different shapes, or every pixel is
        masked.
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
    Subtract the median height of the background so it is centred on zero.

    Parameters
    ----------
    data : numpy.ndarray
        2D AFM image.
    mask : numpy.ndarray, optional
        Boolean array with the same shape as ``data``. True marks foreground
        pixels, which are left out of the median. If None, the median of the
        whole image is used and a warning is logged.

    Returns
    -------
    numpy.ndarray
        Zero-median image, as float64.

    Raises
    ------
    ValueError
        If ``mask`` and ``data`` have different shapes, or every pixel is
        masked.
    """
    img = data.astype(np.float64).copy()
    if mask is None:
        logger.warning(
            "Masked zero_median filter selected but no mask found. Applying unmasked."
        )
        median_val = np.median(img)
    else:
        if mask.shape != img.shape:
            raise ValueError("Mask must have same shape as data.")
        # Compute median over background (where mask is False)
        unmasked = img[~mask]
        if unmasked.size == 0:
            raise ValueError("Mask excludes all pixels, cannot compute median.")
        median_val = np.median(unmasked)
    return img - median_val


def register_mask_filters():
    """
    Return the masked filters, keyed by the name of the unmasked filter they replace.

    Returns
    -------
    dict
        Maps each step name (str) to its function, for example
        ``"remove_plane"`` to :func:`remove_plane_masked`.
    """
    return {
        "remove_plane": remove_plane_masked,
        "polynomial_flatten": polynomial_flatten_masked,
        "row_median_align": row_median_align_masked,
        "zero_mean": zero_mean_masked,
        "zero_median": zero_median_masked,
    }
