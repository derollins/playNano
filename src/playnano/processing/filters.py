"""
2D filters for flattening and smoothing AFM frames.

Each filter takes one frame as a 2D :class:`numpy.ndarray` and returns the
result; the input is not changed. The
:class:`~playnano.processing.pipeline.ProcessingPipeline` applies them to every
frame of a stack. Masked versions of the flattening filters are in
:mod:`playnano.processing.masked_filters`.
"""

import logging

import numpy as np
from scipy import ndimage

from playnano.utils.versioning import versioned_filter

logger = logging.getLogger(__name__)


def _scaled_axis(n: int) -> np.ndarray:
    """Return ``n`` evenly spaced coordinates from -1 to 1 (0 if ``n`` is 1)."""
    return np.linspace(-1.0, 1.0, n) if n > 1 else np.zeros(1)


def _fit_polynomial_surface(
    data: np.ndarray, order: int, fit_mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Fit a 2D polynomial surface by least squares and return it for every pixel.

    The surface has every term ``x**i * y**j`` with ``i + j <= order``, so
    ``order=1`` is a plane. Shared by the plane and polynomial filters in this
    module and in :mod:`playnano.processing.masked_filters`.

    Parameters
    ----------
    data : np.ndarray
        2D image.
    order : int
        Total polynomial order (1 for a plane).
    fit_mask : np.ndarray of bool, optional
        Pixels to fit to (True = use). By default all pixels are used.

    Returns
    -------
    np.ndarray
        The fitted surface, float64, same shape as ``data``.

    Raises
    ------
    ValueError
        If fewer usable pixels remain than the polynomial has terms.
    RuntimeError
        If the least-squares fit fails.

    Notes
    -----
    x and y are scaled to [-1, 1] before the terms are built. With raw pixel
    indices, high powers span many orders of magnitude (1023**6 is about
    1e18), and the solver silently drops terms, so high orders on large frames
    fitted badly. Pixels that are NaN or infinite are left out of the fit, so
    their value in the output of the calling filter stays NaN or infinite.
    """
    z = np.asarray(data, dtype=np.float64)
    use = np.isfinite(z)
    if fit_mask is not None:
        use &= fit_mask

    n_terms = (order + 1) * (order + 2) // 2
    n_use = int(np.count_nonzero(use))
    if n_use < n_terms:
        raise ValueError(
            f"Not enough usable pixels to fit a polynomial of order {order}: "
            f"{n_use} found, at least {n_terms} needed. Masked pixels and pixels "
            "that are NaN or infinite are not used."
        )

    h, w = z.shape
    X, Y = np.meshgrid(_scaled_axis(w), _scaled_axis(h))
    powers = [(i, total - i) for total in range(order + 1) for i in range(total + 1)]
    terms = [X**i * Y**j for i, j in powers]

    A = np.stack([t[use] for t in terms], axis=1)
    try:
        coeff, _, _, _ = np.linalg.lstsq(A, z[use], rcond=None)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Least squares fitting failed: {e}") from e

    surface = np.zeros_like(z)
    for c, t in zip(coeff, terms, strict=True):
        surface += c * t
    return surface


@versioned_filter("0.1.0")
def row_median_align(data: np.ndarray) -> np.ndarray:
    """
    Subtract the median of each row from that row to remove horizontal banding.

    Parameters
    ----------
    data : numpy.ndarray
        2D AFM image.

    Returns
    -------
    numpy.ndarray
        Row-aligned image, as float64.
    """
    aligned = data.astype(np.float64).copy()
    # Compute median for each row
    medians = np.median(aligned, axis=1)
    # Subtract median from each row
    aligned = aligned - medians[:, np.newaxis]
    return aligned


@versioned_filter("0.1.1")
def remove_plane(data: np.ndarray) -> np.ndarray:
    """
    Fit a plane to the image by least squares and subtract it.

    The plane ``z = a*x + b*y + c`` removes the overall tilt of the frame.

    Parameters
    ----------
    data : numpy.ndarray
        2D AFM image.

    Returns
    -------
    numpy.ndarray
        Image with the plane removed, as float64.

    Raises
    ------
    ValueError
        If fewer than 3 pixels are finite.

    Notes
    -----
    NaN and infinite pixels are left out of the fit and stay NaN or infinite
    in the output. Version 0.1.1 fits with NumPy instead of scikit-learn
    (same result) and handles NaN pixels instead of raising.
    """
    z = np.asarray(data, dtype=np.float64)
    return z - _fit_polynomial_surface(z, order=1)


@versioned_filter("0.2.0")
def polynomial_flatten(data: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Fit a 2D polynomial surface to the image and subtract it.

    Removes tilt (order 1) and curvature (order 2 and above) from the frame.

    Parameters
    ----------
    data : numpy.ndarray
        2D AFM image.
    order : int, optional
        Total order of the polynomial: 1 is a plane, 2 is quadratic. Default is 2.

    Returns
    -------
    numpy.ndarray
        Image with the polynomial background removed, as float64.

    Raises
    ------
    ValueError
        If ``data`` is not a 2D array, if ``order`` is not a positive integer,
        or if fewer pixels are finite than the polynomial has terms
        (``(order + 1) * (order + 2) / 2``).
    RuntimeError
        If the least-squares fit fails.

    Notes
    -----
    NaN and infinite pixels are left out of the fit and stay NaN or infinite
    in the output. Version 0.2.0 scales x and y to [-1, 1] before fitting, so
    orders of 3 and above fit correctly on large frames; before, terms were
    silently dropped and the background was not fully removed.
    """
    # Validate input shape and type
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input data must be a 2D NumPy array.")
    if not isinstance(order, int) or order < 1:
        raise ValueError("Polynomial order must be a positive integer.")

    z = data.astype(np.float64)
    return z - _fit_polynomial_surface(z, order)


@versioned_filter("0.2.0")
def zero_mean(data: np.ndarray) -> np.ndarray:
    """
    Subtract the mean height of the image so it is centred on zero.

    Parameters
    ----------
    data : numpy.ndarray
        2D AFM image.

    Returns
    -------
    numpy.ndarray
        Zero-mean image, as float64.

    See Also
    --------
    playnano.processing.masked_filters.zero_mean_masked :
        The same, with the mean taken over background pixels only.
    """
    img = data.astype(np.float64).copy()
    mean_val = np.mean(img)
    logger.debug(f"Mean value calculated: {mean_val}. Subtracting.")
    return img - mean_val


@versioned_filter("0.1.0")
def zero_median(data: np.ndarray) -> np.ndarray:
    """
    Subtract the median height of the image so it is centred on zero.

    Parameters
    ----------
    data : numpy.ndarray
        2D AFM image.

    Returns
    -------
    numpy.ndarray
        Zero-median image, as float64.

    See Also
    --------
    playnano.processing.masked_filters.zero_median_masked :
        The same, with the median taken over background pixels only.
    """
    img = data.astype(np.float64).copy()
    median_val = np.median(img)
    return img - median_val


@versioned_filter("0.1.0")
def gaussian_filter(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Smooth the image with a Gaussian filter to reduce high-frequency noise.

    Parameters
    ----------
    data : numpy.ndarray
        2D AFM image.
    sigma : float, optional
        Standard deviation of the Gaussian kernel, in pixels. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        Smoothed image, with the same dtype as ``data``.

    See Also
    --------
    scipy.ndimage.gaussian_filter : The function used.
    """
    return ndimage.gaussian_filter(data, sigma=sigma)


@versioned_filter("0.1.0")
def vertical_flip(data: np.ndarray) -> np.ndarray:
    """
    Flip the image vertically, reversing the order of the rows.

    Parameters
    ----------
    data : numpy.ndarray
        2D AFM image.

    Returns
    -------
    numpy.ndarray
        Flipped image (a view of ``data``).
    """
    return np.flipud(data)


def register_filters():
    """
    Return the built-in 2D filters.

    Returns
    -------
    dict
        Maps each step name (str) to its function, for example
        ``"remove_plane"`` to :func:`remove_plane`.
    """
    return {
        "remove_plane": remove_plane,
        "row_median_align": row_median_align,
        "zero_mean": zero_mean,
        "zero_median": zero_median,
        "polynomial_flatten": polynomial_flatten,
        "gaussian_filter": gaussian_filter,
        "vertical_flip": vertical_flip,
    }
