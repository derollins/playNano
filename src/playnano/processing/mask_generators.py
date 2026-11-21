"""
Functions that make boolean masks of features in 2D AFM frames.

Each function takes a frame and returns a boolean array of the same shape, in
which True marks foreground (features) and False marks background. In a
:class:`~playnano.processing.pipeline.ProcessingPipeline`, masks made one after
another are combined with logical OR until a ``clear`` step. Masked filters
(:mod:`playnano.processing.masked_filters`) use the mask to leave features out
of background fits.
"""

import logging

import numpy as np
from scipy import ndimage

from playnano.utils.versioning import versioned_filter

logger = logging.getLogger(__name__)


@versioned_filter("0.1.0")
def mask_threshold(data: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Mask pixels higher than a threshold.

    Parameters
    ----------
    data : numpy.ndarray
        2D AFM image.
    threshold : float, optional
        Height threshold, in the units of ``data`` (usually nm). Default is 0.0.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True where ``data > threshold``. NaN and infinite pixels
        are False.
    """
    return (data > threshold) & np.isfinite(data)


@versioned_filter("0.1.0")
def mask_below_threshold(data: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Mask pixels lower than a threshold.

    Parameters
    ----------
    data : numpy.ndarray
        2D AFM image.
    threshold : float, optional
        Height threshold, in the units of ``data`` (usually nm). Default is 0.0.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True where ``data < threshold``. NaN and infinite pixels
        are False.
    """
    return (data < threshold) & np.isfinite(data)


@versioned_filter("0.1.0")
def mask_mean_offset(data: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Mask pixels more than ``factor`` standard deviations above the mean.

    Parameters
    ----------
    data : numpy.ndarray
        2D AFM image.
    factor : float, optional
        Number of standard deviations above the image mean at which pixels are
        masked. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True where ``data - mean(data) > factor * std(data)``.
    """
    return (data - np.mean(data)) > factor * np.std(data)


@versioned_filter("0.1.0")
def mask_morphological(
    data: np.ndarray, threshold: float = 0.0, structure_size: int = 3
) -> np.ndarray:
    """
    Mask pixels far from zero, then close small gaps with binary closing.

    Unlike :func:`mask_threshold`, this uses the absolute height, so both
    features above ``threshold`` and holes below ``-threshold`` are masked.

    Parameters
    ----------
    data : numpy.ndarray
        2D AFM image, normally already flattened so the background is near zero.
    threshold : float, optional
        Pixels with ``abs(data) > threshold`` are masked before closing. Default
        is 0.0, which masks every pixel that is not exactly zero.
    structure_size : int, optional
        Side length, in pixels, of the square structuring element used for
        closing. Default is 3.

    Returns
    -------
    numpy.ndarray
        Boolean mask after closing.

    See Also
    --------
    scipy.ndimage.binary_closing : The closing operation used.
    """
    binary = np.abs(data) > threshold
    structure = np.ones((structure_size, structure_size), dtype=bool)
    return ndimage.binary_closing(binary, structure=structure)


@versioned_filter("0.1.0")
def mask_adaptive(
    data: np.ndarray, block_size: int = 15, offset: float = 0.0
) -> np.ndarray:
    """
    Mask pixels above the mean of their local block.

    The image is divided into non-overlapping square blocks of ``block_size``
    pixels (smaller at the right and bottom edges), and each pixel is compared
    with the mean of its own block.

    Parameters
    ----------
    data : numpy.ndarray
        2D AFM image.
    block_size : int, optional
        Side length of each block, in pixels. Default is 15.
    offset : float, optional
        Added to each block mean before comparing, in the units of ``data``.
        Default is 0.0.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True where a pixel is greater than its block mean plus
        ``offset``.
    """
    h, w = data.shape
    mask = np.zeros_like(data, dtype=bool)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = data[i : i + block_size, j : j + block_size]
            local_mean = np.mean(block)
            mask_block = block > (local_mean + offset)
            mask[i : i + block_size, j : j + block_size] = mask_block
    return mask


def register_masking():
    """
    Return the built-in mask functions.

    Returns
    -------
    dict
        Maps each step name (str) to its function, for example
        ``"mask_threshold"`` to :func:`mask_threshold`.
    """
    return {
        "mask_threshold": mask_threshold,
        "mask_below_threshold": mask_below_threshold,
        "mask_mean_offset": mask_mean_offset,
        "mask_morphological": mask_morphological,
        "mask_adaptive": mask_adaptive,
    }
