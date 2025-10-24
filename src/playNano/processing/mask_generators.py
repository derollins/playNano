"""Module for masking features of AFM images in Numpy arrays."""

import logging

import numpy as np
from scipy import ndimage

from playNano.utils.versioning import versioned_filter

logger = logging.getLogger(__name__)


@versioned_filter("0.1.0")
def mask_threshold(data: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Mask where data > threshold.

    Parameters
    ----------
    data : numpy.ndarray
        Input 2D array.

    threshold : float, optional
        Threshold value. Pixels greater than this will be True. Default is 0.0.

    Returns
    -------
    np.ndarray
        Boolean mask array.
    """
    return (data > threshold) & np.isfinite(data)


@versioned_filter("0.1.0")
def mask_below_threshold(data: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Mask where data < threshold.

    Parameters
    ----------
    data : numpy.ndarray
        Input 2D array.

    threshold : float, optional
        Threshold value. Pixels less than this will be True. Default is 0.0.

    Returns
    -------
    np.ndarray
        Boolean mask array.
    """
    return (data < threshold) & np.isfinite(data)


@versioned_filter("0.1.0")
def mask_mean_offset(data: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Mask values greater than mean plus factor * standard deviation.

    Parameters
    ----------
    data : numpy.ndarray
        Input 2D array.

    factor : float, optional
        Factor multiplied by the standard deviation to define the threshold.
        Default is 1.0.

    Returns
    -------
    np.ndarray
        Boolean mask array.
    """
    return (data - np.mean(data)) > factor * np.std(data)


@versioned_filter("0.1.0")
def mask_morphological(
    data: np.ndarray, threshold: float = 0.0, structure_size: int = 3
) -> np.ndarray:
    """
    Apply threshold and morphological closing to mask foreground.

    Parameters
    ----------
    data : numpy.ndarray
        Input 2D array.

    threshold : float, optional
        Threshold value. Default is 0.0.

    structure_size : int, optional
        Size of the structuring element for binary closing. Default is 3.

    Returns
    -------
    np.ndarray
        Boolean mask array after morphological closing.
    """
    binary = np.abs(data) > threshold
    structure = np.ones((structure_size, structure_size), dtype=bool)
    return ndimage.binary_closing(binary, structure=structure)


@versioned_filter("0.1.0")
def mask_adaptive(
    data: np.ndarray, block_size: int = 15, offset: float = 0.0
) -> np.ndarray:
    """
    Adaptive local mean threshold per block.

    Parameters
    ----------
    data : numpy.ndarray
        Input 2D array.

    block_size : int, optional
        Size of the local block. Default is 15.

    offset : float, optional
        Value added to local mean when thresholding. Default is 0.0.

    Returns
    -------
    np.ndarray
        Boolean mask array where True indicates pixels above the threshold.
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
    Return dictionary of available masking functions.

    Returns
    -------
    dict
        Keys are function names (str), values are the corresponding callable functions.
    """
    return {
        "mask_threshold": mask_threshold,
        "mask_below_threshold": mask_below_threshold,
        "mask_mean_offset": mask_mean_offset,
        "mask_morphological": mask_morphological,
        "mask_adaptive": mask_adaptive,
    }
