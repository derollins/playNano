"""Module for masking features of AFM images in Numpy arrays."""

import logging

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def mask_threshold(data: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Mask where abs(data) > threshold."""
    return np.abs(data) > threshold


def mask_mean_offset(data: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Mask values > mean +/- factor*std."""
    return np.abs(data - np.mean(data)) > factor * np.std(data)


def mask_morphological(
    data: np.ndarray, threshold: float, structure_size: int = 3
) -> np.ndarray:
    """Threshold+closing to mask foreground."""
    binary = np.abs(data) > threshold
    structure = np.ones((structure_size, structure_size), dtype=bool)
    return ndimage.binary_closing(binary, structure=structure)


def mask_adaptive(
    data: np.ndarray, block_size: int = 15, offset: float = 0.0
) -> np.ndarray:
    """Adaptive local mean threshold per block."""
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
    """Return list of masking options."""
    return {
        "mask_threshold": mask_threshold,
        "mask_mean_offset": mask_mean_offset,
        "mask_morphological": mask_morphological,
        "mask_adaptive": mask_adaptive,
    }
