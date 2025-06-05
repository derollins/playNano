"""Module for filtering whole image stacks of AFM time."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def zero_mean_stack(stack: np.ndarray) -> np.ndarray:
    """
    Subtract the mean of the stack from each frame in a 3D stack.

    Parameters
    ----------
    stack : np.ndarray
        3D NumPy array representing the AFM image stack.

    Returns
    -------
    np.ndarray
        Stack with zero mean per frame.
    """
    if stack.ndim != 3:
        raise ValueError("Input stack must be a 3D NumPy array.")

    mean_of_stack = np.mean(stack, keepdims=True)
    return stack - mean_of_stack
