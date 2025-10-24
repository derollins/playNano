"""
Functions for editing AFM image stacks by removing or selecting frames.

These functions are designed to be called by the ProcessingPipeline, which
handles provenance tracking and updates to the AFMImageStack object. The
functions here operate purely on data arrays or frame index lists.

Only 'drop_frames' performs actual stack edits.
Other registered stack_edit functions return indices to drop, which are
then passed to 'drop_frames' to ensure consistent provenance tracking.

Functions
---------
- drop_frames : Remove specific frames from a 3D array.
- drop_frame_range : Generate a list of frame indices to drop within a given range.
- select_frames : Generate a list of frame indices to drop, keeping only the selected
  frames.
"""

from typing import Callable

import numpy as np

from playNano.utils.versioning import versioned_filter


@versioned_filter("0.1.0")
def drop_frames(data: np.ndarray, indices_to_drop: list[int]) -> np.ndarray:
    """
    Remove specific frames from a 3D array.

    Parameters
    ----------
    data : np.ndarray
        3D array of shape (n_frames, height, width) representing the image stack.
    indices_to_drop : list of int
        List of frame indices to remove from the stack.

    Returns
    -------
    np.ndarray
        New array with the specified frames removed.

    Raises
    ------
    ValueError
        If any provided indices are out of bounds or if `data` is not 3D.

    Notes
    -----
    - The function does not modify the input array in place.
    - The ProcessingPipeline is responsible for updating metadata and provenance.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D array, got {data.ndim}D.")

    n_frames = data.shape[0]
    indices_to_drop = sorted(set(indices_to_drop))
    if any(i < 0 or i >= n_frames for i in indices_to_drop):
        raise ValueError(
            f"Indices out of range for {n_frames} frames: {indices_to_drop}"
        )

    keep_mask = np.ones(n_frames, dtype=bool)
    keep_mask[indices_to_drop] = False
    return data[keep_mask]


@versioned_filter("0.1.0")
def drop_frame_range(data: np.ndarray, start: int, end: int) -> list[int]:
    """
    Generate indices to drop within a given range of frames.

    Parameters
    ----------
    data : np.ndarray
        3D array of shape (n_frames, height, width) representing the image stack.
    start : int
        Starting index of the range (inclusive).
    end : int
        Ending index of the range (exclusive).

    Returns
    -------
    list of int
        List of indices that should be dropped.

    Raises
    ------
    ValueError
        If the range is invalid or out of bounds.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D array, got {data.ndim}D.")

    n_frames = data.shape[0]
    if start < 0 or end > data.shape[0] or start >= end:
        raise ValueError(f"Invalid range: start={start}, end={end}, total={n_frames}")
    return list(range(start, end))


@versioned_filter("0.1.0")
def select_frames(data: np.ndarray, keep_indices: list[int]) -> list[int]:
    """
    Generate a list of frame indices to drop, keeping only the selected frames.

    Parameters
    ----------
    data : np.ndarray
        3D array of shape (n_frames, height, width) representing the image stack.
    keep_indices : list of int
        Indices of frames to retain in the stack.

    Returns
    -------
    list of int
        List of frame indices that should be dropped.

    Raises
    ------
    ValueError
        If `keep_indices` contains out-of-range values.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D array, got {data.ndim}D.")

    n_frames = data.shape[0]
    keep_indices = sorted(set(keep_indices))
    if any(i < 0 or i >= n_frames for i in keep_indices):
        raise ValueError(f"Invalid frame indices: {keep_indices}")
    all_indices = set(range(n_frames))
    drop_indices = sorted(all_indices - set(keep_indices))
    return drop_indices


def register_stack_edit_processing() -> dict[str, Callable]:
    """
    Return a dictionary of registered stack editing processing filters.

    Keys are names of the operations, values are the functions themselves.
    drop_frames is the operational function takes a 3D stack (n_frames,
    H, W) and a list of indicies and returns a ndarray. drop_frame_range and
    select_frames are helper functions that return lists of indices to drop
    which can be passed to drop_frames.
    """
    return {
        "drop_frames": drop_frames,
        "drop_frame_range": drop_frame_range,
        "select_frames": select_frames,
    }
