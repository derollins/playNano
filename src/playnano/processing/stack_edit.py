"""
Functions for removing or selecting frames of an AFM image stack.

Only :func:`drop_frames` changes a stack. The other functions return the
indices of the frames to drop; in a
:class:`~playnano.processing.pipeline.ProcessingPipeline` these indices are
passed to :func:`drop_frames`, so every edit is recorded the same way in the
provenance. The pipeline also removes the matching entries from the stack's
frame metadata and keeps a copy of the metadata from before the first edit.

- :func:`drop_frames` removes the given frames.
- :func:`drop_frame_range` drops the frames from ``start`` up to, but not
  including, ``end``.
- :func:`select_frames` keeps only the given frames.
- :func:`drop_final_frames` drops frames from the end (by default, the last
  one).
"""

from typing import Callable

import numpy as np

from playnano.utils.versioning import versioned_filter


@versioned_filter("0.1.0")
def drop_frames(data: np.ndarray, indices_to_drop: list[int]) -> np.ndarray:
    """
    Remove the given frames from a stack.

    Parameters
    ----------
    data : numpy.ndarray
        3D image stack of shape (n_frames, height, width).
    indices_to_drop : list of int
        Indices of the frames to remove. Duplicates are ignored; negative
        indices are not allowed.

    Returns
    -------
    numpy.ndarray
        A new array without the given frames. ``data`` is not changed.

    Raises
    ------
    ValueError
        If ``data`` is not 3D or an index is outside ``0 .. n_frames - 1``.
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
    Return the indices of a range of frames to drop.

    Parameters
    ----------
    data : numpy.ndarray
        3D image stack of shape (n_frames, height, width).
    start : int
        First frame to drop (inclusive).
    end : int
        Frame after the last one to drop (exclusive).

    Returns
    -------
    list of int
        ``[start, start + 1, ..., end - 1]``.

    Raises
    ------
    ValueError
        If ``data`` is not 3D, ``start < 0``, ``end > n_frames`` or
        ``start >= end``.
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
    Return the indices of the frames to drop so that only the given frames are kept.

    Parameters
    ----------
    data : numpy.ndarray
        3D image stack of shape (n_frames, height, width).
    keep_indices : list of int
        Indices of the frames to keep. Duplicates are ignored.

    Returns
    -------
    list of int
        Sorted indices of every other frame.

    Raises
    ------
    ValueError
        If ``data`` is not 3D or an index is outside ``0 .. n_frames - 1``.
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


@versioned_filter("0.1.0")
def drop_final_frames(data: np.ndarray, n_frames_to_drop: int = 1) -> list[int]:
    """
    Return the indices of the last frames of a stack, to drop.

    Parameters
    ----------
    data : numpy.ndarray
        3D image stack of shape (n_frames, height, width).
    n_frames_to_drop : int, optional
        Number of frames to drop from the end. Default is 1.

    Returns
    -------
    list of int
        Indices of the last ``n_frames_to_drop`` frames (empty if it is 0).

    Raises
    ------
    ValueError
        If ``data`` is not 3D, or ``n_frames_to_drop`` is negative or larger
        than the number of frames.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D array, got {data.ndim}D.")

    n_frames = data.shape[0]
    if n_frames_to_drop < 0 or n_frames_to_drop > n_frames:
        raise ValueError(f"Invalid number of frames to drop: {n_frames_to_drop}")

    return list(range(n_frames - n_frames_to_drop, n_frames))


def register_stack_edit_processing() -> dict[str, Callable]:
    """
    Return the built-in stack edit functions.

    Returns
    -------
    dict
        Maps each step name (str) to its function, for example
        ``"drop_frames"`` to :func:`drop_frames`.
    """
    return {
        "drop_frames": drop_frames,
        "drop_frame_range": drop_frame_range,
        "select_frames": select_frames,
        "drop_final_frames": drop_final_frames,
    }
