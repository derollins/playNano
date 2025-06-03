"""Defines AFMImageStack for managing AFM time-series data and processing steps."""

from __future__ import annotations

import logging
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import dateutil.parser
import numpy as np

from playNano.processing import filters

FILTER_MAP = filters.register_filters()

logger = logging.getLogger(__name__)


def normalize_timestamps(metadata_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize timestamp data to a float in seconds.

    Given a list of per-frame metadata dicts, parse each 'timestamp' entry
    (if present) into a float (seconds). Returns a new list of dicts
    with 'timestamp' replaced by float or None.

    - ISO-format strings → parsed with dateutil.isoparse()
    - datetime objects       → .timestamp()
    - numeric (int/float)    → float()
    - missing/unparsable     → None

    Parameters
    ----------
    metadata_list : list of dict
        List of metadata dictionaries, each possibly containing a 'timestamp'.

    Returns
    -------
    list of dict
        List of metadata dicts with 'timestamp' normalized to float seconds or None.
    """
    normalized: list[dict[str, Any]] = []
    for md in metadata_list:
        new_md = dict(md)  # shallow copy so we don't mutate the original
        t = new_md.get("timestamp", None)

        if t is None:
            new_md["timestamp"] = None

        elif isinstance(t, str):
            try:
                dt = dateutil.parser.isoparse(t)
                new_md["timestamp"] = dt.timestamp()
            except Exception:
                # parsing failed
                new_md["timestamp"] = None

        elif isinstance(t, datetime):
            new_md["timestamp"] = t.timestamp()

        elif isinstance(t, (int, float)):
            new_md["timestamp"] = float(t)

        else:
            new_md["timestamp"] = None

        normalized.append(new_md)

    return normalized


class AFMImageStack:
    """Class for managing AFM image stacks and applying processing steps."""

    def __init__(
        self,
        data: np.ndarray,
        pixel_size_nm: float,
        channel: str,
        file_path: Path,
        frame_metadata: list[dict[str, Any]] = None,
    ) -> None:
        """
        Represent a stack of AFM images and associated metadata.

        Parameters
        ----------
        data : np.ndarray
            NumPy array of shape (n_frames, height, width) with image data.
        pixel_size_nm : float
            Physical pixel size in nanometers.
        channel : str
            Channel name.
        file_path : str | Path
            Path to source file or folder.
        frame_metadata : list of dict, optional
            Per-frame metadata; will be padded or trimmed to length n_frames
        """
        self.data = data
        self.pixel_size_nm = pixel_size_nm
        self.channel = channel
        self.file_path = file_path

        # Validate and pad/trim metadata to match number of frames
        n = self.data.shape[0]
        if frame_metadata is None:
            self.frame_metadata = [{} for _ in range(n)]
        else:
            if len(frame_metadata) < n:
                self.frame_metadata = frame_metadata + [{}] * (n - len(frame_metadata))
            elif len(frame_metadata) > n:
                raise ValueError(
                    f"Metadata length ({len(frame_metadata)}) does not match number of frames ({n})."  # noqa
                )
            else:
                self.frame_metadata = frame_metadata

        # Store processed results; 'raw' is populated on first processing
        self.processed: dict[str, np.ndarray] = {}

    @classmethod
    def load_data(
        cls, path: str | Path, channel: str = "height_trace"
    ) -> AFMImageStack:
        """
        Load AFM data from a file or folder into an AFMImageStack instance.

        Parameters
        ----------
        path : str or Path
            Path to AFM data file or folder.
        channel : str, optional
            Channel name to load (default is "height_trace").

        Returns
        -------
        AFMImageStack
            Loaded AFMImageStack instance with normalized timestamps.
        """
        from playNano.io.loader import load_afm_stack

        afm = load_afm_stack(path, channel)

        normalized_metadata: list[dict[str, Any]] = normalize_timestamps(
            afm.frame_metadata
        )  # noqa

        return cls(
            data=afm.data,
            pixel_size_nm=afm.pixel_size_nm,
            channel=channel,
            file_path=path,
            frame_metadata=normalized_metadata,
        )

    @property
    def n_frames(self) -> int:
        """Number of frames in the stack."""
        return self.data.shape[0]

    @property
    def height(self) -> int:
        """Return frame height."""
        return self.data.shape[1]

    @property
    def width(self) -> int:
        """Return frame width."""
        return self.data.shape[2]

    @property
    def image_shape(self) -> tuple[int, int]:
        """Return (height, width)."""
        return self.data.shape[1:]

    def get_frame(self, index: int) -> np.ndarray:
        """
        Retrieve image data for a specific frame index.

        Parameters
        ----------
        index : int
            Frame index to retrieve.

        Returns
        -------
        np.ndarray
            2D array of image data for the specified frame.
        """
        return self.data[index]

    def get_frame_metadata(self, index: int) -> dict[str, Any]:
        """
        Retrieve metadata for a specific frame index.

        Parameters
        ----------
        index : int
            Frame index for which to retrieve metadata.

        Returns
        -------
        dict
            Metadata dictionary for the specified frame.

        Raises
        ------
        IndexError
            If the frame index is out of range.
        """
        if 0 <= index < len(self.frame_metadata):
            return self.frame_metadata[index]
        else:
            raise IndexError(f"Frame metadata index {index} out of range")

    def get_frames(self) -> list[np.ndarray]:
        """
        Return a list of all individual frames in the stack.

        Returns
        -------
        list of np.ndarray
            List containing each frame as a 2D NumPy array.
        """
        return [self.get_frame(i) for i in range(self.n_frames)]

    def frames_with_metadata(self):
        """
        Yield tuples of (frame_index, frame_image, metadata) for all frames.

        Yields
        ------
        tuple
            A tuple (index, frame image array, metadata dict) for each frame.
        """
        for idx, (image, meta) in enumerate(
            zip(self.data, self.frame_metadata, strict=False)
        ):
            if image is not None:
                yield idx, image, meta
            else:
                print(f"Warning: Frame {idx} is None and skipped")

    def _snapshot_raw(self):
        """
        Store the very first raw data under 'raw' in processed dict.

        Subsequent calls do nothing.
        """
        if "raw" not in self.processed:
            # copy the array so later modifications to self.data don’t touch 'raw'
            self.processed["raw"] = self.data.copy()

    def _load_plugin(self, name: str):
        """
        Load a filter plugin dynamically from entry points.

        Parameters
        ----------
        name : str
            Name of the plugin filter to load.

        Returns
        -------
        Callable
            Loaded filter function.

        Raises
        ------
        ValueError
            If the plugin name is not found among entry points.
        """
        for ep in metadata.entry_points(group="playNano.filters"):
            if ep.name == name:
                logger.debug(f"Loaded plugin '{name}' from {ep.value}")
                return ep.load()

        raise ValueError(f"Unknown filter plugin: {name}")

    def apply(self, steps: list[str], **kwargs) -> np.ndarray:
        """
        Apply a sequence of filter steps to the AFM image stack.

        Parameters
        ----------
        steps : list of str
            List of filter names or method names to apply in order.
        **kwargs : dict
            Additional parameters passed to filter functions.

        Returns
        -------
        np.ndarray
            The processed data array after applying all steps.

        Raises
        ------
        ValueError
            If a filter step name is not found.
        """
        if "raw" not in self.processed:
            self.processed["raw"] = self.data.copy()

        arr = self.data  # shape: (N, H, W)

        for step in steps:
            fn = getattr(self, step, None)
            if not callable(fn):
                # Try load from entry points first
                try:
                    fn = self._load_plugin(step)
                except ValueError:
                    # fallback to local filters dict
                    fn = FILTER_MAP.get(step)

            if not callable(fn):
                raise ValueError(f"Filter step '{step}' not found")

            # Apply filter frame-wise
            filtered_frames = []
            for i in range(arr.shape[0]):
                frame = arr[i]
                filtered_frame = fn(frame, **kwargs)
                if filtered_frame is None:
                    logger.warning(f"Filter '{step}' returned None for frame {i}")
                    filtered_frame = frame  # fallback to original frame
                filtered_frames.append(filtered_frame)

            arr = np.stack(filtered_frames, axis=0)
            self.processed[step] = arr
            logger.info(f"Applied filter step: {step}")

        self.data = arr
        return arr

    def time_for_frame(self, idx: int) -> Any:
        """
        Get the timestamp for a given frame index.

        Parameters
        ----------
        idx : int
            Frame index.

        Returns
        -------
        float or None
            Timestamp in seconds or None if unavailable.
        """
        return self.frame_metadata[idx].get("timestamp")

    def channel_for_frame(self, idx: int) -> str:
        """
        Get the channel name for a given frame index.

        Parameters
        ----------
        idx : int
            Frame index.

        Returns
        -------
        str
            Channel name for the frame.
        """
        return self.frame_metadata[idx].get("channel", self.channel)

    def restore_raw(self) -> np.ndarray:
        """
        Restore self.data from the 'raw' snapshot in self.processed.

        Returns
        -------
        np.ndarray
            The restored raw data.

        Raises
        ------
        KeyError
            If 'raw' data is not available in self.processed.
        """
        if "raw" not in self.processed:
            raise KeyError("No raw data snapshot available to restore.")

        self.data = self.processed["raw"].copy()
        logger.info("Data restored from raw snapshot.")
        return self.data
