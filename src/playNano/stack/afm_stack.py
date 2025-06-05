"""Defines AFMImageStack for managing AFM time-series data and processing steps."""

from __future__ import annotations

import logging
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import dateutil.parser
import numpy as np

import playNano.processing.filters as filters
import playNano.processing.masked_filters as masked_filters
import playNano.processing.masking as masking

# Built-in filters and mask dictionaries
FILTER_MAP = filters.register_filters()
MASK_MAP = masking.register_masking()
MASK_FILTERS_MAP = masked_filters.register_mask_filters()

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
        Apply a sequence of processing steps to each frame in the AFM image stack.

        Allows mixing of unmasked filters, mask-generators, masked filters,
            and 'clear'.

        - “clear” resets/discards the current mask (returning to unmasked mode).
        - If a step name is in MASK_MAP, compute a new boolean mask from
            the current data.
        - If a step name is in FILTER_MAP, dispatch to the masked version
            (if mask≠None and MASK_FILTERS_MAP contains that key) or else to
            the unmasked version.
        - Unknown step names are first looked up as plugins, then fall
            back to FILTER_MAP or raise ValueError.

        At the end, self.data is replaced by the final processed array.
        """
        # 1) Take a snapshot of raw data
        if "raw" not in self.processed:
            self.processed["raw"] = self.data.copy()

        # 2) Initialize working array and mask
        arr = self.data  # shape: (N, H, W)
        n_frames = arr.shape[0]
        mask: np.ndarray | None = None

        for step in steps:
            # ---------- (A) CLEAR STEP ----------
            if step == "clear":
                logger.info("Step 'clear' → dropping existing mask (unmasked mode).")
                mask = None
                continue

            # ---------- (B) MASK GENERATOR ----------
            if step in MASK_MAP:
                logger.info(
                    f"Step '{step}' → computing new mask based on current data."
                )  # noqa
                new_mask = np.zeros_like(arr, dtype=bool)
                for i in range(n_frames):
                    try:
                        new_mask[i] = MASK_MAP[step](arr[i], **kwargs)
                    except TypeError:
                        # In case the mask‐generator needs different args,
                        # pass only 'frame'
                        new_mask[i] = MASK_MAP[step](arr[i])
                    except Exception as e:
                        logger.error(
                            f"Error computing mask '{step}' for frame {i}: {e}"
                        )  # noqa
                        new_mask[i] = np.zeros_like(arr[i], dtype=bool)
                mask = new_mask
                # Do not change arr itself—mask only
                continue

            # ---------- (C) FILTER OR PLUGIN ----------
            # 1) Attempt to find a method on self (e.g. self.some_method)
            fn = getattr(self, step, None)
            if not callable(fn):
                # 2) Try loading a plugin entry‐point
                try:
                    fn = self._load_plugin(step)
                except ValueError:
                    fn = None

            if not callable(fn):
                # 3) Fall back to FILTER_MAP (unmasked)
                fn = FILTER_MAP.get(step)

            if not callable(fn):
                # Step is not recognized anywhere
                raise ValueError(
                    f"Filter step '{step}' not found in plugins or FILTER_MAP."
                )  # noqa

            # Now we know ‘fn’ is the unmasked version (callable(frame→frame)).
            # Decide if we should run its masked counterpart:
            if mask is not None and step in MASK_FILTERS_MAP:
                # Use the masked version for each frame
                logger.info(f"Step '{step}' (masked) → applying masked filter.")
                new_arr = np.zeros_like(arr, dtype=arr.dtype)
                for i in range(n_frames):
                    try:
                        new_arr[i] = MASK_FILTERS_MAP[step](arr[i], mask[i], **kwargs)
                    except TypeError:
                        # Drop **kwargs if the masked fn expects only (data, mask)
                        new_arr[i] = MASK_FILTERS_MAP[step](arr[i], mask[i])
                    except Exception as e:
                        logger.error(
                            f"Error in masked filter '{step}' for frame {i}: {e}"
                        )
                        new_arr[i] = arr[i]  # fallback: keep original
                arr = new_arr

            else:
                # Unmasked path: just call fn(frame) for each frame
                logger.info(f"Step '{step}' (unmasked) → applying unmasked filter.")
                new_arr = np.zeros_like(arr, dtype=arr.dtype)
                for i in range(n_frames):
                    try:
                        new_arr[i] = fn(arr[i], **kwargs)
                    except TypeError:
                        # If fn only expects (data,), drop **kwargs
                        new_arr[i] = fn(arr[i])
                    except Exception as e:
                        logger.warning(
                            f"Filter '{step}' returned None or error for frame {i}: {e}"
                        )  # noqa
                        new_arr[i] = arr[i]
                arr = new_arr

            # 4) Store a snapshot in processed dict
            self.processed[step] = arr.copy()

        # 5) After all steps, overwrite self.data
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
