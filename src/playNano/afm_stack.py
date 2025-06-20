"""Defines AFMImageStack for managing AFM time-series data and processing steps."""

from __future__ import annotations

import logging
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from playNano.processing import filters, mask_generators, masked_filters
from playNano.utils.time_utils import normalize_timestamps

# Built-in filters and mask dictionaries
FILTER_MAP = filters.register_filters()
MASK_MAP = mask_generators.register_masking()
MASK_FILTERS_MAP = masked_filters.register_mask_filters()

logger = logging.getLogger(__name__)


class AFMImageStack:
    """Class for managing AFM image stacks and applying processing steps.

    Attributes
    ----------
    processed : dict[str, np.ndarray]
        Stores snapshot arrays produced by filter steps. Keys are typically
        named like 'step_1_remove_plane', 'step_2_gaussian', etc., and values
        are 3D arrays of shape (n_frames, height, width).

    masks : dict[str, np.ndarray]
        Stores boolean masks produced by mask steps. Keys follow a similar
        naming pattern (e.g., 'step_3_mask_threshold'), and values are boolean
        arrays of shape (n_frames, height, width).
    """

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
        # Validate that data is a 3D NumPy array
        if not isinstance(data, np.ndarray):
            raise TypeError(f"`data` must be a NumPy array; got {type(data).__name__}")
        if data.ndim != 3:
            raise ValueError(
                f"`data` must be a 3D array (n_frames, height, width); got shape {data.shape}"  # noqa
            )
        # Validate pixel_size_nm
        if not isinstance(pixel_size_nm, (int, float)) or pixel_size_nm <= 0:
            raise ValueError(
                f"`pixel_size_nm` must be a positive number; got {pixel_size_nm!r}"
            )

        self.data = data
        self.pixel_size_nm = pixel_size_nm
        self.channel = channel
        self.file_path = file_path

        # Validate and pad/trim metadata to match number of frames
        n = self.data.shape[0]
        if frame_metadata is None:
            frame_metadata = [{} for _ in range(n)]

        if len(frame_metadata) < n:
            frame_metadata = frame_metadata + [{}] * (n - len(frame_metadata))
        elif len(frame_metadata) > n:
            raise ValueError(
                f"Metadata length ({len(frame_metadata)}) does not match number of frames ({n})."  # noqa: E501
            )

        # Normalize all timestamps
        self.frame_metadata = normalize_timestamps(frame_metadata)

        # Stores processed data arrays from filters, keyed by step
        # name (e.g. 'gaussian_filter', 'remove_plane')
        self.processed: dict[str, np.ndarray] = {}
        # Stores generated masks, keyed by mask generator name (e.g. 'otsu',
        # 'threshold')
        self.masks: dict[str, np.ndarray] = {}

    def _resolve_step(self, step: str) -> tuple[str, callable]:
        """
        Resolves a processing step identifier to its corresponding type and callable.

        This method determines the nature of the given `step` string and returns a tuple
        containing the step type and the associated function or method.

        The step can be one of:

        - "clear": A special step that clears previous operations (returns `None`
        as the function).
        - "mask": A predefined mask operation from `MASK_MAP`.
        - "method": A bound method defined on the current instance.
        - "plugin": A dynamically loaded plugin from the `playNano.filters`
        entry point group.
        - "filter": A predefined filter operation from `FILTER_MAP`.

        Parameters
        ----------
        step : str
            The name of the processing step to resolve.

        Returns
        -------
        tuple[str, Callable | None]
            A tuple containing:
            - The type of the step as a string.
            - The corresponding callable object, or `None` if the step is "clear".

        Raises
        ------
        ValueError
            If the step name is not recognized as any of the supported types.

        Notes
        -----
        The resolution order is:
        1. "clear"
        2. Mask from `MASK_MAP`
        3. Bound method on the instance
        4. Plugin from `playNano.filters`
        5. Filter from `FILTER_MAP`
        """
        if step == "clear":
            return "clear", None

        # 1) Mask generator?
        if step in MASK_MAP:
            return "mask", MASK_MAP[step]

        # 2) Bound method on self? (e.g. a custom method on AFMImageStack)
        method = getattr(self, step, None)
        if callable(method):
            return "method", method

        # 3) Plugin filter entry point?
        try:
            ep = next(
                ep
                for ep in metadata.entry_points(group="playNano.filters")
                if ep.name == step
            )
        except StopIteration:
            ep = None
        if ep is not None:
            fn = ep.load()
            return "plugin", fn

        # 4) Unmasked filter in FILTER_MAP?
        if step in FILTER_MAP:
            return "filter", FILTER_MAP[step]

        # 5) No match
        raise ValueError(
            f"Unrecognized step '{step}'. Available masks: {list(MASK_MAP)}; "
            f"built-in filters: {list(FILTER_MAP)}; methods: {[m for m in dir(self) if callable(getattr(self,m))]}; "  # noqa
            f"plugins: {[ep.name for ep in metadata.entry_points(group='playNano.filters')]}."  # noqa
        )

    def _execute_mask_step(
        self, mask_fn: callable, arr: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Return a boolean mask for each frame in 'arr' using the provided mask function.

        Run a mask generator function (mask_fn) on each frame of 'arr'
        (shape (N, H, W)), returning a boolean array of shape (N, H, W).
        If mask_fn(frame, **kwargs) raises TypeError, try mask_fn(frame).
        If any other exception occurs, log an error and set that frame's
        mask to all False.
        """
        n_frames, H, W = arr.shape
        new_mask = np.zeros((n_frames, H, W), dtype=bool)
        for i in range(n_frames):
            try:
                # First, attempt to call with kwargs
                new_mask[i] = mask_fn(arr[i], **kwargs)
            except TypeError:
                try:
                    new_mask[i] = mask_fn(arr[i])
                except Exception as e:
                    logger.error(
                        f"Mask generator '{mask_fn.__name__}' failed on frame {i}: {e}"
                    )  # noqa
                    new_mask[i] = np.zeros((H, W), dtype=bool)
            except Exception as e:
                logger.error(
                    f"Mask generator '{mask_fn.__name__}' failed on frame {i}: {e}"
                )  # noqa
                new_mask[i] = np.zeros((H, W), dtype=bool)
        return new_mask

    def _execute_filter_step(
        self,
        filter_fn: callable,
        arr: np.ndarray,
        mask: np.ndarray | None,
        step_name: str,
        **kwargs,
    ) -> np.ndarray:
        """
        Apply a filter (filter_fn) to each frame in 'arr'.

        If mask is not None AND
        step_name is in MASK_FILTERS_MAP, call the masked version for each frame:
            MASK_FILTERS_MAP[step_name](frame, mask[i], **kwargs)
        Otherwise, call filter_fn(frame, **kwargs).
        Returns a new array of same shape as arr.
        Logs warnings if any frame's filter raises an exception, and keeps original
        frame in that case.
        """
        n_frames, H, W = arr.shape
        new_arr = np.zeros_like(arr)

        if mask is not None and step_name in MASK_FILTERS_MAP:
            masked_fn = MASK_FILTERS_MAP[step_name]
            for i in range(n_frames):
                try:
                    new_arr[i] = masked_fn(arr[i], mask[i], **kwargs)
                except TypeError:
                    try:
                        new_arr[i] = masked_fn(arr[i], mask[i])
                    except Exception as e:
                        logger.error(
                            f"Masked filter '{step_name}' failed on frame {i}: {e}"
                        )
                        new_arr[i] = arr[i]
                except Exception as e:
                    logger.error(
                        f"Masked filter '{step_name}' failed on frame {i}: {e}"
                    )
                    new_arr[i] = arr[i]
        else:
            for i in range(n_frames):
                try:
                    new_arr[i] = filter_fn(arr[i], **kwargs)
                except TypeError:
                    try:
                        new_arr[i] = filter_fn(arr[i])
                    except Exception as e:
                        logger.warning(
                            f"Filter '{step_name}' failed on frame {i}: {e}"
                        )  # noqa
                        new_arr[i] = arr[i]
                except Exception as e:
                    logger.warning(f"Filter '{step_name}' failed on frame {i}: {e}")
                    new_arr[i] = arr[i]

        return new_arr

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
        )

        return cls(
            data=afm.data,
            pixel_size_nm=afm.pixel_size_nm,
            channel=channel,
            file_path=Path(path),
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
                logger.warning(f"Warning: Frame {idx} is None and skipped")

    def __getitem__(self, idx: int | slice) -> np.ndarray | AFMImageStack:
        """
        Get a specific frame or a slice of frames from the stack.

        Allow stack[i] to return the i-th frame (2D array), or stack[i:j]
        to return a new AFMImageStack containing frames [i:j].
        Raises TypeError for invalid index types.
        """
        if isinstance(idx, int):
            return self.data[idx]
        if isinstance(idx, slice):
            sub_data = self.data[idx]
            sub_meta = self.frame_metadata[idx]
            return AFMImageStack(
                data=sub_data,
                pixel_size_nm=self.pixel_size_nm,
                channel=self.channel,
                file_path=self.file_path,
                frame_metadata=sub_meta,
            )
        raise TypeError(f"Invalid index type: {type(idx).__name__}")

    def _snapshot_raw(self):
        """
        Store the very first raw data under 'raw' in processed dict.

        Subsequent calls do nothing.
        """
        if "raw" not in self.processed:
            # copy the array so later modifications to self.data don’t touch 'raw'
            self.processed["raw"] = self.data.copy()

    def export_processing_log(self, path: str) -> None:
        """
        Export the processing history, snapshot keys, and environment metadata to JSON.

        Parameters
        ----------
        path : str
            File path to write the log.
        """
        import json
        import os

        from playNano.analysis.utils import NumpyEncoder

        record = {
            "environment": getattr(self, "processing_environment", {}),
            "steps": getattr(self, "processing_history", []),
            "keys_by_name": getattr(self, "processing_keys_by_name", {}),
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(record, f, indent=2, cls=NumpyEncoder)

    def export_analysis_log(self, path: str) -> None:
        """
        Export the analysis record (steps, results, and environment metadata) to JSON.

        Parameters
        ----------
        path : str
            File path to write the log.
        """
        import json
        import os

        from playNano.analysis.utils import NumpyEncoder

        if not hasattr(self, "analysis_results") or not self.analysis_results:
            raise ValueError(
                "No analysis results found. Run an AnalysisPipeline first."
            )

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.analysis_results, f, indent=2, cls=NumpyEncoder)

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

        Steps can be:
          - "clear"       : reset any existing mask
          - mask names    : keys in MASK_MAP
          - filter names  : keys in FILTER_MAP
          - plugin names  : entry points in 'playNano.filters'
          - method names  : bound methods on this class
        **kwargs are forwarded to mask functions or filter functions as appropriate.

        Returns
        -------
        np.ndarray
          The processed data array of shape (n_frames, height, width).

        NOTES
        -----
        This method does not populate processing_history or assign unique
        snapshot keys per step. For tracked, reproducible processing,
        use ProcessingPipeline.
        """
        # 1) Snapshot raw data if not already done
        if "raw" not in self.processed:
            self.processed["raw"] = self.data.copy()

        arr = self.data
        mask = None

        for step in steps:
            step_type, fn = self._resolve_step(step)

            # (A) CLEAR: drop any existing mask
            if step_type == "clear":
                logger.info("Step 'clear' → dropping existing mask.")
                mask = None
                continue

            # (B) MASK GENERATOR
            if step_type == "mask":
                logger.info(
                    f"Step '{step}' → computing new mask based on current data."
                )
                # Compute mask over all frames
                new_mask = self._execute_mask_step(fn, arr, **kwargs)
                mask = new_mask
                # Do not modify arr itself
                continue

            # (C) FILTER OR PLUGIN
            # fn is now a callable that processes a 2D frame → 2D frame
            logger.info(f"Step '{step}' (filter) → applying to all frames.")
            new_arr = self._execute_filter_step(fn, arr, mask, step, **kwargs)

            # Store a snapshot in processed dict
            self.processed[step] = new_arr.copy()

            # Update arr for next iteration
            arr = new_arr

        # 5) After all steps, overwrite self.data
        self.data = arr
        return arr

    def time_for_frame(self, idx: int) -> float:
        """
        Get the timestamp for a given frame index.

        Parameters
        ----------
        idx : int
            Frame index in the stack.

        Returns
        -------
        float
            Timestamp in seconds for the frame. If not available,
            the index itself is returned as a fallback.

        Notes
        -----
        This fallback (index-as-time) assumes uniform frame intervals
        and is useful for stacks without explicit time metadata.

        Examples
        --------
        >>> stack.frame_metadata = [{"timestamp": 0.0}, {}, {"timestamp": 2.0}]
        >>> stack.time_for_frame(1)
        1.0
        >>> stack.time_for_frame(2)
        2.0
        """
        ts = self.frame_metadata[idx].get("timestamp", None)
        return float(idx) if ts is None else ts

    def get_frame_times(self) -> list[float]:
        """
        Return a list of timestamps (in seconds) for each frame in the stack.

        This method uses `time_for_frame()` to retrieve the timestamp for
        each frame, which allows central control over fallback behavior.

        Returns
        -------
        list of floats
            List of timestamps per frame. If unavailable, the frame index is
            used as a fallback.

        Examples
        --------
        >>> stack.frame_metadata = [{"timestamp": 0.0}, {"timestamp": 1.0}]
        >>> stack.get_frame_times()
        [0.0, 1.0]

        >>> stack.frame_metadata = [{"timestamp": 0.0}, {}]
        >>> stack.get_frame_times()
        [0.0, 1.0]
        """
        return [self.time_for_frame(i) for i in range(len(self.frame_metadata))]

    def channel_for_frame(self, idx: int) -> str:
        """
        Get the channel name for a given frame index.

        Returns the value of the 'channel' key in the frame's metadata if present,
        otherwise falls back to the global stack-level channel.

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
