"""Defines AFMImageStack for managing AFM time-series data and processing steps."""

from __future__ import annotations

import json
import logging
import os
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from playNano.processing import (
    filters,
    mask_generators,
    masked_filters,
    stack_edit,
    video_processing,
)
from playNano.utils.time_utils import normalize_timestamps

# Built-in filters and mask dictionaries
FILTER_MAP = filters.register_filters()
MASK_MAP = mask_generators.register_masking()
MASK_FILTERS_MAP = masked_filters.register_mask_filters()
VIDEO_FILTER_MAP = video_processing.register_video_processing()
STACK_EDIT_MAP = stack_edit.register_stack_edit_processing()

logger = logging.getLogger(__name__)


class AFMImageStack:
    """
    Manage stacks of AFM images with metadata, analysis results, and provenance.

    Contains snapshots of each stage of processing (including the raw data after the
    first processing step), any masks generated, analysis results, and the provenance
    of each processing and analysis step.

    Attributes
    ----------
    data : np.ndarray
        3D array of shape (n_frames, ``height``, ``width``) holding the raw or current
        data.

    pixel_size_nm : float
        Physical pixel size in nanometers.

    channel : str
        Channel name.

    file_path : Path
        Path to the source file or folder.

    frame_metadata : list[dict[str, Any]]
        Per-frame metadata dicts; each will include a normalized 'timestamp' key.

    processed : dict[str, np.ndarray]
        Snapshots of processed data arrays from filters. Keys like
        'step_1_remove_plane'.

    masks : dict[str, np.ndarray]
        Boolean mask arrays from mask steps. Keys like 'step_2_threshold'.

    analysis : dict[str, Any]
        Results of analysis modules, keyed by 'step_<i>_<module_name>'.

    provenance : dict[str, Any]
        Records environment info and provenance of processing and analysis pipelines::

            {
                "environment": {...},
                "processing": {"steps": [...], "keys_by_name": {...}},
                "analysis": {"frame_times": [...], "steps": [...],
                "results_by_name": {...}},
            }
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
        Initialize AFMImageStack with data, spatial metadata, and per-frame metadata.

        Parameters
        ----------
        data : np.ndarray
            3D array of shape (n_frames, ``height``, ``width``) containing AFM image
            stack.

        pixel_size_nm : float
            Pixel size in nanometers; must be positive.

        channel : str
            Channel name (e.g., 'height_trace').

        file_path : Path
            Source file or folder path.

        frame_metadata : list of dict, optional
            List of per-frame metadata dicts. Will be padded or trimmed to length
            n_frames. After initialization, each entry is normalized to include a
            numeric 'timestamp' (fallback to frame index if missing).

        Raises
        ------
        TypeError
            If data is not an np.ndarray.

        ValueError
            If data.ndim != 3 or pixel_size_nm <= 0, or metadata length mismatch.
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
        # Stores analysis results by name
        self.analysis: dict[str, np.ndarray] = {}
        # Stores provenance information for the processing and analysis
        # environments and pipelines.
        self.provenance: dict[str, Any] = {
            "environment": {},  # to be filled when pipelines run
            "processing": {"steps": [], "keys_by_name": {}},
            "analysis": {"frame_times": None, "steps": [], "results_by_name": {}},
        }
        self.state_backups: dict[str, Any] = {}

    def _resolve_step(self, step: str) -> tuple[str, callable]:
        """
        Determine the type of a step and return its callable (or None for 'clear').

        Resolution order:
          1. "clear"
          2. Mask from MASK_MAP
          3. Bound method on this AFMImageStack instance
          4. Plugin from entry points "playNano.filters"
          5. Filter from FILTER_MAP
          6. Video filter from VIDEO_FILTER_MAP
          7. Stack edit function from STEP_EDIT_MAP (only ``drop_frames`` actually edits
             the stack, the other funcitons return lists of indices to be passed to
             ``drop_frames`` - this is done within the ProcesssingPipeline)

        Parameters
        ----------
        step : str
            Name of the processing step (e.g., 'remove_plane', 'threshold_mask',
            'clear').

        Returns
        -------
        tuple[str, callable | None]
            - step type: one of "clear", "mask", "method", "plugin", "filter",
              "video_filter", "stack_edit"
            - callable implementing the step, or None if step_type == "clear".

        Raises
        ------
        ValueError
            If the step name is not recognized among clear, masks, methods,
            plugins, or filters.
        """
        # 1) Clear existing mask?
        if step == "clear":
            return "clear", None

        # 2) Mask generator?
        if step in MASK_MAP:
            return "mask", MASK_MAP[step]

        # 3) Bound method on self? (e.g. a custom method on AFMImageStack)
        method = getattr(self, step, None)
        if callable(method):
            return "method", method

        # 4) Plugin filter entry point?
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

        # 5) Unmasked filter in FILTER_MAP?
        if step in FILTER_MAP:
            return "filter", FILTER_MAP[step]

        # 6) Video processing step in VIDEO_FILTER_MAP?
        if step in VIDEO_FILTER_MAP:
            return "video_filter", VIDEO_FILTER_MAP[step]

        # 7) Stack edit step ie. drop_frames?
        if step in STACK_EDIT_MAP:
            return "stack_edit", STACK_EDIT_MAP[step]

        # 8) No match
        raise ValueError(
            f"Unrecognized step '{step}'. "
            f"Available masks: {list(MASK_MAP)}; "
            f"built-in filters: {list(FILTER_MAP)}; "
            f"video filters: {list(VIDEO_FILTER_MAP)}; "
            f"methods: {[m for m in dir(self) if callable(getattr(self,m))]}; "
            f"plugins: {[ep.name for ep in metadata.entry_points(group='playNano.filters')]}."  # noqa
            f"stack_edit: {list(STACK_EDIT_MAP)}; "
        )

    def _execute_mask_step(
        self, mask_fn: callable, arr: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Compute a boolean mask array by applying mask_fn to each frame.

        Parameters
        ----------
        mask_fn : callable
            Function taking a 2D array (frame) and returning a boolean 2D mask.
        arr : np.ndarray
            3D array of shape (n_frames, H, W) to mask.
        **kwargs
            Additional parameters forwarded to mask_fn.

        Returns
        -------
        np.ndarray
            Boolean array of shape (n_frames, H, W). If mask_fn fails on a frame,
            that frame's mask is all False (and an error is logged).
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
        Apply a filter function to each frame, optionally using a mask.

        If mask is not None and step_name is in MASK_FILTERS_MAP, applies the masked
        version (takes frame and mask). Otherwise applies filter_fn(frame).

        Parameters
        ----------
        filter_fn : callable
            Function taking a 2D array (and optionally mask) and returning a 2D array.
        arr : np.ndarray
            3D array of shape (n_frames, H, W) to filter.
        mask : np.ndarray or None
            Boolean mask array of same shape, or None.
        step_name : str
            Name of this step, used to look up masked version in MASK_FILTERS_MAP.
        **kwargs
            Additional parameters forwarded to filter function.

        Returns
        -------
        np.ndarray
            New 3D array of same shape as arr. If filtering fails on a frame,
            original frame is kept (and a warning/error is logged).
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

    def _execute_video_processing_step(
        self, video_fn: callable, arr: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Execute a video processing function on a full image stack.

        This method applies a given callable to a 3D array representing
        an image stack of shape ``(n_frames, height, width)``. It safely
        handles errors by catching exceptions and returning the original
        array if the function fails.

        Parameters
        ----------
        video_fn : callable
            A function that accepts a 3D NumPy array and optionally keyword arguments,
            and returns a new 3D array of processed frames. The callable should have
            a signature such as:

            ``video_fn(arr: np.ndarray, **kwargs) -> np.ndarray``
        arr : np.ndarray
            Input 3D image stack with shape ``(n_frames, height, width)``.
        **kwargs : dict, optional
            Additional keyword arguments passed to ``video_fn``.

        Returns
        -------
        np.ndarray
            Processed 3D image stack. If ``video_fn`` raises an exception,
            the original array ``arr`` is returned unchanged.

        Notes
        -----
        - This method is typically used internally by the processing pipeline
        to apply a uniform operation (e.g., flattening, filtering,
        background subtraction) to all frames in a video stack.
        - If the provided function does not accept keyword arguments, the method
        will attempt to call it without them before failing.
        - All errors are logged via the module logger with the name of the
        failing function.
        """
        try:
            return video_fn(arr, **kwargs)
        except TypeError:
            try:
                return video_fn(arr)
            except Exception as e:
                logger.warning(
                    f"Video processing step '{video_fn.__name__}' failed: {e}"
                )
                return arr
        except Exception as e:
            logger.warning(f"Video processing step '{video_fn.__name__}' failed: {e}")
            return arr

    def _execute_stack_edit_step(self, fn, arr: np.ndarray, **kwargs) -> np.ndarray:
        """
        Execute a structural edit operation on the AFMImageStack.

        This method should not be used directly outside of ProcessingPipeline.
        This internal method applies a function that modifies the structure of the stack
        (for example, dropping or reordering frames) and ensures that the associated
        frame metadata remains synchronized with the resulting array.

        The previous frame metadata is automatically backed up in
        ``self.state_backups["frame_metadata_before_edit"]`` before any modification,
        allowing later restoration through :meth:`restore_frame_metadata`.

        Parameters
        ----------
        fn : callable
            A function that performs the edit operation. Must accept the current 3D
            image stack (``arr``) as its first argument and return a modified
            3D NumPy array with shape ``(n', h, w)``.
        arr : numpy.ndarray
            The 3D input array representing the current stack data, with shape
            ``(n, height, width)``.
        **kwargs : dict
            Additional keyword arguments to pass to the edit function.
            Typically includes:

            - ``indices`` : list of int, optional
              The frame indices that were removed. If provided, the corresponding
              entries in ``self.frame_metadata`` are also dropped.

        Returns
        -------
        numpy.ndarray
            The modified 3D array after the edit has been applied.

        Raises
        ------
        RuntimeError
            If the length of ``self.frame_metadata`` does not match the number
            of frames in the returned array after editing.
        Exception
            Propagates any exception raised by the edit function ``fn``.

        Notes
        -----
        - The previous metadata state is only backed up once per edit sequence
          to prevent redundant copies.
        - This method does **not** record provenance directly; provenance
          tracking is handled by the :class:`ProcessingPipeline`.
        - Structural edits should only modify the number or ordering of frames,
          not the spatial dimensions or data type.

        See Also
        --------
        drop_frames : Remove specific frames from a stack.
        restore_frame_metadata : Restore original frame metadata after edits.
        ProcessingPipeline._execute_filter_step : Execute a standard filter step.
        """
        # Run the edit function to get the new 3D array
        new_arr = fn(arr, **kwargs)

        # Only backup once per edit series to avoid redundant copies
        if "frame_metadata_before_edit" not in self.state_backups:
            self.state_backups["frame_metadata_before_edit"] = list(self.frame_metadata)

        # Update frame metadata by removing corresponding entries
        dropped_indices = kwargs.get("indices_to_drop", [])
        if dropped_indices:
            self.frame_metadata = [
                meta
                for i, meta in enumerate(self.frame_metadata)
                if i not in dropped_indices
            ]

        # Optional: sanity check
        if len(self.frame_metadata) != new_arr.shape[0]:
            raise RuntimeError(
                f"frame_metadata length mismatch after edit "
                f"({len(self.frame_metadata)} vs {new_arr.shape[0]})."
            )
        return new_arr

    def restore_frame_metadata(self):
        """Restore frame metadata from the last state backup, if available."""
        backup = self.state_backups.pop("frame_metadata_before_edit", None)
        if backup is not None:
            self.frame_metadata = backup
            logger.info("Frame metadata restored from state_backups.")
        else:
            logger.debug("No frame metadata backup found.")

    @classmethod
    def load_data(
        cls, path: str | Path, channel: str = "height_trace"
    ) -> AFMImageStack:
        """
        Load AFM data from a file or folder into AFMImageStack, normalizing timestamps.

        Parameters
        ----------
        path : str or Path
            Path to AFM data file or folder.
        channel : str, optional
            Channel name to load; default "height_trace".

        Returns
        -------
        AFMImageStack
            Fully reconstructed AFMImageStack with processed snapshots and provenance.
        """
        from playNano.io.loader import load_afm_stack

        afm = load_afm_stack(path, channel)
        afm.frame_metadata = normalize_timestamps(afm.frame_metadata)
        afm.provenance.setdefault("processing", {"steps": [], "keys_by_name": {}})

        return afm

    @property
    def n_frames(self) -> int:
        """
        Number of frames in the stack.

        Returns
        -------
        int
            Number of frames (size along axis 0).
        """
        return self.data.shape[0]

    @property
    def height(self) -> int:
        """
        Get the number of pixel rows (frame height).

        Returns
        -------
        int
            Number of rows per frame.
        """
        return self.data.shape[1]

    @property
    def width(self) -> int:
        """
        Get the frame number of pixel columns.

        Returns
        -------
        int
            Number of columns per frame.
        """
        return self.data.shape[2]

    @property
    def image_shape(self) -> tuple[int, int]:
        """
        Spatial dimensions of a single frame.

        Returns
        -------
        tuple of (``height``, ``width``)
        """
        return self.data.shape[1:]

    def get_frame(self, index: int) -> np.ndarray:
        """
        Retrieve the 2D image array for a specific frame index.

        Parameters
        ----------
        index : int
            Frame index to retrieve.

        Returns
        -------
        np.ndarray
            2D array of shape (``height``, ``width``) for the frame.

        Raises
        ------
        IndexError
            If index is out of bounds.
        """
        return self.data[index]

    def get_frame_metadata(self, index: int) -> dict[str, Any]:
        """
        Retrieve metadata dict for a specific frame index.

        Parameters
        ----------
        index : int
            Frame index.

        Returns
        -------
        dict[str, Any]
            Metadata dict for that frame (includes normalized 'timestamp').

        Raises
        ------
        IndexError
            If index is out of range.
        """
        if 0 <= index < len(self.frame_metadata):
            return self.frame_metadata[index]
        else:
            raise IndexError(f"Frame metadata index {index} out of range")

    def get_frames(self) -> list[np.ndarray]:
        """
        Return a list of all individual 2D frame arrays.

        Returns
        -------
        list of np.ndarray
            List of length n_frames, each a 2D array.
        """
        return [self.get_frame(i) for i in range(self.n_frames)]

    def frames_with_metadata(self):
        """
        Yield tuples of (index, frame_array, metadata) for all frames.

        Yields
        ------
        tuple[int, np.ndarray, dict[str, Any]]
            Frame index, 2D image array, and metadata dict.

        Notes
        -----
        If any frame array is None, it is skipped with a warning log.
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
        Index or slice the stack.

        Parameters
        ----------
        idx : int or slice
            - If int: return 2D array for that frame.
            - If slice: return new AFMImageStack with subset of frames.

        Returns
        -------
        np.ndarray or AFMImageStack
            2D array for int index, or new AFMImageStack for slice.

        Raises
        ------
        TypeError
            If idx is neither int nor slice.
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
        Export processing provenance and environment metadata to a JSON file.

        Writes
        ------

        Example JSON structure::

            {
            "environment": { ... },
            "processing": {
                "steps": [ ... ],
                "keys_by_name": { ... }
            }
            }

        Parameters
        ----------
        path : str
            File path to write the JSON log. Creates parent dirs as needed.
        """
        from playNano.analysis.utils.common import NumpyEncoder

        record = {
            "environment": self.stack.provenance.get("environment", {}),
            "processing": self.stack.provenance.get("processing", {}),
        }

        dir = os.path.dirname(path)
        if dir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(record, f, indent=2, cls=NumpyEncoder)

    def export_analysis_log(self, path: str) -> None:
        """
        Export analysis provenance and results to a JSON file.

        Expects that stack.analysis (or stack.analysis_results) and
        stack.provenance["analysis"] are populated by AnalysisPipeline.run().

        Writes
        ------
        Example JSON structure::

            {
            "environment": { ... },
            "analysis": { <step_key>: outputs, ... },
            "provenance": {
                "steps": [ ... ],
                "results_by_name": { ... },
                "frame_times": [...],
            }
            }

        Parameters
        ----------
        path : str
            File path to write the JSON log. Creates parent dirs as needed.

        Raises
        ------
        ValueError
            If no analysis results found on this stack.
        """
        from playNano.analysis.utils.common import NumpyEncoder

        if not hasattr(self, "analysis_results") or not self.analysis_results:
            raise ValueError(
                "No analysis results found. Run an AnalysisPipeline first."
            )

        dir = os.path.dirname(path)
        if dir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.analysis_results, f, indent=2, cls=NumpyEncoder)

    def _load_plugin(self, name: str):
        """
        Load a filter plugin dynamically from entry points under "playNano.filters".

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

    def _get_plugin_version(fn: callable) -> str | None:
        """
        Attempt to obtain a version string for the package/module defining fn.

        Parameters
        ----------
        fn : callable
            Function object whose module/package version to query.

        Returns
        -------
        str or None
            Version string if retrievable via importlib.metadata, else None.
        """
        module_name = fn.__module__.split(".")[0]
        try:
            return metadata.version(module_name)
        except metadata.PackageNotFoundError:
            return None
        except Exception:
            return None

    def apply(self, steps: list[str], **kwargs) -> np.ndarray:
        """
        Apply a sequence of processing steps to each frame in the AFM image stack.

        Steps can be:

          - "clear"       : reset any existing mask
          - mask names    : keys in MASK_MAP
          - filter names  : keys in FILTER_MAP
          - plugin names  : entry points in 'playNano.filters'
          - method names  : bound methods on this class

        ``**kwargs`` are forwarded to mask functions or filter functions as appropriate.

        This is a stateless convenience: applies clear/mask/filter steps in order,
        snapshots only 'raw' and final data in self.processed, but does not assign
        unique keys per step or update provenance.

        Parameters
        ----------
        steps : list of str
            Sequence of step names (e.g. ["remove_plane", "threshold_mask",
            "gaussian_filter"]).
        **kwargs
            Forwarded to each mask/filter function.

        Returns
        -------
        np.ndarray
            Final processed 3D array (shape (n_frames, ``height``, ``width``)).

        Notes
        -----
        For tracked, reproducible processing,
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
        Get timestamp for a given frame index, fallback to index if missing.

        Parameters
        ----------
        idx : int
            Frame index.

        Returns
        -------
        float
            Timestamp in seconds; if metadata lacks 'timestamp', returns float(idx).

        Raises
        ------
        IndexError
            If idx is out of range.

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
        list[float]
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
        # Restore frame metadata if it was backed up
        self.restore_frame_metadata()
        logger.debug(
            "Metadata restored from state_backups['frame_metadata_before_edit']."
        )
        return self.data
