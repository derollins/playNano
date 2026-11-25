"""
The :class:`AFMImageStack` class, which holds an AFM video and everything done to it.

The module also builds the lookup tables of built-in steps that
:meth:`AFMImageStack.apply` and the processing pipeline use: ``FILTER_MAP``,
``MASK_MAP``, ``MASK_FILTERS_MAP``, ``VIDEO_FILTER_MAP`` and ``STACK_EDIT_MAP``.
"""

from __future__ import annotations

import json
import logging
import os
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from playnano.processing import (
    filters,
    mask_generators,
    masked_filters,
    stack_edit,
    video_processing,
)
from playnano.utils.io_utils import FRAME_METADATA_KEYS
from playnano.utils.time_utils import normalize_timestamps

# Built-in filters and mask dictionaries
FILTER_MAP = filters.register_filters()
MASK_MAP = mask_generators.register_masking()
MASK_FILTERS_MAP = masked_filters.register_mask_filters()
VIDEO_FILTER_MAP = video_processing.register_video_processing()
STACK_EDIT_MAP = stack_edit.register_stack_edit_processing()

logger = logging.getLogger(__name__)


class AFMImageStack:
    """
    An AFM video: frames, pixel size, per-frame metadata, results and provenance.

    Usually made by :meth:`load_data` or
    :func:`playnano.io.loader.load_afm_stack`, and changed by
    :class:`playnano.processing.pipeline.ProcessingPipeline` and
    :class:`playnano.analysis.pipeline.AnalysisPipeline`.

    Parameters
    ----------
    data : numpy.ndarray
        Frames, shape (n_frames, height, width), heights in nm. Stored as given,
        without copying.
    pixel_size_nm : int or float
        Pixel size in nm, used when a frame has no ``frame_pixel_size_nm``. Must
        be a positive Python ``int`` or ``float``: NumPy numbers are rejected.
    channel : str
        Channel name, for example ``"height_trace"``.
    file_path : str or pathlib.Path
        File or folder the data came from. Stored as given; used for default
        output names.
    frame_metadata : list of dict, optional
        One dict per frame, with keys from ``FRAME_METADATA_KEYS`` in
        :mod:`playnano.utils.io_utils`; other keys give a warning. A shorter
        list is padded with empty dicts. Timestamps are converted with
        :func:`playnano.utils.time_utils.normalize_timestamps`, and a frame
        without one gets ``"timestamp": None``. Default is None (no metadata).

    Attributes
    ----------
    data : numpy.ndarray
        The current frames, shape (n_frames, height, width). Processing
        replaces this array.
    pixel_size_nm : float
        Pixel size in nm for frames without their own.
    channel : str
        Channel name.
    file_path : str or pathlib.Path
        File or folder the data came from.
    frame_metadata : list of dict
        One dict per frame. Every dict has ``"timestamp"`` (s, or None); files
        loaded by playNano also have the other ``FRAME_METADATA_KEYS``.
    processed : dict
        Copies of the data, as arrays: ``"raw"`` before the first step, then
        one per step, keyed ``"step_<n>_<name>"`` by the processing pipeline
        or by step name by :meth:`apply`.
    masks : dict
        Boolean mask arrays from the processing pipeline, keyed
        ``"step_<n>_<name>"``.
    analysis : dict
        Analysis results, keyed ``"step_<n>_<module name>"``.
    provenance : dict
        What was done and with what: ``"environment"`` (software versions),
        ``"acquisition"`` (see :attr:`acquisition`), ``"processing"`` (steps and
        their keys) and ``"analysis"`` (steps, results by name, frame times).
    state_backups : dict
        Data kept so changes can be undone: ``"frame_metadata_before_edit"``
        holds the frame metadata from before frames were removed.

    Raises
    ------
    TypeError
        If ``data`` is not a NumPy array.
    ValueError
        If ``data`` is not 3D, ``pixel_size_nm`` is not a positive int or
        float, or ``frame_metadata`` has more entries than there are frames.

    Examples
    --------
    >>> import numpy as np
    >>> from playnano.afm_stack import AFMImageStack
    >>> data = np.zeros((3, 64, 64))
    >>> stack = AFMImageStack(
    ...     data,
    ...     pixel_size_nm=2.0,
    ...     channel="height_trace",
    ...     file_path="demo.h5-jpk",
    ...     frame_metadata=[{"timestamp": 0.0}, {"timestamp": 0.5}],
    ... )
    >>> stack.n_frames, stack.image_shape
    (3, (64, 64))
    >>> stack.get_frame_times()
    [0.0, 0.5, 2.0]
    """

    def __init__(
        self,
        data: np.ndarray,
        pixel_size_nm: float,
        channel: str,
        file_path: Path,
        frame_metadata: list[dict[str, Any]] = None,
    ) -> None:
        """Create the stack. The parameters are described in the class docstring."""
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

        # Check only known keys are present in frame_metadata
        _known = set(FRAME_METADATA_KEYS)
        for i, md in enumerate(self.frame_metadata):
            unknown = set(md) - _known
            if unknown:
                logger.warning("frame_metadata[%d] has unknown keys: %s", i, unknown)

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
            "acquisition": {},
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
          4. Plugin from entry points "playnano.filters"
          5. Plugin from entry points "playnano.video_processing"
          6. Filter from FILTER_MAP
          7. Video filter from VIDEO_FILTER_MAP
          8. Stack edit function from STACK_EDIT_MAP (only ``drop_frames``
             actually edits the stack, the other functions return lists of
             indices to be passed to ``drop_frames`` - this is done within the
             ProcessingPipeline)

        Parameters
        ----------
        step : str
            Name of the processing step (e.g., 'remove_plane', 'mask_threshold',
            'clear').

        Returns
        -------
        tuple[str, callable | None]
            - step type: one of "clear", "mask", "method", "plugin", "filter",
              "video_filter", "video_plugin", "stack_edit"
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
                for ep in metadata.entry_points(group="playnano.filters")
                if ep.name == step
            )
        except StopIteration:
            ep = None
        if ep is not None:
            fn = ep.load()
            return "plugin", fn

        # 5) Plugin 3D filter 'video processing'entry point?
        try:
            ep = next(
                ep
                for ep in metadata.entry_points(group="playnano.video_processing")
                if ep.name == step
            )
        except StopIteration:
            ep = None
        if ep is not None:
            fn = ep.load()
            return "video_plugin", fn

        # 6) Unmasked filter in FILTER_MAP?
        if step in FILTER_MAP:
            return "filter", FILTER_MAP[step]

        # 7) Video processing step in VIDEO_FILTER_MAP?
        if step in VIDEO_FILTER_MAP:
            return "video_filter", VIDEO_FILTER_MAP[step]

        # 8) Stack edit step ie. drop_frames?
        if step in STACK_EDIT_MAP:
            return "stack_edit", STACK_EDIT_MAP[step]

        # 9) No match
        raise ValueError(
            f"Unrecognized step '{step}'. "
            f"Available masks: {list(MASK_MAP)}; "
            f"built-in filters: {list(FILTER_MAP)}; "
            f"video filters: {list(VIDEO_FILTER_MAP)}; "
            f"methods: {[m for m in dir(self) if callable(getattr(self, m))]}; "
            f"plugins: {[ep.name for ep in metadata.entry_points(group='playnano.filters')]}."  # noqa
            f"video_plugins: {[ep.name for ep in metadata.entry_points(group='playnano.video_processing')]}."  # noqa
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
            logger.debug(f"Mask present and masked {step_name} filter selected.")
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
            logger.debug(f"{step_name} filter applied without mask.")
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
        """
        Put back the frame metadata saved before frames were removed.

        Takes it out of ``state_backups["frame_metadata_before_edit"]``. Does
        nothing if there is none. :attr:`data` is not changed; see
        :meth:`restore_raw`.
        """
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
        Load a stack from an AFM file, a folder of frames, or a saved playNano file.

        Calls :func:`playnano.io.loader.load_afm_stack`, which picks the reader
        from the file extension.

        Parameters
        ----------
        path : str or pathlib.Path
            File or folder to load.
        channel : str, optional
            Channel to read. Ignored for files saved by playNano, which keep their
            own channel. Default is ``"height_trace"``.

        Returns
        -------
        AFMImageStack
            The stack. Files saved by playNano also restore ``processed``,
            ``masks`` and ``provenance``.

        Raises
        ------
        FileNotFoundError
            If the path does not exist, or a folder holds no known file type.
        ValueError
            If the file type is not supported.
        """
        from playnano.io.loader import load_afm_stack

        afm = load_afm_stack(path, channel)
        afm.frame_metadata = normalize_timestamps(afm.frame_metadata)
        afm.provenance.setdefault("processing", {"steps": [], "keys_by_name": {}})

        return afm

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return self.data.shape[0]

    @property
    def height(self) -> int:
        """Height of each frame, in pixels (rows)."""
        return self.data.shape[1]

    @property
    def width(self) -> int:
        """Width of each frame, in pixels (columns)."""
        return self.data.shape[2]

    @property
    def image_shape(self) -> tuple[int, int]:
        """Shape of each frame, ``(height, width)``, in pixels."""
        return self.data.shape[1:]

    def get_frame(self, index: int) -> np.ndarray:
        """
        Return one frame.

        Parameters
        ----------
        index : int
            Frame index; negative values count from the end.

        Returns
        -------
        numpy.ndarray
            The frame, shape (height, width). It is a view of :attr:`data`, so
            changing it changes the stack.

        Raises
        ------
        IndexError
            If ``index`` is out of range.
        """
        return self.data[index]

    def get_frame_metadata(self, index: int) -> dict[str, Any]:
        """
        Return one frame's metadata.

        Parameters
        ----------
        index : int
            Frame index, from 0 to ``n_frames - 1``. Negative values are not
            accepted.

        Returns
        -------
        dict
            The frame's metadata dict itself, not a copy.

        Raises
        ------
        IndexError
            If ``index`` is out of range.
        """
        if 0 <= index < len(self.frame_metadata):
            return self.frame_metadata[index]
        else:
            raise IndexError(f"Frame metadata index {index} out of range")

    def get_frames(self) -> list[np.ndarray]:
        """
        Return all frames as a list.

        Returns
        -------
        list of numpy.ndarray
            One 2D array per frame; each is a view of :attr:`data`.
        """
        return [self.get_frame(i) for i in range(self.n_frames)]

    def frames_with_metadata(self):
        """
        Iterate over the frames with their index and metadata.

        Yields
        ------
        tuple of (int, numpy.ndarray, dict)
            Frame index, frame (a view of :attr:`data`) and metadata dict.
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
        Return a frame (``stack[i]``) or a new stack of some frames (``stack[a:b]``).

        Parameters
        ----------
        idx : int or slice
            A Python ``int`` gives that frame; NumPy integers are not accepted. A
            slice gives a new stack with those frames and their metadata, the same
            pixel size, channel and file path, and no processed data, masks,
            results or provenance. Its data is a view of this stack's data.

        Returns
        -------
        numpy.ndarray or AFMImageStack
            The frame, or the new stack.

        Raises
        ------
        TypeError
            If ``idx`` is neither an ``int`` nor a slice.
        IndexError
            If an ``int`` index is out of range.
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
        Save the environment and processing provenance as a JSON file.

        The file holds ``{"environment": ..., "processing": {"steps": ...,
        "keys_by_name": ...}}`` from :attr:`provenance`.

        Parameters
        ----------
        path : str or pathlib.Path
            File to write. Its folder is created if missing; an existing file is
            overwritten.
        """
        from playnano.analysis.utils.common import NumpyEncoder

        record = {
            "environment": self.provenance.get("environment", {}),
            "processing": self.provenance.get("processing", {}),
        }

        dir = os.path.dirname(path)
        if dir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(record, f, indent=2, cls=NumpyEncoder)

    def export_analysis_log(self, path: str) -> None:
        """
        Save the analysis results as a JSON file. Currently always raises an error.

        It looks for results in an ``analysis_results`` attribute, which nothing in
        playNano sets, so it raises ``ValueError`` even after an analysis has run.
        To save results, pass the record returned by
        :meth:`playnano.analysis.pipeline.AnalysisPipeline.run` to
        :func:`playnano.analysis.export.export_analysis_to_json`.

        Parameters
        ----------
        path : str or pathlib.Path
            File to write. Its folder is created if missing.

        Raises
        ------
        ValueError
            If the stack has no ``analysis_results`` attribute, or it is empty.
        """
        from playnano.analysis.utils.common import NumpyEncoder

        if not hasattr(self, "analysis_results") or not self.analysis_results:
            raise ValueError(
                "No analysis results found. Run an AnalysisPipeline first."
            )

        dir = os.path.dirname(path)
        if dir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.analysis_results, f, indent=2, cls=NumpyEncoder)

    def apply(self, steps: list[str], **kwargs) -> np.ndarray:
        """
        Apply processing steps by name, without recording provenance.

        A quick way to process a stack interactively. For processing you can
        repeat and report, use
        :class:`playnano.processing.pipeline.ProcessingPipeline`.

        Each name is looked up in this order:

        1. ``"clear"``: forget the current mask;
        2. a mask generator (``MASK_MAP``), which sets the mask for later
           filters;
        3. a method of this class;
        4. a plugin registered under ``playnano.filters`` or
           ``playnano.video_processing``;
        5. a built-in filter (``FILTER_MAP``); if a mask is set and the filter
           has a masked version (``MASK_FILTERS_MAP``), that version is used;
        6. a video filter (``VIDEO_FILTER_MAP``);
        7. a stack edit (``STACK_EDIT_MAP``), which also clears the mask.

        Parameters
        ----------
        steps : list of str
            Step names, applied in order, for example
            ``["remove_plane", "mask_threshold", "row_median_align"]``.
        **kwargs
            Passed to every step. A step that does not accept them is called
            without any.

        Returns
        -------
        numpy.ndarray
            The processed data, also stored in :attr:`data`.

        Raises
        ------
        ValueError
            If a step name is not found.

        Notes
        -----
        :attr:`processed` gets ``"raw"`` (if not already there) and a copy of the
        data after each step except ``"clear"`` and masks, keyed by step name, so a
        repeated step replaces its earlier copy. Masks are not stored. If a filter
        fails on a frame, that frame is left unchanged and a warning is logged; if
        a video filter fails, the step is skipped.

        Examples
        --------
        >>> import numpy as np
        >>> from playnano.afm_stack import AFMImageStack
        >>> yy, xx = np.mgrid[0:32, 0:32]
        >>> data = np.stack([0.1 * xx + 0.05 * yy] * 2).astype(float)
        >>> stack = AFMImageStack(data, 1.0, "height_trace", "demo.h5-jpk")
        >>> out = stack.apply(["remove_plane"])
        >>> sorted(stack.processed), bool(np.allclose(out, 0))
        (['raw', 'remove_plane'], True)
        """
        # 1) Snapshot raw data once
        if "raw" not in self.processed:
            self.processed["raw"] = self.data.copy()

        arr = self.data
        mask = None

        for step in steps:
            step_type, fn = self._resolve_step(step)

            # (A) CLEAR
            if step_type == "clear":
                logger.info("Step 'clear' → dropping existing mask.")
                mask = None
                continue

            # (B) MASK GENERATOR
            elif step_type == "mask":
                logger.info(f"Step '{step}' → computing new mask.")
                mask = self._execute_mask_step(fn, arr, **kwargs)
                continue  # masks don't modify arr

            # (C) FILTER OR PLUGIN (2D/frame)
            elif step_type in ("filter", "plugin"):
                logger.info(f"Step '{step}' (filter) → applying per-frame.")
                arr = self._execute_filter_step(fn, arr, mask, step, **kwargs)

            # (D) VIDEO FILTER / VIDEO PLUGIN (3D)
            elif step_type in ("video_filter", "video_plugin"):
                logger.info(f"Step '{step}' (video filter) → applying to full stack.")
                arr = self._execute_video_processing_step(fn, arr, **kwargs)

            # (E) STACK EDIT
            elif step_type == "stack_edit":
                logger.info(f"Step '{step}' (stack edit) → applying to full stack.")
                arr = self._execute_stack_edit_step(fn, arr, **kwargs)
                mask = None  # masks invalid after structural change

            # (F) METHOD
            elif step_type == "method":
                logger.info(f"Step '{step}' (method) → applying.")
                try:
                    new_arr = fn(arr, **kwargs)
                except TypeError:
                    new_arr = fn(**kwargs)
                if new_arr is None:
                    new_arr = getattr(self, "data", arr)
                arr = new_arr

            # (G) UNRECOGNIZED - should not happen due to _resolve_step
            else:
                raise ValueError(f"Unsupported step type '{step_type}'.")

            # Store a snapshot in processed dict
            self.processed[step] = arr.copy()

        # 5) Save final output
        self.data = arr
        return arr

    def time_for_frame(self, idx: int) -> float:
        """
        Return a frame's time in seconds, or its index if it has no time.

        Parameters
        ----------
        idx : int
            Frame index.

        Returns
        -------
        float
            The frame's ``"timestamp"``, or ``float(idx)`` if it is missing or
            None.

        Raises
        ------
        IndexError
            If ``idx`` is out of range.

        Examples
        --------
        >>> import numpy as np
        >>> from playnano.afm_stack import AFMImageStack
        >>> stack = AFMImageStack(
        ...     np.zeros((3, 4, 4)),
        ...     1.0,
        ...     "height_trace",
        ...     "demo.h5-jpk",
        ...     frame_metadata=[{"timestamp": 0.0}, {}, {"timestamp": 0.4}],
        ... )
        >>> stack.time_for_frame(1), stack.time_for_frame(2)
        (1.0, 0.4)
        """
        ts = self.frame_metadata[idx].get("timestamp", None)
        return float(idx) if ts is None else ts

    def scaling_for_frame(self, idx: int) -> float:
        """
        Return a frame's pixel size in nm.

        Parameters
        ----------
        idx : int
            Frame index.

        Returns
        -------
        float
            The frame's ``"frame_pixel_size_nm"``, or :attr:`pixel_size_nm` if the
            frame has no such key. A key set to None returns None.

        Raises
        ------
        IndexError
            If ``idx`` is out of range.

        Examples
        --------
        >>> import numpy as np
        >>> from playnano.afm_stack import AFMImageStack
        >>> stack = AFMImageStack(
        ...     np.zeros((2, 4, 4)),
        ...     1.0,
        ...     "height_trace",
        ...     "demo.h5-jpk",
        ...     frame_metadata=[{"frame_pixel_size_nm": 2.0}, {}],
        ... )
        >>> stack.scaling_for_frame(0), stack.scaling_for_frame(1)
        (2.0, 1.0)
        """
        return self.frame_metadata[idx].get("frame_pixel_size_nm", self.pixel_size_nm)

    def get_frame_times(self) -> list[float]:
        """
        Return every frame's time in seconds.

        Uses :meth:`time_for_frame`, so a frame without a time gets its index.

        Returns
        -------
        list of float
            One time per frame.

        Examples
        --------
        >>> import numpy as np
        >>> from playnano.afm_stack import AFMImageStack
        >>> stack = AFMImageStack(
        ...     np.zeros((3, 4, 4)),
        ...     1.0,
        ...     "height_trace",
        ...     "demo.h5-jpk",
        ...     frame_metadata=[{"timestamp": 0.0}, {"timestamp": 0.5}],
        ... )
        >>> stack.get_frame_times()
        [0.0, 0.5, 2.0]
        """
        return [self.time_for_frame(i) for i in range(len(self.frame_metadata))]

    def channel_for_frame(self, idx: int) -> str:
        """
        Return a frame's channel name.

        Parameters
        ----------
        idx : int
            Frame index.

        Returns
        -------
        str
            The frame's ``"channel"`` metadata if it has one, otherwise
            :attr:`channel`.

        Raises
        ------
        IndexError
            If ``idx`` is out of range.
        """
        return self.frame_metadata[idx].get("channel", self.channel)

    @property
    def acquisition(self) -> dict[str, Any]:
        """
        Settings of the whole recording, from the file's header.

        For example whether the scan was bidirectional, or the file version,
        depending on the format; empty if the reader found none. Stored in
        ``provenance["acquisition"]``, and kept when frames are removed.
        """
        return self.provenance.setdefault("acquisition", {})

    def restore_raw(self) -> np.ndarray:
        """
        Restore the original data by loading from the 'raw' snapshot in self.processed.

        Copies ``processed["raw"]`` into :attr:`data` and calls
        :meth:`restore_frame_metadata`. :attr:`processed`, :attr:`masks` and
        :attr:`provenance` are not changed.

        Returns
        -------
        numpy.ndarray
            The restored data.

        Raises
        ------
        KeyError
            If there is no ``processed["raw"]``.
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
